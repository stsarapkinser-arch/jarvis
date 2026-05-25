from __future__ import annotations

import asyncio
import logging
import random
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import ollama

from commands import CommandBook
from event_bus import Event, EventBus, EventType
from kwin import KWinOrchestrator
from memory_engine import ChronoMemory
from parser import (
    ParsedResponse, clean_bash, inject_sudo, parse_response, run_bash, wrap_sandbox,
)
from repair import QuickPatcher
from singleton import Singleton
from state_snapshot import StateSnapshot, format_snapshot, snapshot

log = logging.getLogger("jarvis.core")

PIPER_PATH = "./piper/piper"
VOICE_MODEL = "./piper/ru_RU-dmitry-medium.onnx"
SYSTEM_PROMPT_FILE = "./system_prompt"
MODEL = "qwen2.5-coder:3b"

KWIN_MACRO_PREFIX = "__kwin__:"
MAX_HEAL_ATTEMPTS = 3
RECENT_OS_EVENTS_MAX = 5
CALL_FRESH_WINDOW_SEC = 300
CLIPBOARD_FRESH_WINDOW_SEC = 600
CONFIRMATION_TIMEOUT_SEC = 30.0

TONE_PRESETS: dict[str, tuple[float, float]] = {
    "alert":  (0.85, 0.10),
    "normal": (1.00, 0.20),
    "idle":   (1.12, 0.40),
}

ACK_VARIANTS: dict[str, tuple[str, ...]] = {
    "ok": ("Готово.", "Сделано.", "Принято.", "Выполнено.", "Есть."),
    "fail": ("Не удалось.", "Не получилось.", "Ошибка."),
    "confirm": ("Подтверждено. Выполняю.", "Принял. Действую.", "Есть, выполняю."),
    "cancel": ("Отменено.", "Отбой.", "Отменил."),
}

AFFIRMATIVE_RE = re.compile(
    r"\b(да|конечно|подтверждаю|подтвердить|выполни(?:ть)?|давай|ок|окей|yes|confirm|do\s+it|go\s+ahead)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(нет|отмени(?:ть)?|стоп|остановить|cancel|no|stop|abort|don'?t)\b",
    re.IGNORECASE,
)
DESTRUCTIVE_RE = re.compile(
    r"(\brm\s+-(?:r[fF]|fr|rf)\b"
    r"|\bapt(?:-get)?\s+(?:remove|purge|autoremove)\b"
    r"|\bdpkg\s+--purge\b"
    r"|\bmkfs(?:\.\w+)?\b"
    r"|\bdd\s+(?:[^|]*\s+)?of=/dev/"
    r"|\b(?:user|group)del\b"
    r"|(?:>|>>)\s*/dev/sd[a-z]"
    r"|\bshutdown\s+-h"
    r"|\bsystemctl\s+(?:poweroff|halt|reboot)"
    r"|\bdrop\s+(?:table|database)\b"
    r"|\bchmod\s+-R\s+(?:000|777)\s+/)",
    re.IGNORECASE,
)


class Jarvis(metaclass=Singleton):
    """Central async orchestrator. Subscribes to the event bus.
    Holds short-lived context state (recent OS events, last Pixel data)
    that feeds into every LLM prompt as a CONTEXT_BLOCK."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.memory = ChronoMemory()
        self.book = CommandBook()
        self.kwin = KWinOrchestrator()
        self.patcher = QuickPatcher()
        self.model = MODEL
        self.system_prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
        self._ollama = ollama.AsyncClient()

        self._recent_os: deque[dict] = deque(maxlen=RECENT_OS_EVENTS_MAX)
        self._last_clipboard: str = ""
        self._last_clipboard_ts: float = 0.0
        self._last_call: dict | None = None
        self._last_call_ts: float = 0.0

        self._pending_confirmation: dict | None = None
        self._state_label: str = "IDLE"
        self._last_error_window_hint: str = ""

    async def _state(self, state: str, thought: str = "") -> None:
        self._state_label = state
        await self.bus.publish(Event(EventType.STATE_CHANGE, (state, thought)))

    def _tone_for_state(self) -> str:
        if self._state_label == "ALERT":
            return "alert"
        if self._state_label == "IDLE":
            return "idle"
        return "normal"

    def ack(self, category: str) -> str:
        return random.choice(ACK_VARIANTS.get(category, ("ok",)))

    def say(self, text: str, tone: str | None = None) -> None:
        """Speak via Piper in a daemon thread. Tone adjusts length-scale (speed)
        and sentence-silence (pauses) per current Jarvis state."""
        text = (text or "").strip()
        if not text:
            return
        chosen = tone or self._tone_for_state()
        length_scale, sentence_silence = TONE_PRESETS.get(chosen, TONE_PRESETS["normal"])

        def _play() -> None:
            try:
                piper = subprocess.Popen(
                    [
                        PIPER_PATH, "--model", VOICE_MODEL, "--output_raw",
                        "--length-scale", f"{length_scale:.2f}",
                        "--sentence-silence", f"{sentence_silence:.2f}",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                aplay = subprocess.Popen(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                    stdin=piper.stdout,
                    stderr=subprocess.DEVNULL,
                )
                piper.stdout.close()
                assert piper.stdin is not None
                piper.stdin.write(text.encode("utf-8"))
                piper.stdin.close()
                aplay.wait()
            except Exception:
                log.exception("say() pipeline failed")

        threading.Thread(target=_play, daemon=True).start()

    def _build_context_block(self, snap: StateSnapshot | dict | None) -> str:
        lines: list[str] = []

        snap_view = format_snapshot(snap or {})
        if snap_view:
            lines.append(f"NOW: {snap_view}")

        if self._recent_os:
            recent = list(self._recent_os)[-3:]
            parts = []
            for e in recent:
                sensor = e.get("sensor", "?")
                level = e.get("level", "?")
                value = e.get("value")
                ago = int(time.time() - e.get("ts", time.time()))
                parts.append(f"{sensor}={value}({level},{ago}s_ago)")
            lines.append(f"SENTINEL_RECENT: {', '.join(parts)}")

        now = time.time()
        if self._last_clipboard and (now - self._last_clipboard_ts) < CLIPBOARD_FRESH_WINDOW_SEC:
            ago = int(now - self._last_clipboard_ts)
            cb = self._last_clipboard[:200].replace("\n", " ")
            lines.append(f"PIXEL_CLIPBOARD ({ago}s_ago): {cb}")

        if self._last_call and (now - self._last_call_ts) < CALL_FRESH_WINDOW_SEC:
            ago = int(now - self._last_call_ts)
            caller = self._last_call.get("caller", "Unknown")
            lines.append(f"PIXEL_CALL ({ago}s_ago, media_paused): {caller}")

        return "\n".join(lines) if lines else "none"

    def _format_past(self, past: list[dict]) -> str:
        if not past:
            return "none"
        lines: list[str] = []
        for p in past:
            meta = p.get("metadata") or {}
            snap_view = format_snapshot(
                {
                    "window": meta.get("snap_window"),
                    "cpu_pct": meta.get("snap_cpu"),
                    "ram_pct": meta.get("snap_ram"),
                    "load_avg": meta.get("snap_load"),
                    "hour": meta.get("snap_hour"),
                }
            )
            doc = p["document"].replace("\n", " | ")
            lines.append(f"{doc} {snap_view}".strip())
        return "\n".join(lines)

    async def _generate_streaming(
        self,
        user_text: str,
        past: list[dict],
        current_snap: StateSnapshot | dict | None = None,
        extra: str = "",
    ) -> str:
        past_text = self._format_past(past)
        context_block = self._build_context_block(current_snap)
        prompt = (
            f"CONTEXT_BLOCK:\n{context_block}\n"
            f"PAST_CONTEXT:\n{past_text}\n"
            f"{extra}"
            f"USER: {user_text}"
        )
        chunks: list[str] = []
        try:
            stream = await self._ollama.generate(
                model=self.model,
                system=self.system_prompt,
                prompt=prompt,
                stream=True,
                options={"temperature": 0.0, "num_ctx": 4096, "num_gpu": 99},
            )
            async for piece in stream:
                tok = piece.get("response", "")
                if not tok:
                    continue
                chunks.append(tok)
                await self.bus.publish(Event(EventType.TOKEN_STREAM, tok))
        except Exception:
            log.exception("ollama streaming failed")
        return "".join(chunks)

    async def _run(self, cmd: str) -> tuple[Optional[int], str, str]:
        return await asyncio.to_thread(run_bash, inject_sudo(cmd))

    def _is_destructive(self, cmd: str) -> bool:
        return bool(DESTRUCTIVE_RE.search(cmd))

    async def _gate_destructive(
        self, cmd: str, intent: str, snap: StateSnapshot | dict | None
    ) -> bool:
        """Returns True if execution may proceed, False if confirmation requested."""
        if not self._is_destructive(cmd):
            return True
        self._pending_confirmation = {
            "cmd": cmd, "intent": intent, "snap": snap, "ts": time.time(),
        }
        await self._state("ALERT", f"CONFIRM: {cmd[:60]}")
        self.say(
            f"Команда потенциально разрушительная. {cmd[:80]}. "
            f"Подтвердите голосом или скажите «отмени».",
            tone="alert",
        )
        log.info("destructive command queued for confirmation: %s", cmd[:200])
        return False

    async def _try_resolve_confirmation(self, text: str) -> bool:
        """If a destructive command is pending and `text` is yes/no, handle it.
        Returns True if the confirmation flow consumed this intent."""
        pending = self._pending_confirmation
        if not pending:
            return False
        if time.time() - pending["ts"] > CONFIRMATION_TIMEOUT_SEC:
            self._pending_confirmation = None
            self.say("Окно подтверждения истекло.", tone="alert")
            return False

        if NEGATIVE_RE.search(text):
            cmd = pending["cmd"]
            self._pending_confirmation = None
            await self._state("IDLE", "")
            self.say(self.ack("cancel"))
            await asyncio.to_thread(
                self.memory.remember,
                f"CANCELLED: {pending['intent']}", cmd, "user_cancelled", "cancel", pending["snap"],
            )
            return True

        if AFFIRMATIVE_RE.search(text):
            cmd = pending["cmd"]
            intent = pending["intent"]
            snap = pending["snap"]
            self._pending_confirmation = None
            await self._state("THINKING", "Executing confirmed action…")
            self.say(self.ack("confirm"), tone="alert")
            rc, stdout, stderr = await self._execute_with_healing(
                cmd, intent, current_snap=snap, skip_confirmation=True,
            )
            result = stdout.strip() or stderr.strip() or (f"rc={rc}" if rc is not None else "detached")
            await asyncio.to_thread(self.memory.remember, intent, cmd, result, "confirmed", snap)
            await self._state("IDLE", "")
            return True

        return False

    async def _execute_with_healing(
        self,
        bash: str,
        intent: str,
        current_snap: StateSnapshot | dict | None = None,
        skip_confirmation: bool = False,
    ) -> tuple[Optional[int], str, str]:
        if not skip_confirmation and not await self._gate_destructive(bash, intent, current_snap):
            return None, "", "awaiting confirmation"

        current = bash
        rc, stdout, stderr = await self._run(current)
        attempts = 0
        while rc not in (0, None) and attempts < MAX_HEAL_ATTEMPTS:
            attempts += 1
            self._last_error_window_hint = self._extract_app_hint(current, stderr)

            await self._highlight_error_window()

            quick = self.patcher.patch(current, stderr)
            if quick is not None:
                name, fixed = quick
                await self._state("THINKING", f"Quick-patch [{name}]: {fixed[:50]}")
                log.info("attempt %d quick-patch %s: %s", attempts, name, fixed)
                current = fixed
            else:
                await self._state("THINKING", f"LLM heal #{attempts}")
                heal = await self._generate_streaming(
                    intent, past=[], current_snap=current_snap,
                    extra=(
                        f"PREVIOUS_CMD: {current}\n"
                        f"ERROR: {stderr.strip()[:400]}\n"
                        f"Return ONLY a corrected bash command.\n"
                    ),
                )
                parsed = parse_response(heal)
                fixed = clean_bash(parsed.bash)
                if not fixed or fixed == current:
                    break
                current = fixed

            if not skip_confirmation and self._is_destructive(current):
                await self._gate_destructive(current, intent, current_snap)
                return None, "", "awaiting confirmation after heal"

            rc, stdout, stderr = await self._run(current)

        if rc not in (0, None):
            tail = stderr.strip()[:160] or stdout.strip()[:160]
            self.say(f"Не справился за {attempts} попытки. {tail}", tone="alert")
            log.error("final fail after %d attempts: rc=%s err=%s", attempts, rc, tail)

        return rc, stdout, stderr

    @staticmethod
    def _extract_app_hint(cmd: str, stderr: str) -> str:
        """Best-effort: which window caption substring relates to this command/error?"""
        m = re.match(r"\s*([\w.+-]+)", cmd or "")
        if not m:
            return ""
        first = m.group(1)
        if first in ("sudo", "sudo-n"):
            m2 = re.match(r"\s*sudo(?:\s+-n)?\s+([\w.+-]+)", cmd or "")
            if m2:
                first = m2.group(1)
        return first[:40]

    async def _highlight_error_window(self) -> None:
        hint = self._last_error_window_hint
        if not hint:
            return
        try:
            await self.kwin.highlight_window(hint)
        except Exception:
            log.exception("KWin highlight failed")

    async def process_intent(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if await self._try_resolve_confirmation(text):
            return "[confirmation_handled]"

        await self._state("THINKING", text[:60])
        snap = await asyncio.to_thread(snapshot)

        macro = self.book.match(text)
        if macro and macro.startswith(KWIN_MACRO_PREFIX):
            action = macro[len(KWIN_MACRO_PREFIX):].strip()
            ok = await self.kwin.execute(action)
            self.say(self.ack("ok") if ok else self.ack("fail"))
            await asyncio.to_thread(
                self.memory.remember, text, macro, "ok" if ok else "fail", "kwin", snap
            )
            await self._state("IDLE", "")
            return macro

        if macro:
            if not await self._gate_destructive(macro, text, snap):
                return "[awaiting_confirmation]"
            rc, stdout, stderr = await asyncio.to_thread(run_bash, macro)
            result = stdout.strip() or stderr.strip() or f"rc={rc}"
            await asyncio.to_thread(self.memory.remember, text, macro, result, "macro", snap)
            await self._state("IDLE", "")
            return macro

        past = await asyncio.to_thread(self.memory.recall, text)
        raw = await self._generate_streaming(text, past, current_snap=snap)
        parsed: ParsedResponse = parse_response(raw)

        await self._state("SPEAKING", parsed.thought or parsed.say or text[:60])
        if parsed.say:
            self.say(parsed.say)

        bash = parsed.bash
        if bash and parsed.sandbox:
            bash = wrap_sandbox(bash)
            log.info("sandbox wrap applied: %s", bash[:200])
            await self._state("THINKING", "Sandbox engaged")

        rc, stdout, stderr = (0, "", "")
        if bash:
            rc, stdout, stderr = await self._execute_with_healing(bash, text, current_snap=snap)

        result = stdout.strip() or stderr.strip() or (
            f"rc={rc}" if rc is not None else "detached"
        )
        kind = "qwen_sandbox" if parsed.sandbox else "qwen"
        await asyncio.to_thread(self.memory.remember, text, bash, result, kind, snap)
        await self._state("IDLE", "")
        return raw

    async def on_voice_intent(self, event: Event) -> None:
        await self.process_intent(str(event.data))

    async def on_daemon_alert(self, event: Event) -> None:
        line = str(event.data)
        await self._state("ALERT", line[:80])
        await self.process_intent(f"[DAEMON_ALERT] {line.strip()}")

    async def on_os_event(self, event: Event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        sensor = data.get("sensor", "?")
        level = data.get("level", "warn")
        value = data.get("value", "?")
        unit = data.get("unit", "")
        suggest = data.get("suggest", "")

        self._recent_os.append({
            "sensor": sensor, "level": level, "value": value,
            "suggest": suggest, "ts": event.ts,
        })

        await self._state("ALERT", f"{sensor}={value}{unit} [{level}]")
        if sensor == "thermal":
            self.say(
                f"Внимание. Температура {int(float(value))} градусов. "
                f"Рекомендую energy-saving.",
                tone="alert",
            )
        elif sensor == "loadavg":
            self.say(f"Нагрузка превышена: {float(value):.1f}.", tone="alert")
        elif sensor == "packages":
            count = int(float(value)) if value not in ("", None) else 0
            if level == "info":
                self.say(f"Ночное обновление выполнено. Установлено пакетов: {count}.", tone="idle")
            else:
                self.say(f"Доступно обновлений: {count}. Хотите установить сейчас?")
        else:
            self.say(f"Системное событие: {sensor}, уровень {level}.", tone="alert")
        await asyncio.to_thread(
            self.memory.remember,
            f"[OS_EVENT] {sensor}={value} {level}",
            suggest,
            "warned",
            "os_event",
            None,
        )

    async def on_pixel_event(self, event: Event) -> None:
        data = event.data if isinstance(event.data, dict) else {"kind": "raw", "payload": str(event.data)}
        kind = str(data.get("kind", "event"))

        if kind == "call_incoming":
            caller = data.get("caller", "Unknown")
            self._last_call = {"caller": caller, "raw": data.get("raw", {})}
            self._last_call_ts = event.ts
            await self._state("ALERT", f"Звонок: {caller}")
            self.say(f"Входящий звонок. {caller}.", tone="alert")
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel incoming call from {caller}",
                "playerctl --all-players pause",
                "media paused",
                "pixel_call",
                None,
            )
            return

        if kind == "clipboard":
            payload = str(data.get("payload", ""))
            self._last_clipboard = payload
            self._last_clipboard_ts = event.ts
            return

        if kind == "quick_command":
            cmd_text = str(data.get("payload", "")).strip()
            if cmd_text:
                log.info("Pixel quick command: %s", cmd_text)
                await self.process_intent(cmd_text)
            return

        if kind == "image_ocr":
            text = str(data.get("text", "")).strip()
            path = str(data.get("path", ""))
            await self._state("SPEAKING", f"OCR: {path}")
            self.say(f"Распознал изображение с телефона. Открываю результат в konsole.", tone="normal")
            if text:
                preview = text[:200].replace("\n", " ")
                await asyncio.to_thread(
                    self.memory.remember,
                    f"Pixel OCR: {path}", "konsole", preview, "pixel_ocr", None,
                )
            return

        payload = str(data.get("payload", ""))
        await self.process_intent(f"[PIXEL_{kind.upper()}] {payload}".strip())
