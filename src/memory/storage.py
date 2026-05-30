"""Mnemosyne — chrono-brain background harvester.

Every :data:`HARVEST_INTERVAL` seconds the daemon silently captures a
"slice of reality" — the user's current focus across the desktop — and
asks :class:`ChronoMemory` to remember it under ``kind="snapshot"``.

What we collect per tick (all best-effort; missing pieces are simply
omitted from the document text):

* **Active window caption** via the existing ``KWinOrchestrator``.
* **Firefox / Chromium URL** by:
    1. parsing the window title (browsers append " — Mozilla Firefox"
       and the *page title* is the prefix), and
    2. as a fall-back, asking the MPRIS DBus interface for the
       currently-playing media URL (only fires when the user watches
       video — the second path is rare but handy when the browser title
       got truncated).
* **Editor context** by walking ``/proc`` for live ``vim``, ``nano`` or
  ``code`` processes and capturing the file argument; if the file is
  Python/JS/Rust we also scan it for the nearest ``def``/``fn``/``class``
  on the line currently being edited (heuristic, capped at 8 KB read).
* **Last shell command** by reading the latest ``HISTFILE`` we can locate
  (``~/.bash_history`` or ``~/.zsh_history``); we only keep the tail line.

Clipboard assimilation is handled in :func:`assimilate_clipboard`. It is
event-driven (subscribed to ``PIXEL_EVENT`` of kind ``clipboard`` and to a
local ``klipper`` watcher) so we can run the heavy "think about this IP"
step out-of-band from the 10s harvest cadence. When something interesting
is detected (IP, MAC, CVE id, an obvious code block, a long URL) we
synchronously query :class:`ChronoMemory.recall` for related history,
generate a one-line thought from the LLM, and store it as
``kind="assimilation"``.

Everything runs under :mod:`asyncio` and never blocks the voice loop —
heavy I/O (DBus, file globs, LLM generate) is awaited inside
``asyncio.to_thread`` / via async DBus / via the AsyncClient.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.event_bus import Event, EventBus, EventType, SystemLoad, SystemState
from src.common.singleton import Singleton

log = logging.getLogger("jarvis.mnemosyne")

# 30с (было 10): каждый снимок персистится через эмбеддинг в ollama (CPU +
# память). На N100 это конкурирует за единственный канал LPDDR5 с iGPU-декодом
# llama-server. Втрое реже harvest = втрое меньше паразитной нагрузки на мозг.
HARVEST_INTERVAL = 30.0          # seconds between scheduled snapshots
ASSIMILATE_COOLDOWN = 12.0       # don't re-think the same clipboard within 12s
MAX_EDITOR_FILE_BYTES = 8192     # cap how much of an open file we read

IP_RE = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
URL_RE = re.compile(r"\bhttps?://[^\s'\"<>]+", re.IGNORECASE)
CODE_HINT_RE = re.compile(
    r"\b(def\s+\w+|class\s+\w+|fn\s+\w+|function\s+\w+|import\s+\w+|#include\s*<|SELECT\s+.+?\s+FROM\b|exploit\s*\()",
    re.IGNORECASE,
)


# --- Tiny pure helpers (separately testable) -------------------------------
def parse_firefox_url(caption: str) -> str | None:
    """Best-effort URL extraction from a Firefox / Chromium window title.

    The browsers append a trailing dash-separated brand to every window
    title — splitting on it leaves the page title. If the page title is
    itself a URL we return it; otherwise nothing.
    """
    if not caption:
        return None
    for sep in (" — Mozilla Firefox", " - Mozilla Firefox", " — Chromium", " - Google Chrome"):
        if caption.endswith(sep):
            head = caption[: -len(sep)].strip()
            if head.startswith("http://") or head.startswith("https://"):
                return head
            return None
    return None


def parse_editor_file(cmdline: str) -> str | None:
    """Pluck the first non-flag argument from a vim/nano/code cmdline."""
    parts = [p for p in cmdline.split("\x00") if p]
    if not parts:
        return None
    binary = os.path.basename(parts[0])
    if binary not in {"vim", "nvim", "vi", "nano", "micro", "code", "codium"}:
        return None
    for arg in parts[1:]:
        if arg.startswith("-") or arg.startswith("/dev/"):
            continue
        if Path(arg).suffix or Path(arg).is_file():
            return arg
    return None


def nearest_symbol(src: str, near_line: int = 1) -> str | None:
    """Return the closest ``def``/``class``/``fn``/``function`` line above
    ``near_line`` (1-indexed). Used to give the assimilation prompt a hint
    about what the user is editing."""
    lines = src.splitlines()
    if not lines:
        return None
    target = max(1, min(near_line, len(lines)))
    pat = re.compile(r"^\s*(def|class|fn|function)\s+([\w_]+)")
    for i in range(target - 1, -1, -1):
        m = pat.match(lines[i])
        if m:
            return f"{m.group(1)} {m.group(2)}"
    return None


def detect_assimilable(payload: str) -> dict[str, list[str]]:
    """Classify free-form text into interesting categories. Returns a dict
    with the lists of matches per category — never returns ``None``."""
    if not payload:
        return {}
    out: dict[str, list[str]] = {}
    ips = [m.group(0) for m in IP_RE.finditer(payload)]
    # Filter trivial / reserved IPs to keep the LLM call meaningful.
    ips = [ip for ip in ips if _interesting_ip(ip)]
    if ips:
        out["ip"] = list(dict.fromkeys(ips))[:5]
    macs = list(dict.fromkeys(m.group(0) for m in MAC_RE.finditer(payload)))[:5]
    if macs:
        out["mac"] = macs
    cves = list(dict.fromkeys(m.group(0).upper() for m in CVE_RE.finditer(payload)))[:5]
    if cves:
        out["cve"] = cves
    urls = list(dict.fromkeys(m.group(0) for m in URL_RE.finditer(payload)))[:3]
    if urls:
        out["url"] = urls
    if CODE_HINT_RE.search(payload) or _looks_like_code(payload):
        out["code"] = [payload[:512]]
    return out


def _interesting_ip(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_unspecified or ip.is_link_local:
        return False
    return True


def _looks_like_code(payload: str) -> bool:
    # crude heuristic — multi-line + at least one bracket/semicolon density.
    if "\n" not in payload:
        return False
    text = payload[:1024]
    density = sum(text.count(c) for c in "{};()[]:")
    return density >= 4 and len(payload) >= 60


# --- Harvester singleton ---------------------------------------------------
@dataclass
class RealitySlice:
    window: str = ""
    url: str | None = None
    editor_file: str | None = None
    editor_symbol: str | None = None
    last_shell: str | None = None
    ts: float = field(default_factory=time.time)

    def to_document(self) -> str:
        bits: list[str] = []
        if self.window:
            bits.append(f"WINDOW: {self.window}")
        if self.url:
            bits.append(f"URL: {self.url}")
        if self.editor_file:
            sym = f" ({self.editor_symbol})" if self.editor_symbol else ""
            bits.append(f"EDIT: {self.editor_file}{sym}")
        if self.last_shell:
            bits.append(f"SHELL: {self.last_shell}")
        return "\n".join(bits)


class Mnemosyne(metaclass=Singleton):
    """Chrono-brain. Owns the 10-second harvest loop + clipboard listener."""

    def __init__(
        self,
        bus: EventBus | None = None,
        memory: Any = None,            # ChronoMemory (Singleton)
        kwin: Any = None,              # KWinOrchestrator
        llm_client: Any = None,        # ollama.AsyncClient
        llm_model: str = "qwen2.5-coder:3b",
    ) -> None:
        self.bus = bus or EventBus()
        self.memory = memory
        self.kwin = kwin
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._last_slice: RealitySlice | None = None
        self._last_assim: dict[str, float] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    # ----- lifecycle ---------------------------------------------------
    def start(self) -> list[asyncio.Task]:
        if self._tasks:
            return self._tasks
        self.bus.subscribe(EventType.PIXEL_EVENT, self._on_pixel_event)
        self._tasks = [
            asyncio.create_task(self._harvest_loop(), name="mnemosyne.harvest"),
            asyncio.create_task(self._klipper_loop(), name="mnemosyne.klipper"),
        ]
        log.info("Mnemosyne started (interval=%.1fs)", HARVEST_INTERVAL)
        return self._tasks

    def stop(self) -> None:
        self._stopping = True
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    # ----- the 10s harvest loop ---------------------------------------
    async def _harvest_loop(self) -> None:
        while not self._stopping:
            try:
                # Под HIGH/CRITICAL пропускаем тик: персист слайса = эмбеддинг в
                # ollama (CPU + память), а это душит iGPU-декод llama-server на
                # общей шине LPDDR5. Память подождёт — отзывчивость мозга важнее.
                if SystemState().load in (SystemLoad.HIGH, SystemLoad.CRITICAL):
                    await asyncio.sleep(HARVEST_INTERVAL)
                    continue
                slice_ = await self._capture_slice()
                # None = тик пропущен (KWin не ответил). Не персистим, не
                # сравниваем, ждём следующий цикл.
                if slice_ is not None and self._is_different(slice_):
                    await self._persist_slice(slice_)
                    self._last_slice = slice_
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("harvest tick failed")
            await asyncio.sleep(HARVEST_INTERVAL)

    async def _capture_slice(self) -> RealitySlice | None:
        win = ""
        if self.kwin is not None:
            try:
                layout = await self.kwin.query_windows()
            except Exception:
                log.exception("kwin query failed")
                layout = None

            # Дуракоустойчивая проверка по ТЗ: KWin под Wayland может вернуть
            # None при DBus-таймауте / перезапуске compositor'а, либо ответить
            # объектом без поля windows (другая версия Plasma). Любая такая
            # ситуация — повод просто пропустить тик, а не ронять harvester.
            if not layout or not hasattr(layout, "windows") or layout.windows is None:
                return None

            for w in layout.windows:
                if getattr(w, "active", False):
                    win = w.caption
                    break
            if not win and layout.windows:
                win = layout.windows[0].caption

        url = parse_firefox_url(win)
        editor_file, editor_symbol = await asyncio.to_thread(self._scan_editor_proc)
        last_shell = await asyncio.to_thread(self._tail_history)

        return RealitySlice(
            window=win,
            url=url,
            editor_file=editor_file,
            editor_symbol=editor_symbol,
            last_shell=last_shell,
        )

    def _scan_editor_proc(self) -> tuple[str | None, str | None]:
        """Walk /proc for live editors; return (file, symbol) or (None, None)."""
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                cmdline_path = Path("/proc") / pid / "cmdline"
                try:
                    raw = cmdline_path.read_text(errors="ignore")
                except Exception:
                    continue
                file_ = parse_editor_file(raw)
                if not file_:
                    continue
                # Try to surface the nearest symbol near the modification
                # timestamp. We don't know the cursor row, so we use the
                # mtime modulo the line count as a coarse proxy — good
                # enough to consistently fingerprint long sessions.
                sym = None
                p = Path(file_)
                if p.is_file() and p.stat().st_size < MAX_EDITOR_FILE_BYTES * 8:
                    try:
                        src = p.read_text(errors="ignore")[:MAX_EDITOR_FILE_BYTES]
                        line_estimate = max(1, int(p.stat().st_mtime) % max(src.count("\n"), 1) + 1)
                        sym = nearest_symbol(src, line_estimate)
                    except Exception:
                        sym = None
                return file_, sym
        except FileNotFoundError:
            return None, None
        return None, None

    def _tail_history(self) -> str | None:
        for fname in (".bash_history", ".zsh_history"):
            path = Path.home() / fname
            if not path.is_file():
                continue
            try:
                # Read only the tail to keep this cheap.
                with path.open("rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    chunk = min(size, 4096)
                    fh.seek(size - chunk)
                    data = fh.read().decode("utf-8", errors="ignore")
                last = [ln for ln in data.splitlines() if ln.strip()]
                if last:
                    return last[-1][:256]
            except Exception:
                continue
        return None

    def _is_different(self, slice_: RealitySlice) -> bool:
        if self._last_slice is None:
            return True
        prev = self._last_slice
        return (
            slice_.window != prev.window
            or slice_.url != prev.url
            or slice_.editor_file != prev.editor_file
            or slice_.last_shell != prev.last_shell
        )

    async def _persist_slice(self, slice_: RealitySlice) -> None:
        if self.memory is None:
            return
        doc = slice_.to_document()
        if not doc:
            return
        try:
            await asyncio.to_thread(
                self.memory.remember,
                intent="chrono.snapshot",
                command="",
                result=doc,
                kind="snapshot",
                snapshot=None,
            )
        except Exception:
            log.exception("chrono persist failed")

    # ----- Clipboard assimilation -------------------------------------
    async def _klipper_loop(self) -> None:
        last_seen: str | None = None
        while not self._stopping:
            try:
                text = await asyncio.to_thread(self._read_local_clipboard)
                if text and text != last_seen and text.strip():
                    last_seen = text
                    await self.assimilate_clipboard(text, source="local")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("klipper poll failed")
            await asyncio.sleep(2.0)

    def _read_local_clipboard(self) -> str | None:
        import subprocess
        for cmd in (
            ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-out"],
        ):
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1.0,
                )
                if out.returncode == 0 and out.stdout:
                    return out.stdout[:8192]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            except Exception:
                continue
        return None

    async def _on_pixel_event(self, event: Event) -> None:
        data = event.data or {}
        if data.get("kind") != "clipboard":
            return
        payload = str(data.get("payload", "") or "")
        if payload:
            await self.assimilate_clipboard(payload, source="pixel")

    async def assimilate_clipboard(self, payload: str, source: str = "local") -> dict | None:
        """Public entry-point: analyze, recall, think, persist. Returns the
        assimilation dict (or ``None`` if nothing interesting was found)."""
        hits = detect_assimilable(payload)
        if not hits:
            return None
        key = self._fingerprint(payload)
        now = time.time()
        if now - self._last_assim.get(key, 0.0) < ASSIMILATE_COOLDOWN:
            return None
        self._last_assim[key] = now

        recall_hits: list[dict] = []
        if self.memory is not None:
            try:
                recall_hits = await asyncio.to_thread(
                    self.memory.recall, payload[:512], 3, 30,
                )
            except Exception:
                log.exception("clipboard recall failed")

        thought = await self._generate_thought(payload, hits, recall_hits)

        if self.memory is not None and thought:
            try:
                await asyncio.to_thread(
                    self.memory.remember,
                    intent=f"assimilate:{','.join(hits.keys())}",
                    command=payload[:512],
                    result=thought,
                    kind="assimilation",
                    snapshot=None,
                )
            except Exception:
                log.exception("assimilation persist failed")

        # Surface the thought into the HUD ticker / nav graph.
        await self.bus.publish(Event(
            EventType.HUD_OVERLAY,
            {"kind": "thought", "text": thought or "(no thought)", "source": source, "hits": hits},
        ))
        return {"hits": hits, "recall": recall_hits, "thought": thought}

    async def _generate_thought(
        self, payload: str, hits: dict[str, list[str]], recall_hits: Iterable[dict]
    ) -> str:
        if self.llm_client is None:
            return self._fallback_thought(hits)
        # Short, blunt prompt. We want the LLM to produce ONE sentence so we
        # can drop it onto the ticker without paragraph-formatting.
        cats = ", ".join(f"{k}={v}" for k, v in hits.items())
        recall_blob = "\n".join(
            f"- {r.get('document', '')[:200]}" for r in recall_hits
        ) or "(no prior context)"
        prompt = (
            "You are Jarvis. The user just copied this payload. "
            "In ONE concise sentence (Russian), state what you think it is and "
            "what action would be useful. Do not greet, do not list, no markdown.\n"
            f"PAYLOAD:\n{payload[:1024]}\n"
            f"CATEGORIES: {cats}\n"
            f"PRIOR CONTEXT:\n{recall_blob}\n"
        )
        try:
            res = await self.llm_client.generate(model=self.llm_model, prompt=prompt)
            return (res.get("response") or "").strip().splitlines()[0][:280]
        except Exception:
            log.exception("LLM assimilate failed")
            return self._fallback_thought(hits)

    @staticmethod
    def _fallback_thought(hits: dict[str, list[str]]) -> str:
        bits = []
        if hits.get("ip"):
            bits.append(f"IP-адреса для проверки: {', '.join(hits['ip'])}")
        if hits.get("cve"):
            bits.append(f"CVE: {', '.join(hits['cve'])}")
        if hits.get("code"):
            bits.append("Замечен блок кода для анализа")
        if hits.get("url"):
            bits.append(f"URL: {hits['url'][0]}")
        return "; ".join(bits) or "(thought unavailable)"

    @staticmethod
    def _fingerprint(payload: str) -> str:
        import hashlib
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]


__all__ = [
    "Mnemosyne",
    "RealitySlice",
    "HARVEST_INTERVAL",
    "ASSIMILATE_COOLDOWN",
    "parse_firefox_url",
    "parse_editor_file",
    "nearest_symbol",
    "detect_assimilable",
]
