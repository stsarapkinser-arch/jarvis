from __future__ import annotations

import asyncio
import logging
import random
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from src.audio.audio_engine import AcousticEngine
from src.memory.ephemeral import EphemeralRunner
from src.common.event_bus import Event, EventBus, EventType, SystemLoad, SystemState
from src.ui.window_manager import KWinOrchestrator
from src.memory.engine import SIG_WARM, ChronoMemory
from src.network.scanner import is_nmap_command, stream_nmap
from src.common.parser import (
    clean_bash,
    inject_sudo,
    run_bash,
    wrap_sandbox,
)
from src.common.repair import QuickPatcher
from src.security.execution import ShadowExec, needs_shadow
from src.common.singleton import Singleton
from src.memory.snapshot import StateSnapshot, snapshot
from src.inference.openai_client import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    LlamaServerClient,
    LlamaServerError,
)
from src.inference.router import IntentCategory, IntentRouter
from src.inference.agent import AgentRun, ToolCall, ToolResult, run_agent
from src.inference import tools as tooldefs

log = logging.getLogger("jarvis.core")

# Абсолютные пути относительно корня проекта (директория выше src/).
# systemd, разные CWD, тесты — везде работает одинаково.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPER_PATH = str(_PROJECT_ROOT / "piper" / "piper")
VOICE_MODEL = str(_PROJECT_ROOT / "piper" / "ru_RU-dmitry-medium.onnx")
# По умолчанию piper ищет config рядом с моделью под именем
# <model>.onnx.json — но у оператора файл лежит как <model>.json (без
# .onnx в середине). Передаём явно через --config, ничего не переименовывая.
VOICE_CONFIG = str(_PROJECT_ROOT / "piper" / "ru_RU-dmitry-medium.json")
SYSTEM_PROMPT_FILE = str(_PROJECT_ROOT / "config" / "system_prompt")

# ───────────── Out-of-process LLM (native llama-server, OpenAI REST) ─────────
# Фаза 1 ТЗ: Python больше НЕ грузит веса в RAM. Инференс — отдельный системный
# процесс (бинарь llama-server из llama.cpp, Vulkan-сборка; см. setup_server.sh
# и config/jarvis-llm.service). Общаемся по HTTP (httpx) через OpenAI-совместимый
# /v1/chat/completions с массивом tools (Native Function Calling).
#
# Что это даёт на N100:
#   * llama-server держит системный промпт в KV-кэше (--cache-reuse) — микро-
#     промпт (≤100 слов) не пересчитывается на каждый запрос → низкий TTFT;
#   * запрос неблокирующий → EventBus, HUD и голос не замирают на раздумьях;
#   * нативный краш бэкенда не убивает Jarvis — падает лишь отдельный демон,
#     systemd (jarvis-llm.service, Restart=always) его поднимает.
#
# Имя модели — для диагностики/setup-скрипта; сам файл живёт на стороне сервера.
LLAMA_MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_MODEL_PATH = str(_PROJECT_ROOT / "models" / LLAMA_MODEL_NAME)
LLM_ENDPOINT = DEFAULT_ENDPOINT
LLM_MODEL = DEFAULT_MODEL
# Агентный цикл: максимум раундов tool-calling за один интент. 4 хватает на
# read_telemetry → internal_monologue → set_hud_state → speak_response/execute_bash,
# и держит TTFT под контролем на слабом железе.
# 2 раунда: один tool-вызов + опциональный follow-up (напр. read_telemetry→speak).
# На N100 декод ~1 т/с — каждый лишний раунд это +десятки секунд, так что 2 < 4.
AGENT_MAX_STEPS = 2
# 160 — одного speak_response (+ set_hud_state/execute_bash) хватает. Жёсткий
# потолок критичен: на iGPU N100 декод ~1 токен/сек (в логах eval 1473 мс/ток),
# поэтому каждый сгенерированный токен ≈ секунда ответа. Раньше 3B уходила в
# простыню (n_tokens=1489) → таймаут.
LLM_MAX_TOKENS_DEFAULT = 160
# Болтовня заслуживает чуть больше «температуры» и места; системные операции —
# почти детерминированы (точность важнее креатива).
_CATEGORY_TEMPERATURE: dict[IntentCategory, float] = {
    IntentCategory.CONVERSATION: 0.6,
    IntentCategory.SYSTEM_OPS: 0.2,
    IntentCategory.UI_CONTROL: 0.2,
    IntentCategory.PENTEST_RECON: 0.25,
}
# curl|bash, wget|sh, pip install <url>, скачанный и тут же запущенный бинарь —
# авто-оборачиваем в firejail/systemd-run, не доверяя «флагу» от модели.
_RISKY_DOWNLOAD_RE = re.compile(
    r"(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|python\d?)"
    r"|pip\d?\s+install\s+(?:https?://|git\+)"
    r"|\bbash\s+<\(\s*curl",
    re.IGNORECASE,
)

MAX_HEAL_ATTEMPTS = 3
RECENT_OS_EVENTS_MAX = 5
CALL_FRESH_WINDOW_SEC = 300
CLIPBOARD_FRESH_WINDOW_SEC = 600
CONFIRMATION_TIMEOUT_SEC = 30.0

# Context Weaver — Qwen2.5-Coder:3B input budget. Real ctx is 4096 toks but
# we want to leave 3000+ for the actual response, so each section caps itself.
WEAVER_MEMORY_FACTS = 3
WEAVER_FACT_MAX_CHARS = 120
WEAVER_INTENT_MAX_CHARS = 300
WEAVER_BATTERY_FRESH_SEC = 900  # 15 min

# ───────────── Acoustic tract → src/audio/audio_engine.py ─────────────
# Кинематографический голосовой тракт (когнитивная просодия + SoX «Bettany
# Signature» + FFT-перекачка для сферы HUD) вынесен в AcousticEngine. Ядро
# больше не держит SOX-профили/очередь/воркер — оно лишь зовёт
# ``self._acoustic.speak(text, state, speed=, pause=)``. Ambient/night-логика,
# прерывание ALERT'ом и модуляция нагрузкой системы — внутри движка.

ACK_VARIANTS: dict[str, tuple[str, ...]] = {
    "ok": ("Готово.", "Сделано.", "Принято.", "Выполнено.", "Есть."),
    "fail": ("Не удалось.", "Не получилось.", "Ошибка."),
    "confirm": ("Подтверждено. Выполняю.", "Принял. Действую.", "Есть, выполняю."),
    "cancel": ("Отменено.", "Отбой.", "Отменил."),
}

# ───────── Raw error filter ─────────────────────────────────────────────────
# Паттерны «технического мусора», которые не должны попадать в речь.
_RAW_ERROR_RE = re.compile(
    r"(?:"
    r"Error:\s"
    r"|Traceback\s"
    r"|^\s*at\s+\w"
    r"|No such file or directory"
    r"|Permission denied"
    r"|command not found"
    r"|bash:\s"
    r"|rc=\d+"
    r"|stderr:"
    r"|WARN(?:ING)?:"
    r"|^\[Errno\s"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ───── OS event dedup: не повторять один sensor чаще чем N секунд ───────────
OS_ALERT_COOLDOWN_SEC: dict[str, float] = {
    "thermal":   300.0,   # температура — раз в 5 минут
    "loadavg":   120.0,
    "disk":      600.0,
    "memory":    300.0,
    "battery":   180.0,
    "internet":  120.0,
    "packages":  3600.0,
    "storage":    60.0,
}
OS_ALERT_COOLDOWN_DEFAULT = 120.0

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

# Голосовой триггер «диагностики» — пользователь говорит «джарвис, диагностика»
# или «диагностику», по латиннице тоже ловим. На матче публикуем
# DIAGNOSTIC_START → HUD показывает 10-сек оверлей с CPU/RAM/iGPU/temp.
DIAGNOSTIC_RE = re.compile(r"\b(диагностик\w*|diagnostic\w*)\b", re.IGNORECASE)
DIAGNOSTIC_DURATION_SEC = 10.0


@dataclass(slots=True)
class _IntentState:
    """Транзиентное состояние одного интента, мутируемое tool-диспетчером.

    Живёт ровно один вызов ``process_intent`` — собирает побочные эффекты
    (озвучил ли что-то, какие bash выполнил, итог) для записи в память и
    fallback-логики."""
    category: IntentCategory
    spoke: bool = False
    last_thought: str = ""
    last_result: str = ""
    bash_commands: list[str] = field(default_factory=list)
    run: AgentRun | None = None


class _LlamaCompletionAdapter:
    """Совместимый shim под ollama.AsyncClient API surface.

    Mnemosyne / Pixel-thinker и пр. подсистемы исторически писались под
    ``.generate(model=, prompt=)`` ollama-клиента, который возвращает
    ``{"response": "..."}``. Этот адаптер транслирует такие вызовы в общий
    ``Jarvis()._llama_complete`` — модель грузится РАЗ за процесс, второй
    1.9 GiB-резидент в RAM не плодим.
    """

    def __init__(self, jarvis: Jarvis) -> None:
        self._jarvis = jarvis

    async def generate(
        self,
        model: str = "",
        prompt: str = "",
        options: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, str]:
        del model, options  # honoured globally в core.py-конфиге
        text = await self._jarvis._llama_complete(prompt or "")
        return {"response": text}


class Jarvis(metaclass=Singleton):
    """Central async orchestrator. Subscribes to the event bus.
    Holds short-lived context state (recent OS events, last Pixel data)
    that feeds into every LLM prompt as a CONTEXT_BLOCK."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.memory = ChronoMemory()
        self.kwin = KWinOrchestrator()
        self.patcher = QuickPatcher()
        self.model = LLAMA_MODEL_NAME
        try:
            self.system_prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            log.error("system_prompt missing at %s — using minimal fallback", SYSTEM_PROMPT_FILE)
            self.system_prompt = "Ты — JARVIS, краткий ассистент. Отвечай по делу."

        # Inference client: async HTTP к нативному llama-server (OpenAI REST).
        # Веса модели живут в ОТДЕЛЬНОМ процессе; нативный краш бэкенда не валит
        # Jarvis — клиент просто получит LlamaServerError, а Jarvis озвучит сбой.
        self._llm = LlamaServerClient()
        # Семантический маршрутизатор: core-identity берём из config/system_prompt
        # (короткий стабильный префикс), правила категорий — внутри роутера.
        self.router = IntentRouter(core_identity=self.system_prompt)
        # Адаптер под ollama-совместимый интерфейс — для Mnemosyne и т.п.
        self.llm_adapter = _LlamaCompletionAdapter(self)

        self._recent_os: deque[dict] = deque(maxlen=RECENT_OS_EVENTS_MAX)
        self._last_clipboard: str = ""
        self._last_clipboard_ts: float = 0.0
        self._last_call: dict | None = None
        self._last_call_ts: float = 0.0
        self._last_battery: dict | None = None
        self._last_battery_ts: float = 0.0
        self._last_say_ts: float = 0.0  # updated by say(); drives proactive loop

        self._pending_confirmation: dict | None = None
        self._pending_shadow: dict | None = None
        self._state_label: str = "IDLE"
        self._last_error_window_hint: str = ""

        # Shadow Exec — predictive sandbox for LLM-emitted bash, and the
        # ephemeral Python runner that depends on it.
        self._shadow = ShadowExec()
        self._ephemeral = EphemeralRunner(shadow=self._shadow)

        # ── Кинематографический голос (AcousticEngine) ──────────────────────
        # Движок владеет очередью речи, рабочим потоком, SoX-трактом «Bettany
        # Signature» и FFT-перекачкой для сферы HUD. Ядро лишь зовёт .speak().
        self._acoustic = AcousticEngine(
            self.bus,
            piper_path=PIPER_PATH,
            voice_model=VOICE_MODEL,
            voice_config=VOICE_CONFIG,
        )

        # ── OS alert dedup ──────────────────────────────────────────────────
        # Для каждого sensor-ключа храним время последнего say(), чтобы
        # не бубнить «Температура 83 градуса» каждые 60 секунд.
        self._os_alert_last_say: dict[str, float] = {}

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

    def say(
        self,
        text: str,
        tone: str | None = None,
        *,
        speed: float | None = None,
        pause: float | None = None,
    ) -> None:
        """Озвучить текст через AcousticEngine (последовательная очередь, кино-DSP).

        ``tone`` (normal/alert/idle) маппится на состояние голоса; ALERT прерывает
        текущую речь. Опциональные ``speed``/``pause`` — когнитивная просодия от
        LLM (speak_response). Инлайн-тег ``<say speed pause>`` в тексте тоже
        учитывается движком. Метод неблокирующий — лишь кладёт фразу в очередь."""
        text = (text or "").strip()
        if not text:
            return
        self._last_say_ts = time.time()
        chosen = tone or self._tone_for_state()
        self._acoustic.speak(text, chosen, speed=speed, pause=pause)

    # ───────────── Context Weaver (Pillar 4) ─────────────
    # Builds the structured short prompt the spec asks for:
    #   [Memory: N facts]\n - ...\n
    #   [System: load=... cpu=...% ram=...% disk=...% thermal=...C]\n
    #   [Pixel: battery=...% (...) | clipboard=... (...)]\n
    #   [User Intent: ...]
    # The Weaver is sync + side-effect-free so it can be exercised from a
    # smoke test without an event loop.

    @staticmethod
    def _fmt_memory_facts(past: list[dict]) -> str:
        if not past:
            return "[Memory: none]"
        facts: list[str] = []
        for p in past[:WEAVER_MEMORY_FACTS]:
            meta = p.get("metadata") or {}
            kind = str(meta.get("kind", "task"))
            ts = float(meta.get("ts") or 0.0)
            stamp = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts > 0 else "?"
            doc = str(p.get("document", "")).replace("\n", " | ").strip()
            if len(doc) > WEAVER_FACT_MAX_CHARS:
                doc = doc[: WEAVER_FACT_MAX_CHARS - 1] + "…"
            facts.append(f" - {stamp} [{kind}] {doc}")
        return f"[Memory: {len(facts)} facts]\n" + "\n".join(facts)

    @staticmethod
    def _disk_percent(path: str = "/") -> float | None:
        if not _HAS_PSUTIL:
            return None
        try:
            return float(psutil.disk_usage(path).percent)
        except Exception:
            return None

    def _fmt_system(self) -> str:
        snap = SystemState().snapshot()
        disk = self._disk_percent("/")
        parts = [f"load={snap.load.value}"]
        if snap.cpu:
            parts.append(f"cpu={snap.cpu:.0f}%")
        if snap.ram:
            parts.append(f"ram={snap.ram:.0f}%")
        if disk is not None:
            parts.append(f"disk={disk:.0f}%")
        if snap.thermal:
            parts.append(f"thermal={snap.thermal:.0f}C")
        if snap.gpu:
            parts.append(f"gpu={snap.gpu:.0f}%")
        if snap.heavy:
            parts.append("heavy_app=true")
        # Локальное время суток — без него Jarvis не отличит 02:00 от 14:00.
        # System Prompt использует это для тональности (ночью тише, утром энергичнее).
        parts.append(f"time={time.strftime('%H:%M')}")
        head = f"[System: {' '.join(parts)}]"

        # Recent OS warnings — only show if there's something inside the
        # last few minutes; otherwise the live state line above suffices.
        if self._recent_os:
            now = time.time()
            recent: list[str] = []
            for e in list(self._recent_os)[-3:]:
                ts = float(e.get("ts") or now)
                ago = int(now - ts)
                if ago > 300:  # 5 min
                    continue
                sensor = e.get("sensor", "?")
                level = e.get("level", "?")
                value = e.get("value")
                recent.append(f"{sensor}={value}({level},{ago}s_ago)")
            if recent:
                return head + f"\n[SystemRecent: {', '.join(recent)}]"
        return head

    def _fmt_pixel(self) -> str:
        now = time.time()
        parts: list[str] = []

        if self._last_battery and (now - self._last_battery_ts) < WEAVER_BATTERY_FRESH_SEC:
            ago = int(now - self._last_battery_ts)
            charge = self._last_battery.get("charge")
            charging = self._last_battery.get("charging")
            tag = "charging" if charging else "discharging"
            parts.append(f"battery={charge}% ({tag}, {ago}s_ago)")

        if self._last_clipboard and (now - self._last_clipboard_ts) < CLIPBOARD_FRESH_WINDOW_SEC:
            ago = int(now - self._last_clipboard_ts)
            cb = self._last_clipboard[:120].replace("\n", " ")
            parts.append(f'clipboard="{cb}" ({ago}s_ago)')

        if self._last_call and (now - self._last_call_ts) < CALL_FRESH_WINDOW_SEC:
            ago = int(now - self._last_call_ts)
            caller = self._last_call.get("caller", "Unknown")
            parts.append(f"call={caller} ({ago}s_ago, media_paused)")

        if not parts:
            return "[Pixel: none]"
        return "[Pixel: " + " | ".join(parts) + "]"

    @staticmethod
    def _fmt_intent(user_text: str) -> str:
        text = (user_text or "").strip().replace("\n", " ")
        if len(text) > WEAVER_INTENT_MAX_CHARS:
            text = text[: WEAVER_INTENT_MAX_CHARS - 1] + "…"
        return f"[User Intent: {text}]"

    def build_context(self, user_text: str, past: list[dict]) -> str:
        """Compose the four-section structured prompt for Qwen2.5-Coder:3B.

        Order is fixed (Memory → System → Pixel → User Intent) so the
        model learns a stable schema; missing slots render as ``none``.
        Sync + pure (apart from a single psutil disk read), so unit tests
        can exercise it directly."""
        return (
            self._fmt_memory_facts(past) + "\n"
            + self._fmt_system() + "\n"
            + self._fmt_pixel() + "\n"
            + self._fmt_intent(user_text)
        )

    # ───────────── llama-server engine (HTTP, OpenAI REST) ─────────────
    async def _llama_complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
        temperature: float = 0.0,
    ) -> str:
        """Plain (no-tools) completion. Используется адаптером совместимости
        (Mnemosyne) и проактивным циклом. На сбой сервера возвращает ""."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await self._llm.chat(
                messages, tools=None, temperature=temperature, max_tokens=max_tokens,
            )
            return resp.content.strip()
        except LlamaServerError as e:
            log.error("llama-server completion failed: %s", e)
            return ""

    async def warmup(self) -> bool:
        """Прогрев llama-server при старте.

        Один крошечный запрос (с tools, по самому частому пути SYSTEM_OPS):
          * компилирует Vulkan-шейдеры iGPU (главная статья холодного старта —
            десятки секунд при первой генерации);
          * греет KV-префикс системного промпта, который переиспользуют реальные
            запросы.
        Чтобы ПЕРВАЯ голосовая команда не ждала холодный старт. max_tokens=4 —
        нужен лишь прогон prefill+decode, результат отбрасываем."""
        try:
            system = self.router.system_prompt_for(IntentCategory.SYSTEM_OPS)
            tools = tooldefs.tools_for_category(IntentCategory.SYSTEM_OPS)
            await self._llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "ping"},
                ],
                tools=tools, temperature=0.0, max_tokens=4,
            )
            return True
        except LlamaServerError as e:
            log.warning("warmup пропущен (сервер недоступен): %s", e)
            return False
        except Exception:
            log.debug("warmup unexpected error", exc_info=True)
            return False

    # ───────────── Tool dispatch (Native Function Calling) ─────────────
    @staticmethod
    def _tone_for_mood(mood: tooldefs.SpeakMood) -> str:
        """speak_response.mood → tone-строка для AcousticEngine (state/DSP-профиль)."""
        return {
            tooldefs.SpeakMood.ALERT: "alert",
            tooldefs.SpeakMood.PROFESSIONAL: "normal",
            tooldefs.SpeakMood.IRONIC: "normal",
        }.get(mood, "normal")

    async def _apply_hud_state(self, color: str, animation: tooldefs.HudAnimation) -> None:
        """set_hud_state → событие HUD_STATE. Нейросеть сама рулит визором Aegis."""
        await self.bus.publish(Event(
            EventType.HUD_STATE,
            {"color": color, "animation": animation.value},
        ))

    def _read_sensor(self, sensor: tooldefs.TelemetrySensor) -> str:
        """read_telemetry → короткая строка с живыми метриками (уходит в модель).

        Читаем из SystemState (поддерживается Sentinel'ом, без свежего psutil-
        опроса) — почти мгновенно, что и нужно для горячего пути."""
        snap = SystemState().snapshot()
        if sensor == tooldefs.TelemetrySensor.CPU:
            return (
                f"cpu={snap.cpu:.0f}% thermal={snap.thermal:.0f}C "
                f"gpu={snap.gpu:.0f}% load={snap.load.value}"
            )
        if sensor == tooldefs.TelemetrySensor.RAM:
            disk = self._disk_percent("/")
            disk_part = f" disk={disk:.0f}%" if disk is not None else ""
            return f"ram={snap.ram:.0f}%{disk_part}"
        if sensor == tooldefs.TelemetrySensor.NETWORK:
            for e in reversed(self._recent_os):
                if e.get("sensor") == "internet":
                    return f"network: {e.get('value')} ({e.get('level')})"
            return "network: соединение в норме"
        if sensor == tooldefs.TelemetrySensor.PIXEL_PHONE:
            if self._last_battery:
                tag = "charging" if self._last_battery.get("charging") else "discharging"
                return f"pixel battery={self._last_battery.get('charge')}% ({tag})"
            return "pixel: данных нет"
        return "unknown sensor"

    async def _run_background(self, cmd: str) -> int:
        """Detached фоновый запуск (execute_bash background=true). Возвращает PID."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc.pid

    async def _agent_execute_bash(
        self,
        args: tooldefs.ExecuteBashArgs,
        intent: str,
        snap: StateSnapshot | dict | None,
    ) -> tuple[str, bool]:
        """Исполнить execute_bash через существующий конвейер безопасности.

        Возвращает (краткий результат для модели, stop) — stop=True означает,
        что команда ушла на голосовое подтверждение или была отклонена в
        симуляции, и агентный цикл нужно остановить."""
        cmd = args.command.strip()
        if not cmd:
            return "пустая команда", False

        # sudo: явный флаг модели → префикс; иначе авто-детект известных глаголов.
        if args.requires_sudo and not cmd.lstrip().startswith("sudo "):
            cmd = "sudo -n " + cmd
        else:
            cmd = inject_sudo(cmd)

        # curl|bash и подобное — авто-jail, не доверяя «честности» модели.
        if _RISKY_DOWNLOAD_RE.search(cmd):
            cmd = wrap_sandbox(cmd)
            log.info("agent bash auto-sandboxed: %s", cmd[:200])

        if args.background:
            if self._is_destructive(cmd):
                await self._gate_destructive(cmd, intent, snap)
                return "разрушительная фоновая команда — жду подтверждения", True
            pid = await self._run_background(cmd)
            await asyncio.to_thread(
                self.memory.remember, intent, cmd, f"background pid={pid}", "agent_bg", snap,
            )
            return f"запущено в фоне, pid {pid}", False

        # Shadow Exec dry-run → реальное исполнение (тот же путь, что был).
        if needs_shadow(cmd):
            shadow_kind = await self._shadow_then_real(cmd, intent, snap)
            if shadow_kind == "queued":
                return "симуляция чиста — жду голосового подтверждения оператора", True
            if shadow_kind == "rejected":
                return "симуляция отклонила команду (rc!=0)", True
            # "direct" → стандартный путь ниже.

        rc, stdout, stderr = await self._execute_with_healing(cmd, intent, current_snap=snap)
        if rc is None and "awaiting confirmation" in stderr:
            return "команда разрушительна — жду голосового подтверждения", True

        result = stdout.strip() or stderr.strip() or (
            f"rc={rc}" if rc is not None else "detached"
        )
        await asyncio.to_thread(self.memory.remember, intent, cmd, result, "agent_bash", snap)
        return f"rc={rc}; {result[:400]}", False

    async def _dispatch_tool(
        self,
        call: ToolCall,
        intent: str,
        snap: StateSnapshot | dict | None,
        st: _IntentState,
    ) -> ToolResult:
        """Связывает абстрактный ToolCall с реальной подсистемой Jarvis."""
        name = call.name
        if name == tooldefs.ToolName.INTERNAL_MONOLOGUE:
            thought = tooldefs.MonologueArgs.from_dict(call.arguments).thought
            log.info("CoT: %s", thought[:300])
            st.last_thought = thought
            return ToolResult(call.id, "logged")

        if name == tooldefs.ToolName.SPEAK_RESPONSE:
            sp = tooldefs.SpeakArgs.from_dict(call.arguments)
            if sp.text.strip():
                # Когнитивная просодия: mood → тон, speed/pause → ритм Piper.
                self.say(
                    sp.text.strip(),
                    tone=self._tone_for_mood(sp.mood),
                    speed=sp.speed,
                    pause=sp.pause,
                )
                st.spoke = True
                # HUD ticker continuity (раньше его кормил token-stream).
                await self.bus.publish(Event(EventType.TOKEN_STREAM, sp.text.strip()))
            return ToolResult(call.id, "spoken")

        if name == tooldefs.ToolName.SET_HUD_STATE:
            hud = tooldefs.HudArgs.from_dict(call.arguments)
            await self._apply_hud_state(hud.color, hud.animation)
            return ToolResult(call.id, "hud updated")

        if name == tooldefs.ToolName.READ_TELEMETRY:
            tel = tooldefs.TelemetryArgs.from_dict(call.arguments)
            reading = self._read_sensor(tel.sensor)
            return ToolResult(call.id, reading)

        if name == tooldefs.ToolName.EXECUTE_BASH:
            ba = tooldefs.ExecuteBashArgs.from_dict(call.arguments)
            st.bash_commands.append(ba.command)
            summary, stop = await self._agent_execute_bash(ba, intent, snap)
            st.last_result = summary
            return ToolResult(call.id, summary, stop=stop)

        log.warning("unknown tool call: %s", name)
        return ToolResult(call.id, f"unknown tool {name}")

    async def _run_agent_for_intent(
        self,
        user_text: str,
        past: list[dict],
        snap: StateSnapshot | dict | None,
    ) -> _IntentState:
        """Маршрутизация → микро-промпт + tools → агентный цикл с tool-dispatch."""
        decision = self.router.route(user_text)
        system_prompt = self.router.system_prompt_for(decision.category)
        tools = tooldefs.tools_for_category(decision.category)
        temperature = _CATEGORY_TEMPERATURE.get(decision.category, 0.2)
        log.info(
            "route=%s (backend=%s score=%.1f) tools=%d",
            decision.category.value, decision.backend, decision.score, len(tools),
        )

        context = self.build_context(user_text, past)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        st = _IntentState(category=decision.category)

        async def dispatch(call: ToolCall) -> ToolResult:
            return await self._dispatch_tool(call, user_text, snap, st)

        run: AgentRun = await run_agent(
            self._llm, messages, tools, dispatch,
            max_steps=AGENT_MAX_STEPS, temperature=temperature,
            max_tokens=LLM_MAX_TOKENS_DEFAULT,
        )
        st.run = run
        # Модель ответила голым текстом вместо speak_response — всё равно озвучим.
        if not st.spoke and run.stopped == "no_tools" and run.final_content.strip():
            self.say(run.final_content.strip()[:400])
            st.spoke = True
        return st

    async def _run(self, cmd: str) -> tuple[int | None, str, str]:
        injected = inject_sudo(cmd)
        if is_nmap_command(injected):
            # nmap streams through the HUD port matrix in real time.
            return await stream_nmap(injected, bus=self.bus)
        return await asyncio.to_thread(run_bash, injected)

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

    async def _try_resolve_shadow(self, text: str) -> bool:
        """If a Shadow-verified command is pending and ``text`` is yes/no,
        resolve it. Returns True when this branch consumed the intent."""
        pending = self._pending_shadow
        if not pending:
            return False
        if time.time() - pending["ts"] > CONFIRMATION_TIMEOUT_SEC:
            self._pending_shadow = None
            await self.bus.publish(Event(EventType.HUD_OVERLAY, {"kind": "safe_pulse", "active": False}))
            self.say("Окно симуляции истекло.", tone="alert")
            return False
        if NEGATIVE_RE.search(text):
            self._pending_shadow = None
            await self.bus.publish(Event(EventType.HUD_OVERLAY, {"kind": "safe_pulse", "active": False}))
            await self._state("IDLE", "")
            self.say(self.ack("cancel"))
            return True
        if AFFIRMATIVE_RE.search(text):
            cmd = pending["cmd"]
            intent = pending["intent"]
            snap = pending["snap"]
            self._pending_shadow = None
            await self.bus.publish(Event(EventType.HUD_OVERLAY, {"kind": "safe_pulse", "active": False}))
            await self._state("THINKING", "Executing sandbox-verified action…")
            self.say(self.ack("confirm"))
            rc, stdout, stderr = await self._execute_with_healing(
                cmd, intent, current_snap=snap, skip_confirmation=True,
            )
            result = stdout.strip() or stderr.strip() or (f"rc={rc}" if rc is not None else "detached")
            await asyncio.to_thread(self.memory.remember, intent, cmd, result, "shadow_confirmed", snap)
            await self._state("IDLE", "")
            return True
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
    ) -> tuple[int | None, str, str]:
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
                heal = await self._llama_complete(
                    (
                        "Команда упала. Верни ТОЛЬКО одну исправленную bash-команду, "
                        "без пояснений, без markdown, без backticks.\n"
                        f"PREVIOUS_CMD: {current}\n"
                        f"ERROR: {stderr.strip()[:400]}\n"
                    ),
                    system=self.router.system_prompt_for(IntentCategory.SYSTEM_OPS),
                    max_tokens=160,
                    temperature=0.0,
                )
                fixed = clean_bash(heal)
                if not fixed or fixed == current:
                    break
                current = fixed

            if not skip_confirmation and self._is_destructive(current):
                await self._gate_destructive(current, intent, current_snap)
                return None, "", "awaiting confirmation after heal"

            rc, stdout, stderr = await self._run(current)

        if rc not in (0, None):
            tail = stderr.strip()[:160] or stdout.strip()[:160]
            # Не зачитываем технический мусор — фильтруем raw ошибки.
            if tail and _RAW_ERROR_RE.search(tail):
                tail = ""
            if tail:
                self.say(f"Не справился за {attempts} попытки. {tail}", tone="alert")
            else:
                self.say(f"Не удалось выполнить за {attempts} попытки.", tone="alert")
            log.error("final fail after %d attempts: rc=%s err=%s", attempts, rc, stderr.strip()[:160])

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
        # Also paint a neon AR bracket on the HUD around the matching window
        # so the user sees the diagnosis even without focus stealing.
        try:
            matches = await self.kwin.find_window_rects(hint)
        except Exception:
            log.exception("KWin geometry probe failed")
            matches = []
        if matches:
            await self.bus.publish(Event(
                EventType.HUD_OVERLAY,
                {"kind": "app_glow", "windows": [
                    {**m, "color": "amber", "caption": hint} for m in matches
                ]},
                urgency="normal",
            ))

    async def process_intent(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if await self._try_resolve_shadow(text):
            return "[shadow_resolved]"
        if await self._try_resolve_confirmation(text):
            return "[confirmation_handled]"

        # Голосовой триггер «диагностики» — HUD оверлей с метриками. Не
        # завершаем здесь: LLM/память тоже видят запрос (вдруг оператор
        # хотел не только overlay, но и устный отчёт).
        if DIAGNOSTIC_RE.search(text):
            await self.bus.publish(Event(
                EventType.DIAGNOSTIC_START,
                {"requested_at": time.time(), "duration_sec": DIAGNOSTIC_DURATION_SEC},
            ))
            self.say("Запускаю диагностику.")

        await self._state("THINKING", text[:60])
        snap = await asyncio.to_thread(snapshot)
        # RAG-recall — это эмбеддинг-вызов в ollama (CPU + память). На N100 он
        # конкурирует за единственный канал памяти с iGPU-декодом llama-server.
        # Под HIGH/CRITICAL пропускаем recall: лучше ответить без памяти, чем
        # задушить мозг лишней нагрузкой ровно перед генерацией. Symbiote-degrade.
        if SystemState().load in (SystemLoad.HIGH, SystemLoad.CRITICAL):
            past = []
        else:
            past = await asyncio.to_thread(self.memory.recall, text)

        # Агентный цикл: маршрутизация → микро-промпт + tools → Native Function
        # Calling. Никакого текстового парсинга — модель ДЕЙСТВУЕТ инструментами
        # (speak_response / execute_bash / set_hud_state / read_telemetry).
        try:
            st = await self._run_agent_for_intent(text, past, snap)
        except LlamaServerError as e:
            log.error("llama-server недоступен на интенте %r: %s", text[:80], e)
            self.say("Сэр, мозг не ответил — llama-server недоступен. Проверьте jarvis-llm.")
            await self._state("IDLE", "")
            return ""

        # Защита от «немоты»: модель не позвала ни одного инструмента и не
        # сказала ни слова — ловимый сбой инференса. Озвучиваем, а не уходим
        # молча в IDLE (оператор должен знать, что его услышали).
        if not st.spoke and not st.bash_commands and (
            st.run is None or st.run.tool_calls_made == 0
        ):
            log.error("агент не произвёл действий на интенте %r — озвучиваю сбой", text[:80])
            self.say("Сэр, мозг не ответил — похоже, модель дала сбой. Проверьте llama-server.")
            await self._state("IDLE", "")
            return ""

        # Память: что просили, что выполнили, чем кончилось.
        bash_joined = " && ".join(c for c in st.bash_commands if c)
        result = st.last_result or st.last_thought or ("spoken" if st.spoke else "no-op")
        await asyncio.to_thread(
            self.memory.remember,
            text, bash_joined, result, f"agent_{st.category.value.lower()}", snap,
        )
        await self._state("IDLE", "")
        return "[agent_done]"

    async def _shadow_then_real(
        self, bash: str, intent: str, snap: StateSnapshot | dict | None,
    ) -> str:
        """Run the bash inside Shadow Exec, then decide what to do next.

        Returns one of:
          * ``"queued"``    — sim was clean (rc=0); we asked the operator
                              for a verbal go-ahead. ``process_intent``
                              should bail out and let the next intent
                              resolve the queue.
          * ``"rejected"``  — sandbox returned non-zero; we already spoke
                              the rejection and ``process_intent`` should
                              skip running the real command.
          * ``"direct"``    — no sandbox engine available, we tell the
                              caller to continue with the unsandboxed
                              execution path (still gated by destructive
                              confirmation).
        """
        if not self._shadow.available_engines:
            log.info("shadow: no engine available, falling through to direct exec")
            return "direct"
        await self._state("THINKING", f"Sandbox dry-run: {bash[:60]}")
        shadow_res = await self._shadow.run(bash)
        log.info("shadow rc=%s engine=%s tail=%s",
                 shadow_res.rc, shadow_res.engine, shadow_res.stderr[-160:].strip())

        if shadow_res.rc != 0:
            await self._state("ALERT", f"Shadow rejected rc={shadow_res.rc}")
            tail = (shadow_res.stderr.strip().splitlines() or [""])[-1][:120]
            self.say(
                f"Симуляция упала. rc {shadow_res.rc}. {tail or 'ошибок в выводе нет.'}",
                tone="alert",
            )
            await asyncio.to_thread(
                self.memory.remember, intent, bash,
                f"shadow_rc={shadow_res.rc} {tail}", "shadow_reject", snap,
            )
            return "rejected"

        # rc == 0 → light up the green safe pulse and ask for confirmation.
        await self.bus.publish(Event(EventType.HUD_OVERLAY, {"kind": "safe_pulse", "active": True}))
        await self._state("SPEAKING", "Sim OK — awaiting confirmation")
        self._pending_shadow = {
            "cmd": bash, "intent": intent, "snap": snap, "ts": time.time(),
            "engine": shadow_res.engine,
        }
        self.say(
            f"Модель действий проверена в симуляции без крашей через {shadow_res.engine}. "
            "Вывести в реальную систему, сэр?",
        )
        return "queued"

    async def on_voice_intent(self, event: Event) -> None:
        await self.process_intent(str(event.data))

    async def on_deep_watch(self, event: Event) -> None:
        """Ring-0 eBPF sample (execve / tcp_v4_connect).

        These are firehose-volume events even after DeepWatch dedup/rate-limit.
        We do NOT speak; we only stash them in the rolling context window so
        the next LLM call gets visibility into what the kernel just saw. The
        HUD subscribes separately if it wants a visualisation."""
        data = event.data if isinstance(event.data, dict) else {}
        kind = data.get("kind", "?")
        comm = data.get("comm", "?")
        # Tag as a tiny synthetic OS event for the context block.
        self._recent_os.append({
            "sensor": "ring0",
            "level": "trace",
            "value": f"{kind}:{comm}",
            "suggest": "",
            "ts": event.ts,
        })

    async def _maybe_agentic_event(self, summary: str) -> bool:
        """Phase 5: отдать системное событие мозгу через Function Calling.

        Оркестратор формирует скрытый запрос ``[SYSTEM_EVENT: ...]``; роутер
        отправляет его в SYSTEM_OPS, и модель сама вызывает speak_response +
        execute_bash (например, cpupower frequency-set -g powersave).

        Возвращает True, если llama-server жив и событие обработано агентом.
        False — мозг недоступен, и вызывающий ОБЯЗАН сделать детерминированный
        fallback: защита железа N100 не должна зависеть от доступности LLM."""
        try:
            if not await self._llm.health():
                return False
            await self.process_intent(summary)
            return True
        except Exception:
            log.exception("agentic system-event failed: %s", summary[:80])
            return False

    async def on_daemon_alert(self, event: Event) -> None:
        line = str(event.data)
        await self._state("ALERT", line[:80])
        await self.process_intent(f"[DAEMON_ALERT] {line.strip()}")

    async def on_recon_alert(self, event: Event) -> None:
        """Wraith finding handler.

        Yellow (Wi-Fi vuln) — verbal heads-up only, the HUD already blinks.
        Red (intrusion)     — verbal warning + queue UFW BLOCK as a
                              destructive command so the standard confirmation
                              gate kicks in before any rule is applied.
        """
        data = event.data if isinstance(event.data, dict) else {}
        color = data.get("color", "yellow")
        summary = str(data.get("summary", "recon hit"))
        ufw = data.get("ufw_suggest")

        if color == "red":
            await self._state("ALERT", summary[:80])
            phrase = (
                f"Сэр, подозрительный трафик. {summary}. "
                + (f"Готовлю блокировку: {ufw}." if ufw else "Прикажете заблокировать?")
            )
            self.say(phrase, tone="alert")
            from src.ui.pixel_renderer import PixelBridge
            asyncio.create_task(
                PixelBridge().push_to_phone("Jarvis [INTRUSION]", summary[:120]),
                name="phone-push-recon",
            )
            await asyncio.to_thread(
                self.memory.remember,
                f"[RECON_ALERT/red] {summary}",
                ufw or "",
                "alerted",
                "intrusion",
                None,
                SIG_WARM,
            )
            if ufw:
                await self._gate_destructive(ufw, summary, None)
            return

        # Yellow — Wi-Fi reconnaissance opportunity, no automatic action.
        await self._state("ALERT", summary[:80])
        self.say(
            f"Сэр, в зоне доступа обнаружена уязвимость. {summary}. Вывожу данные на визор.",
            tone="alert",
        )
        await asyncio.to_thread(
            self.memory.remember,
            f"[RECON_ALERT/yellow] {summary}",
            "",
            "noticed",
            "recon",
            None,
            SIG_WARM,
        )

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

        # Per-sensor speech cooldown — не бубним одно и то же раз за разом.
        # Sentinel уже дедупит emit'ы, но если несколько sensor-ключей одновременно
        # активны, _os_alert_last_say защищает от nagging для каждого.
        now = time.time()
        cooldown = OS_ALERT_COOLDOWN_SEC.get(sensor, OS_ALERT_COOLDOWN_DEFAULT)
        last = self._os_alert_last_say.get(sensor, 0.0)
        if now - last < cooldown:
            await self._state("ALERT", f"{sensor}={value}{unit} [{level}]")
            return
        self._os_alert_last_say[sensor] = now

        await self._state("ALERT", f"{sensor}={value}{unit} [{level}]")
        if sensor == "thermal":
            temp_c = int(float(value))
            handled = False
            if level == "critical":
                # Phase 5: мозг сам решает и митигирует через Function Calling.
                # Детерминированная фраза ниже — fallback, если llama-server лёг.
                handled = await self._maybe_agentic_event(
                    f"[SYSTEM_EVENT: thermal={temp_c}C critical] "
                    "Перегрев SoC — требуется немедленная митигация."
                )
            if not handled:
                self.say(
                    f"Внимание. Температура {temp_c} градусов. "
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
        elif sensor == "disk":
            free_gb = data.get("free_gb")
            free_part = f" Свободно {free_gb} гигабайт." if free_gb is not None else ""
            self.say(
                f"Сэр, диск заполнен на {int(float(value))} процентов.{free_part} "
                f"Очистить кэш apt и старые логи?",
                tone="alert",
            )
        elif sensor == "memory":
            avail_mb = data.get("available_mb")
            avail_part = f" Свободно {avail_mb} мегабайт." if avail_mb is not None else ""
            self.say(
                f"Сэр, оперативная память на {int(float(value))} процентах.{avail_part} "
                f"Показать топ процессов?",
                tone="alert",
            )
        elif sensor == "battery":
            pct = int(float(value))
            if level == "critical":
                self.say(
                    f"Сэр, заряд аккумулятора {pct} процентов. "
                    "Подключите зарядку немедленно.",
                    tone="alert",
                )
            else:
                self.say(
                    f"Сэр, батарея на {pct} процентах. "
                    "Времени минут пятнадцать — пора к розетке.",
                    tone="alert",
                )
        elif sensor == "internet":
            reason = data.get("reason", "")
            mean_rtt = data.get("mean_rtt_ms")
            last_rtt = data.get("last_rtt_ms")
            if reason == "jitter" and mean_rtt and last_rtt:
                self.say(
                    f"Сэр, сеть лагает. Пинг подскочил с {int(mean_rtt)} "
                    f"до {int(last_rtt)} миллисекунд.",
                    tone="alert",
                )
            else:
                loss = int(float(value))
                self.say(
                    f"Сэр, потери пакетов {loss} процентов. "
                    "Соединение нестабильное.",
                    tone="alert",
                )
        elif sensor == "storage":
            # daemon_swarm pre-renders the phrase (add/remove + device name)
            phrase = data.get("phrase") or f"Storage {sensor}={value}"
            self.say(phrase, tone="normal")
        else:
            self.say(f"Системное событие: {sensor}, уровень {level}.", tone="alert")

        # Push critical alerts to the phone so the operator is notified even
        # when away from the machine.
        if level == "critical" and sensor in ("disk", "thermal", "memory", "battery"):
            from src.ui.pixel_renderer import PixelBridge
            asyncio.create_task(
                PixelBridge().push_to_phone(
                    f"Jarvis [{sensor}]",
                    f"{sensor}={value} — критический уровень.",
                ),
                name=f"phone-push-{sensor}",
            )

        # Warn/critical OS events live in the warm tier (5 days) so the LLM
        # still sees "yesterday's thermal event" in recall.
        sig = SIG_WARM if level in ("warn", "critical") else 0
        await asyncio.to_thread(
            self.memory.remember,
            f"[OS_EVENT] {sensor}={value} {level}",
            suggest,
            "warned",
            "os_event",
            None,
            sig,
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
            # NB: фактический mute/pause выполняет PixelBridge._pause_media()
            # ДО публикации этого события. Здесь только пишем в память факт
            # звонка — без лживой команды-побочки, которая раньше попадала
            # в Chrono Memory и путала RAG-recall'ы.
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel incoming call from {caller}",
                "",                                    # bash — пусто, действие не наше
                "media paused via pixel.py",
                "pixel_call",
                None,
                SIG_WARM,
            )
            return

        if kind == "clipboard":
            payload = str(data.get("payload", ""))
            self._last_clipboard = payload
            self._last_clipboard_ts = event.ts
            return

        if kind == "battery":
            # Routine snapshot from PixelBridge (every dbus refresh).
            # Silent — only the Context Weaver reads it.
            self._last_battery = {
                "charge": int(data.get("charge", 0)),
                "charging": bool(data.get("charging", False)),
                "device": str(data.get("device", "")),
            }
            self._last_battery_ts = event.ts
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
            self.say("Распознал изображение с телефона. Открываю результат в konsole.", tone="normal")
            if text:
                preview = text[:200].replace("\n", " ")
                await asyncio.to_thread(
                    self.memory.remember,
                    f"Pixel OCR: {path}", "konsole", preview, "pixel_ocr", None,
                )
            return

        if kind == "battery_low":
            charge = int(data.get("charge", 0))
            self._last_battery = {
                "charge": charge,
                "charging": bool(data.get("charging", False)),
                "device": str(data.get("device", "")),
            }
            self._last_battery_ts = event.ts
            await self._state("ALERT", f"Pixel battery {charge}%")
            self.say(
                f"Сэр, телефон разряжен. Заряд {charge} процентов.",
                tone="alert",
            )
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel battery low: {charge}%",
                "",
                "alerted",
                "pixel_battery",
                None,
            )
            return

        if kind == "otp":
            code = str(data.get("code", "")).strip()
            app = str(data.get("app", ""))
            if not code:
                return
            # Push to local clipboard so the operator can paste it without
            # reading off the HUD. xclip is the standard Wayland-friendly
            # path under XWayland; wl-copy is the native Wayland one.
            for tool, args in (
                ("wl-copy", ()),
                ("xclip", ("-selection", "clipboard")),
            ):
                if shutil.which(tool):
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            tool, *args,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await proc.communicate(code.encode())
                    except Exception:
                        log.exception("clipboard copy of OTP failed")
                    break
            await self._state("ALERT", f"OTP: {code}")
            # Speak digit-by-digit so Piper doesn't compress "123456" into
            # "сто двадцать три тысячи..." — much easier to type back.
            spaced = " ".join(code)
            app_hint = f" из {app}" if app else ""
            self.say(
                f"Код подтверждения{app_hint}: {spaced}. Скопирован в буфер.",
                tone="alert",
            )
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel OTP from {app}",
                "",
                code,
                "pixel_otp",
                None,
                SIG_WARM,
            )
            return

        if kind == "call_ended":
            caller = str(data.get("caller", "Unknown"))
            self._last_call = None
            self._last_call_ts = event.ts
            await self._state("IDLE", "call ended")
            # PixelBridge already issued playerctl --all-players play before
            # firing this event; we just log + remember + speak so the
            # operator hears the system clear itself.
            self.say("Звонок завершён. Возобновляю воспроизведение.", tone="normal")
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel call ended with {caller}",
                "",
                "media resumed",
                "pixel_call",
                None,
            )
            return

        if kind == "reminder":
            title = str(data.get("title", "")).strip() or "Pixel"
            body = str(data.get("body", "")).strip()
            app = str(data.get("app", "")).strip()
            # Mirror to KDE via notify-send so the reminder shows up in the
            # Plasma notification daemon even when Jarvis HUD is hidden.
            if shutil.which("notify-send"):
                try:
                    await asyncio.create_subprocess_exec(
                        "notify-send",
                        "--app-name=Jarvis",
                        "--icon=phone",
                        "--category=im.received",
                        title,
                        body,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                except Exception:
                    log.exception("notify-send for reminder failed")
            preview = (body or title)[:120]
            self.say(f"Напоминание с телефона. {preview}.", tone="normal")
            await asyncio.to_thread(
                self.memory.remember,
                f"Pixel reminder ({app}): {title}",
                "",
                body[:512],
                "pixel_reminder",
                None,
                SIG_WARM,
            )
            return

        payload = str(data.get("payload", ""))
        await self.process_intent(f"[PIXEL_{kind.upper()}] {payload}".strip())

    # ─────────────────── system wake handler ────────────────────────────────

    async def on_system_wake(self, event: Event) -> None:
        """Fires when the system resumes from suspend (login1 PrepareForSleep).

        Waits 3 seconds for services to stabilize, then gives a brief verbal
        status briefing. Also pushes the briefing to the phone if PixelBridge
        is available."""
        await asyncio.sleep(3.0)
        state = SystemState()
        snap = state.snapshot
        hour = time.localtime().tm_hour
        if 5 <= hour < 12:
            greeting = "Доброе утро"
        elif 12 <= hour < 18:
            greeting = "Добрый день"
        elif 18 <= hour < 23:
            greeting = "Добрый вечер"
        else:
            greeting = "Глубокая ночь"

        parts: list[str] = [f"{greeting}. Система восстановлена после сна."]

        if snap is not None:
            if snap.thermal >= 75:
                parts.append(f"Температура {int(snap.thermal)} градусов.")
            if snap.ram_pct >= 80:
                parts.append(f"ОЗУ на {int(snap.ram_pct)} процентах.")

        recent = list(self._recent_os)[-3:]
        unresolved = [e for e in recent if e.get("level") in ("warn", "critical")]
        if unresolved:
            sensors = ", ".join(e["sensor"] for e in unresolved)
            parts.append(f"Есть незакрытые события: {sensors}.")
        else:
            parts.append("Все системы в норме.")

        phrase = " ".join(parts)
        self.say(phrase, tone="normal")

        from src.ui.pixel_renderer import PixelBridge
        bridge = PixelBridge()
        asyncio.create_task(
            bridge.push_to_phone("Jarvis", phrase[:120]),
            name="wake-push-phone",
        )

    # ──────────────────── proactive status loop ──────────────────────────────

    PROACTIVE_INTERVAL = 1200  # 20 minutes of silence → volunteer a status

    async def _proactive_loop(self) -> None:
        """Checks every 60s if Jarvis has been silent for PROACTIVE_INTERVAL.

        Generates a proactive phrase via LLM based on live context — no
        hardcoded templates. The LLM sees system state and is asked to
        come up with a natural observation as Jarvis."""
        await asyncio.sleep(60)  # let boot complete before first check
        _last_proactive_ts: float = 0.0
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                if now - self._last_say_ts < self.PROACTIVE_INTERVAL:
                    continue
                # Не проактивничаем чаще чем раз в PROACTIVE_INTERVAL
                if now - _last_proactive_ts < self.PROACTIVE_INTERVAL:
                    continue
                state = SystemState()
                if state.load in (SystemLoad.HIGH, SystemLoad.CRITICAL):
                    continue  # system busy — don't add speech load now

                snap = state.snapshot

                # Контекст для LLM — живые метрики, недавние события
                context_parts: list[str] = []
                if snap is not None:
                    context_parts.append(
                        f"[System: load={state.load.value} "
                        f"cpu={snap.cpu_pct:.0f}% ram={snap.ram_pct:.0f}% "
                        f"thermal={snap.thermal:.0f}C "
                        f"time={time.strftime('%H:%M')}]"
                    )
                recent = list(self._recent_os)
                warns = [e for e in recent if e.get("level") in ("warn", "critical")]
                if warns:
                    sensors = ", ".join(dict.fromkeys(e["sensor"] for e in warns[-3:]))
                    context_parts.append(f"[RecentAlerts: {sensors}]")

                proactive_prompt = (
                    "\n".join(context_parts) + "\n"
                    "[User Intent: (проактивная инициатива Джарвиса — оператор давно молчит)]\n\n"
                    "Придумай одну короткую реплику от лица Джарвиса. "
                    "Она должна органично вытекать из контекста выше: "
                    "заметь что-то конкретное в системе или поведении оператора. "
                    "Никаких шаблонов «Сэр, краткий статус» — только живое наблюдение. "
                    "Ответь одной фразой живой речью, без markdown и без пояснений."
                )
                try:
                    # Лёгкий completion без tools — проактивной реплике не нужен
                    # function-calling, только одна фраза. CONVERSATION-промпт даёт
                    # персону, не нагружая модель системными правилами.
                    phrase = await self._llama_complete(
                        proactive_prompt,
                        system=self.router.system_prompt_for(IntentCategory.CONVERSATION),
                        max_tokens=120,
                        temperature=0.7,
                    )
                    phrase = phrase.strip()[:200]
                except Exception:
                    log.exception("proactive phrase generation failed")
                    phrase = ""

                if phrase:
                    _last_proactive_ts = time.time()
                    self.say(phrase, tone="idle")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("proactive loop error")

    def start_proactive_loop(self) -> asyncio.Task:
        return asyncio.create_task(self._proactive_loop(), name="jarvis-proactive")

