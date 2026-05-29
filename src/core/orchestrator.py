from __future__ import annotations

import asyncio
import logging
import random
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    Llama = None  # type: ignore
    _HAS_LLAMA_CPP = False

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

from src.audio.fft_analyzer import PiperFFTPump
from src.memory.ephemeral import EphemeralRunner, extract_python
from src.common.event_bus import Event, EventBus, EventType, SystemLoad, SystemState
from src.ui.window_manager import KWinOrchestrator
from src.memory.engine import SIG_WARM, ChronoMemory
from src.network.scanner import is_nmap_command, stream_nmap
from src.common.parser import (
    ParsedResponse,
    clean_bash,
    inject_sudo,
    parse_response,
    run_bash,
    wrap_sandbox,
)
from src.common.repair import QuickPatcher
from src.security.execution import ShadowExec, needs_shadow
from src.common.singleton import Singleton
from src.memory.snapshot import StateSnapshot, snapshot

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

# ───────────── In-process LLM (llama-cpp + iGPU offload) ─────────────
# ТЗ оператора: уходим с HTTP-Ollama на прямой in-process llama_cpp.Llama
# с GGUF-моделью и offload'ом всех слоёв в Intel iGPU (N100). HUD/шина и
# Jarvis-ядро живут в одном asyncio-процессе — блокирующий decode оборачиваем
# в ThreadPoolExecutor, чтобы event loop не тормозил под раздумьями ИИ.
#
# n_gpu_layers=-1 — все слои на iGPU (Q4_K_M 3B весит ~1.9 GiB,
#                   укладывается в shared VRAM N100 c запасом).
# n_threads=4    — N100 = 4 P-core, ровно одно ядро на токен decode пайплайна.
# n_ctx=4096     — экономим видеопамять; промпт-погон Context Weaver рассчитан
#                   ровно под этот бюджет (см. WEAVER_* константы ниже).
LLAMA_MODEL_NAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
LLAMA_MODEL_PATH = str(_PROJECT_ROOT / "models" / LLAMA_MODEL_NAME)
LLAMA_N_GPU_LAYERS = -1
LLAMA_N_THREADS = 4
# Контекст 4096 — по ТЗ оператора (8192/4096 на выбор; 4096 — безопасный
# баланс под N100 + Mesa Vulkan). n_batch/n_ubatch остаются урезанными:
# без них iGPU валится в vk::DeviceLostError даже при n_ctx=2048. flash_attn
# тоже выключен — этот combo стабилен по long-run тестам.
LLAMA_N_CTX = 8192
LLAMA_N_BATCH = 256
LLAMA_N_UBATCH = 128
LLAMA_FLASH_ATTN = False
LLAMA_MAX_TOKENS_DEFAULT = 1024
# KV-cache системного промпта: резервируем первые N слотов под static prefix.
# system_prompt в нашем проекте ~520 токенов. Добавим запас → 640.
# llama-cpp удерживает этот префикс в KV-кэше между запросами, не пересчитывая.
LLAMA_SYSTEM_CACHE_TOKENS = 640

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

TONE_PRESETS: dict[str, tuple[float, float]] = {
    "alert":  (0.85, 0.10),
    "normal": (1.00, 0.20),
    "idle":   (1.12, 0.40),
}

# ───────────── Acoustic Code «Bettany-Signature» (SOX DSP) ─────────────
# Голос Piper сам по себе — нейтральный TTS. Чтобы Джарвис звучал как
# Суверенный архитектор, а не как «Алиса в обёртке Kali», прогоняем
# raw-PCM через sox с custom-цепочкой эффектов:
#   - pitch shift вниз → массивность баритона (приблизительно формант-сдвиг;
#     честный формант-shift требует rubberband-cli и тяжелее на N100)
#   - low-mid резонанс (chest) и срез назальности (~1.5kHz)
#   - воздух/шелест на 8-12 kHz — «harmonic exciter» эффект
#   - параллельная компрессия (плотный + чистый микс) через compand
#   - phaser 0.1 Hz, eq depth → «квантовое поле», microvibrato
#   - gain нормализация
# Все профили — sox-аргументы без бинарника и формата (это в DSP_IO_ARGS).
SOX_BIN = shutil.which("sox")
DSP_IO_ARGS: tuple[str, ...] = (
    "-q", "-V0",
    "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-",
    "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-",
)
SOX_PROFILES: dict[str, tuple[str, ...]] = {
    # Обычный режим — глубокий, плотный баритон с шелестом и микровибрато.
    "normal": (
        "pitch", "-80",                                  # -80 cents ≈ полутон вниз
        "equalizer", "200",  "1.5q", "+3",               # грудной резонанс
        "equalizer", "1500", "1.0q", "-2",               # де-назализация
        "equalizer", "8000", "1.0q", "+4",               # air shimmer
        "equalizer", "12000","1.5q", "+3",               # crystalline sparkle
        "compand", "0.01,0.1", "6:-50,-30,-15", "-9", "-90", "0.1",  # parallel-comp approx
        "phaser",  "0.95", "0.8", "3", "0.5", "0.1", "-t",            # 0.1 Hz neural ring
        "gain", "-n", "-3",
    ),
    # Alert: сухой, быстрый, «металлический» — для ALERT state и destructive gate.
    "alert": (
        "pitch", "-40",
        "highpass", "120",
        "equalizer", "3000", "1.0q", "+5",               # presence
        "equalizer", "10000","1.5q", "+5",
        "compand", "0.001,0.05", "6:-50,-25,-10", "-6", "-90", "0.05",
        "gain", "-n", "-1",
    ),
    # Idle: глубже и медленнее — длинные паузы, для IDLE state.
    "idle": (
        "pitch", "-120",
        "equalizer", "200",  "1.5q", "+4",
        "equalizer", "1500", "1.0q", "-2",
        "equalizer", "8000", "1.0q", "+3",
        "compand", "0.02,0.15", "6:-50,-30,-15", "-10", "-90", "0.15",
        "phaser",  "0.95", "0.8", "3", "0.5", "0.08", "-t",
        "gain", "-n", "-4",
    ),
    # Night Stealth: тише, тёплее, нет «air» — если в комнате темно
    # (сенсор Pixel) или поздний час (≥23 или ≤6).
    "night_stealth": (
        "pitch", "-100",
        "lowpass", "3500",                                # никаких высоких на ушах
        "equalizer", "200", "1.5q", "+3",
        "compand", "0.02,0.2", "6:-50,-30,-15", "-12", "-90", "0.2",
        "gain", "-n", "-10",                              # на 7 дБ тише
    ),
}

# Источник данных о свете/proximity — пока KDE Connect не экспортирует
# sensor data штатно. Договорённость: внешний продьюсер (Termux/MQTT-bridge
# /кастомный плагин) пишет JSON в /tmp/jarvis_ambient.json вида:
#   {"light_lux": 12.0, "proximity_near": false, "ts": 1700000000.0}
# Если файл существует и свеж (≤ 60 сек), значения используем — иначе
# мягкий fallback на час дня.
AMBIENT_FILE = Path("/tmp/jarvis_ambient.json")
AMBIENT_FRESH_SEC = 60.0
NIGHT_LUX_THRESHOLD = 30.0   # < 30 lux ≈ темно
NIGHT_HOUR_START = 23
NIGHT_HOUR_END = 6

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
        self._fft_pump = PiperFFTPump(self.bus)
        self.model = LLAMA_MODEL_NAME
        try:
            self.system_prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
        except FileNotFoundError:
            log.error("system_prompt missing at %s — using minimal fallback", SYSTEM_PROMPT_FILE)
            self.system_prompt = "Ты — JARVIS, краткий ассистент. Отвечай по делу."

        # llama-cpp lazy-load: модель тяжёлая (~1.9 GiB GGUF + iGPU layers),
        # грузим на ПЕРВЫЙ запрос, не на boot — оператору не нужно ждать
        # старта Jarvis ради одной верхней команды. Lock защищает от гонки
        # двух одновременных process_intent при холодном кэше.
        self._llama: Any | None = None  # llama_cpp.Llama
        self._llama_load_lock = asyncio.Lock()
        # Single-worker executor: llama_cpp.Llama НЕ thread-safe; одновременных
        # decode'ов быть не должно. Отдельный pool вместо default, чтобы
        # asyncio.to_thread на побочные вещи (run_bash, snapshot) не толкался
        # с inference-петлёй.
        from concurrent.futures import ThreadPoolExecutor
        self._llama_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jarvis-llama"
        )
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

        # ── TTS sequential queue ────────────────────────────────────────────
        # Единственный воркер читает из очереди строки и озвучивает одну
        # за другой — никакого наслоения двух Piper-процессов.
        import queue as _q
        self._tts_queue: "_q.Queue[tuple[str,str]|None]" = _q.Queue()
        # Текущий активный piper-процесс — хранится чтобы можно было убить
        # его при поступлении alert-приоритетного сообщения.
        self._current_piper: Optional[Any] = None
        self._current_piper_lock = threading.Lock()
        # Флаг реального воспроизведения — HUD читает через шину AUDIO_FFT;
        # этот флаг нужен чтобы _tts_worker_loop мог сигнализировать начало/конец.
        self._tts_playing: bool = False
        self._tts_worker_thread = threading.Thread(
            target=self._tts_worker_loop, daemon=True, name="jarvis-tts"
        )
        self._tts_worker_thread.start()

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

    # ───── Ambient awareness & SOX profile selector ─────
    @staticmethod
    def _read_ambient() -> dict[str, Any] | None:
        """Возвращает свежий ambient-снимок (light_lux, proximity_near) если есть.

        Источник — внешний продьюсер пишет JSON в /tmp/jarvis_ambient.json.
        KDE Connect штатно sensor data не отдаёт; договорённость
        document'ирована в SOX_PROFILES — там же fallback на час суток."""
        try:
            if not AMBIENT_FILE.is_file():
                return None
            import json
            data = json.loads(AMBIENT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            ts = float(data.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        if time.time() - ts > AMBIENT_FRESH_SEC:
            return None
        return data

    def _sox_profile_name(self, tone: str) -> str:
        """Решает каким DSP-профилем озвучивать ответ.

        Приоритет (от высшего к низшему):
          1. tone == 'alert' — всегда alert-профиль (никакая темнота
             не приглушит звуковую тревогу).
          2. ambient.light_lux < 30 → night_stealth.
          3. proximity_near=true (телефон в кармане) → night_stealth
             (лаконичнее и тише — оператор не у экрана).
          4. Час ≥ 23 или ≤ 6 → night_stealth (fallback без сенсора).
          5. Иначе tone (normal/idle)."""
        if tone == "alert":
            return "alert"
        amb = self._read_ambient()
        if amb is not None:
            try:
                lux = float(amb.get("light_lux", 1e6))
            except (TypeError, ValueError):
                lux = 1e6
            if lux < NIGHT_LUX_THRESHOLD:
                return "night_stealth"
            if bool(amb.get("proximity_near", False)):
                return "night_stealth"
        hour = time.localtime().tm_hour
        if hour >= NIGHT_HOUR_START or hour < NIGHT_HOUR_END:
            return "night_stealth"
        return tone if tone in SOX_PROFILES else "normal"

    def ack(self, category: str) -> str:
        return random.choice(ACK_VARIANTS.get(category, ("ok",)))

    def say(self, text: str, tone: str | None = None) -> None:
        """Enqueue text for sequential TTS. All speech goes through _tts_worker_loop
        so piper processes never overlap. Alert tone interrupts current speech."""
        text = (text or "").strip()
        if not text:
            return
        self._last_say_ts = time.time()
        chosen = tone or self._tone_for_state()

        # Alert tone: прерываем текущую речь, ставим вперёд очереди
        if chosen == "alert":
            self._interrupt_current_speech()

        self._tts_queue.put((text, chosen))

    def _interrupt_current_speech(self) -> None:
        """Kill the running piper/sox/aplay pipeline immediately."""
        with self._current_piper_lock:
            procs = self._current_piper
            self._current_piper = None
        if procs is None:
            return
        for p in procs:
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    def _tts_worker_loop(self) -> None:
        """Single-worker TTS loop. Consumes (text, tone) pairs from _tts_queue
        and speaks them one at a time — no overlapping piper processes.

        Критически важно: этот воркер сам управляет видимостью сферы HUD.
        Перед _play_tts публикует STATE_CHANGE→SPEAKING, после — IDLE.
        Только так сфера появляется и гаснет синхронно с реальным звуком,
        а не с логикой LLM (которая переходит в IDLE раньше конца речи)."""
        while True:
            try:
                item = self._tts_queue.get()
                if item is None:
                    break
                text, tone = item
                # Сигналим шине: начинаем говорить — сфера появляется
                self._tts_playing = True
                self.bus.publish_threadsafe(
                    Event(EventType.STATE_CHANGE, ("SPEAKING", text[:60]))
                )
                try:
                    self._play_tts(text, tone)
                finally:
                    self._tts_playing = False
                    # Если очередь пуста — возвращаемся в IDLE и гасим сферу
                    if self._tts_queue.empty():
                        self.bus.publish_threadsafe(
                            Event(EventType.STATE_CHANGE, ("IDLE", ""))
                        )
                        # Явный нулевой FFT-кадр — HUD сразу гасит сферу
                        self.bus.publish_threadsafe(
                            Event(EventType.AUDIO_FFT, {"bands": [0.0] * 24, "level": 0.0, "ts": time.time()})
                        )
            except Exception:
                log.exception("tts worker loop error")

    def _play_tts(self, text: str, chosen: str) -> None:
        length_scale, sentence_silence = TONE_PRESETS.get(chosen, TONE_PRESETS["normal"])

        # Symbiote modulation: when the box is under load, talk faster
        # with shorter pauses — the user is busy with a compile or a
        # heavy GUI app and we shouldn't hog the air. CRITICAL goes
        # even faster. Floor at 0.55 so we don't chipmunk.
        load = SystemState().load
        if load == SystemLoad.HIGH:
            length_scale *= 0.88
            sentence_silence *= 0.6
        elif load == SystemLoad.CRITICAL:
            length_scale *= 0.78
            sentence_silence *= 0.4
        length_scale = max(0.55, length_scale)

        # Зафиксируем DSP-профиль СЕЙЧАС — состояние системы могло смениться
        sox_profile_name = self._sox_profile_name(chosen)
        sox_args = SOX_PROFILES.get(sox_profile_name, SOX_PROFILES["normal"])

        if not Path(PIPER_PATH).is_file():
            log.error(
                "Piper binary not found at %s — TTS disabled. "
                "Положите бинарник в %s/piper/piper",
                PIPER_PATH, _PROJECT_ROOT,
            )
            return
        if not Path(VOICE_MODEL).is_file():
            log.error(
                "Voice model not found at %s — TTS disabled.",
                VOICE_MODEL,
            )
            return
        if not Path(VOICE_CONFIG).is_file():
            log.error(
                "Voice config not found at %s — TTS disabled.",
                VOICE_CONFIG,
            )
            return

        piper = sox = aplay = None
        try:
            # ─── 1) piper: text → raw PCM ───
            piper = subprocess.Popen(
                [
                    PIPER_PATH,
                    "--model", VOICE_MODEL,
                    "--config", VOICE_CONFIG,
                    "--output_raw",
                    "--length-scale", f"{length_scale:.2f}",
                    "--sentence-silence", f"{sentence_silence:.2f}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # ─── 2) aplay: PCM → soundcard ───
            try:
                aplay = subprocess.Popen(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                aplay = None

            # ─── 3) sox DSP между piper и aplay ───
            if SOX_BIN and aplay is not None:
                try:
                    sox = subprocess.Popen(
                        [SOX_BIN, *DSP_IO_ARGS, *sox_args],
                        stdin=subprocess.PIPE,
                        stdout=aplay.stdin,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    log.exception("sox DSP startup failed; fallback to direct piper→aplay")
                    sox = None

            sink_stdin = (sox.stdin if sox is not None
                          else (aplay.stdin if aplay is not None else None))

            assert piper.stdin is not None and piper.stdout is not None
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()

            # Регистрируем процессы для возможного прерывания
            with self._current_piper_lock:
                self._current_piper = [p for p in (piper, sox, aplay) if p is not None]

            self._fft_pump.start_pump(piper.stdout, sink_stdin, label=f"piper[{sox_profile_name}]")

            if sox is not None:
                sox.wait()
            if aplay is not None:
                aplay.wait()
            piper.wait()
        except Exception:
            log.exception("_play_tts() pipeline failed")
        finally:
            with self._current_piper_lock:
                self._current_piper = None
            for p in (sox, aplay, piper):
                if p is not None and p.poll() is None:
                    try:
                        p.kill()
                    except Exception:
                        pass

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

    # ───────────── llama-cpp engine ─────────────
    async def _ensure_llama(self) -> bool:
        """Idempotent lazy load of the GGUF brain.

        Возвращает True, если ``self._llama`` готов к ``create_chat_completion``.
        Грузим в ``asyncio.to_thread`` — Llama() в конструкторе делает mmap
        весов + iGPU-инициализацию (на N100 ~3-6 секунд), это блокировало
        бы event loop. Lock защищает: если две process_intent гонкой попали
        в _ensure_llama до прогрева, вторая просто ждёт первую."""
        if self._llama is not None:
            return True
        if not _HAS_LLAMA_CPP:
            log.error(
                "llama-cpp-python не установлен — мозг отключён. "
                "Запустите ./setup_igpu.sh"
            )
            return False
        if not Path(LLAMA_MODEL_PATH).is_file():
            log.error(
                "GGUF-модель не найдена: %s — запустите ./setup_igpu.sh",
                LLAMA_MODEL_PATH,
            )
            return False
        async with self._llama_load_lock:
            if self._llama is not None:
                return True
            log.info(
                "loading %s (n_gpu_layers=%d, n_threads=%d, n_ctx=%d, n_batch=%d, flash_attn=%s) — N100 iGPU offload",
                LLAMA_MODEL_NAME, LLAMA_N_GPU_LAYERS, LLAMA_N_THREADS,
                LLAMA_N_CTX, LLAMA_N_BATCH, LLAMA_FLASH_ATTN,
            )
            try:
                self._llama = await asyncio.to_thread(
                    Llama,
                    model_path=LLAMA_MODEL_PATH,
                    n_gpu_layers=LLAMA_N_GPU_LAYERS,
                    n_threads=LLAMA_N_THREADS,
                    n_ctx=LLAMA_N_CTX,
                    n_batch=LLAMA_N_BATCH,
                    n_ubatch=LLAMA_N_UBATCH,
                    flash_attn=LLAMA_FLASH_ATTN,
                    chat_format="chatml",   # Qwen2.5 — ChatML; форсим явно,
                                            # чтобы не зависеть от metadata конкретного билда GGUF.
                    verbose=False,
                )
                log.info("LLM ready: %s", LLAMA_MODEL_NAME)
                # Прогрев KV-кэша системного промпта: делаем один холостой запрос
                # с system+пустым user prompt, чтобы llama-cpp зафиксировал
                # system-prefix в KV и не пересчитывал его при каждом интенте.
                await asyncio.to_thread(self._warm_system_prompt_cache)
                return True
            except Exception:
                log.exception("Llama(%s) load failed", LLAMA_MODEL_PATH)
                self._llama = None
                return False

    def _warm_system_prompt_cache(self) -> None:
        """Прогреваем KV-кэш системного промпта одним холостым вызовом.

        llama-cpp хранит prefill-результат в KV-кэше. Если system-сообщение
        одинаково между запросами — следующий create_chat_completion просто
        «до-префилит» только user-часть. Без этого каждый запрос тратит
        ~0.5 с на пересчёт 520+ токенов system-промпта заново.
        Вызывается синхронно ВНУТРИ llama_executor-треда, поэтому self._llama
        жив и thread-safe."""
        if self._llama is None:
            return
        try:
            # Минимальный холостой запрос: system + пустой user, max_tokens=1.
            # Цель — не получить ответ, а прогреть prefill-кэш.
            self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": "init"},
                ],
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
            log.info(
                "LLM system-prompt KV-cache warmed (%d chars / ~%d tokens)",
                len(self.system_prompt),
                LLAMA_SYSTEM_CACHE_TOKENS,
            )
        except Exception:
            log.warning("LLM KV-cache warm failed (non-fatal)", exc_info=True)

    async def _llama_stream(
        max_tokens: int = LLAMA_MAX_TOKENS_DEFAULT,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield-каждый-токен generator поверх блокирующего llama-cpp.

        llama_cpp.Llama.create_chat_completion(stream=True) — sync-итератор,
        каждый next() считает следующий токен на CPU/iGPU. Чтобы asyncio
        loop не фризил под decode'ом, producer крутится в нашем single-worker
        executor'е и пушит токены через ``loop.call_soon_threadsafe`` в
        ``asyncio.Queue``; основная корутина их await'ит и отдаёт наружу.

        Sentinel-объект завершает поток — обычный None не годится, llama-cpp
        иногда генерирует пустые delta, которые мы хотим отличать от EOF."""
        if not await self._ensure_llama():
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        SENTINEL = object()

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _producer() -> None:
            try:
                # ── Token budget guard ─────────────────────────────────────────
                # Runs synchronously inside the single-worker executor where
                # self._llama lives. tokenize() is read-only on model weights,
                # safe to call here. Prevents ValueError: Requested tokens exceed
                # context window that happened with large memory/system blocks.
                budget = LLAMA_N_CTX - max_tokens - 16  # 16 = ChatML framing overhead
                total_toks = 3  # BOS + initial assistant marker
                for m in messages:
                    raw = (m.get("content") or "").encode()
                    try:
                        total_toks += len(
                            self._llama.tokenize(raw, add_bos=False, special=False)  # type: ignore[union-attr]
                        ) + 4  # per-message ChatML tokens
                    except Exception:
                        total_toks += len(raw) // 3 + 4

                if total_toks > budget:
                    over = total_toks - budget
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            content = messages[i]["content"]
                            try:
                                toks = self._llama.tokenize(  # type: ignore[union-attr]
                                    content.encode(), add_bos=False, special=False
                                )
                                keep = max(40, len(toks) - over - 10)
                                kept = self._llama.detokenize(toks[:keep]).decode("utf-8", errors="replace")  # type: ignore[union-attr]
                                messages[i] = {**messages[i], "content": kept + "\n[…обрезано…]"}
                                log.warning(
                                    "LLM: user prompt trimmed %d→%d tokens (ctx=%d budget=%d)",
                                    len(toks), keep, LLAMA_N_CTX, budget,
                                )
                            except Exception:
                                trim_chars = (over + 20) * 4
                                messages[i] = {
                                    **messages[i],
                                    "content": content[: max(100, len(content) - trim_chars)] + "\n[…обрезано…]",
                                }
                            break
                # ── end token guard ────────────────────────────────────────────

                stream = self._llama.create_chat_completion(  # type: ignore[union-attr]
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if content:
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception:
                log.exception("llama-cpp producer crashed")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        loop.run_in_executor(self._llama_executor, _producer)
        while True:
            item = await queue.get()
            if item is SENTINEL:
                return
            yield item

    async def _llama_complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = LLAMA_MAX_TOKENS_DEFAULT,
        temperature: float = 0.0,
    ) -> str:
        """Non-streaming convenience: collect the full stream into a string.
        Используется адаптером совместимости (Mnemosyne)."""
        chunks: list[str] = []
        async for tok in self._llama_stream(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature,
        ):
            chunks.append(tok)
        return "".join(chunks)

    async def _generate_streaming(
        self,
        user_text: str,
        past: list[dict],
        current_snap: StateSnapshot | dict | None = None,
        extra: str = "",
    ) -> str:
        # current_snap is accepted for backwards compat with self-heal paths
        # but no longer used — SystemState() is the live source of truth.
        del current_snap
        context = self.build_context(user_text, past)
        prompt = f"{context}\n{extra}" if extra else context
        chunks: list[str] = []
        try:
            async for tok in self._llama_stream(
                prompt, system=self.system_prompt,
                max_tokens=LLAMA_MAX_TOKENS_DEFAULT, temperature=0.0,
            ):
                chunks.append(tok)
                await self.bus.publish(Event(EventType.TOKEN_STREAM, tok))
        except Exception:
            log.exception("llama-cpp streaming failed")
        return "".join(chunks)

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

        past = await asyncio.to_thread(self.memory.recall, text)
        raw = await self._generate_streaming(text, past, current_snap=snap)
        parsed: ParsedResponse = parse_response(raw)

        await self._state("SPEAKING", parsed.thought or parsed.say or text[:60])
        if parsed.say:
            self.say(parsed.say)

        # KWin action: LLM asked for a window-manager operation via <kwin>name</kwin>.
        # name must match a script in scripts/<name>.js. Runs in parallel with any
        # <bash>/<python> the same response carried.
        if parsed.kwin:
            kwin_name = parsed.kwin.strip()
            ok = await self.kwin.execute(kwin_name)
            await asyncio.to_thread(
                self.memory.remember,
                text,
                f"<kwin>{kwin_name}</kwin>",
                "ok" if ok else "fail",
                "kwin",
                snap,
            )
            if not ok:
                log.warning("kwin script '%s' not found or failed", kwin_name)

        # Ephemeral Programming: if the LLM authored a <python>...</python>
        # block, run it through the sandbox runner. The script's last-line
        # JSON becomes a synthesized <thought> in memory.
        if extract_python(raw) is not None:
            try:
                eph = await self._ephemeral.run_from_llm(raw)
            except Exception:
                log.exception("ephemeral runner crashed")
                eph = None
            if eph is not None:
                outcome = (
                    f"ephemeral rc={eph.rc} engine={eph.engine} "
                    f"thought={(eph.parsed or {}).get('thought', '')[:140]}"
                )
                speak = (eph.parsed or {}).get("speak")
                if isinstance(speak, str) and speak.strip():
                    self.say(speak.strip()[:200])
                await asyncio.to_thread(
                    self.memory.remember, text, "<python ephemeral>", outcome, "ephemeral", snap,
                )

        bash = parsed.bash
        if bash and parsed.sandbox:
            bash = wrap_sandbox(bash)
            log.info("sandbox wrap applied: %s", bash[:200])
            await self._state("THINKING", "Sandbox engaged")

        rc, stdout, stderr = (0, "", "")
        if bash:
            # All LLM-emitted bash dry-runs in Shadow Exec first — there is no
            # static command book anymore, so this is the only safety net.
            if needs_shadow(bash):
                shadow_kind = await self._shadow_then_real(bash, text, snap)
                if shadow_kind == "queued":
                    return raw
                if shadow_kind == "rejected":
                    await self._state("IDLE", "")
                    return raw
                # shadow_kind == "direct" → fall through into the standard
                # execute_with_healing path so the regular destructive-gate
                # and quick-patch loop still apply.
            rc, stdout, stderr = await self._execute_with_healing(bash, text, current_snap=snap)

        result = stdout.strip() or stderr.strip() or (
            f"rc={rc}" if rc is not None else "detached"
        )
        kind = "qwen_sandbox" if parsed.sandbox else "qwen"
        await asyncio.to_thread(self.memory.remember, text, bash, result, kind, snap)
        await self._state("IDLE", "")
        return raw

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
            from pixel import PixelBridge
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
            from pixel import PixelBridge
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

        from pixel import PixelBridge
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
                    "Ответь ТОЛЬКО тегом <say>...</say> с одной фразой, без другого текста."
                )
                try:
                    raw = await self._llama_complete(
                        proactive_prompt,
                        system=self.system_prompt,
                        max_tokens=120,
                        temperature=0.7,
                    )
                    from parser import parse_response as _parse
                    parsed = _parse(raw)
                    phrase = (parsed.say or "").strip()
                    if not phrase:
                        # Если LLM не обернул в тег — берём весь ответ как есть
                        phrase = raw.strip()[:200]
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

