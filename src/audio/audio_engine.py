"""Acoustic Engine — кинематографический голосовой тракт JARVIS.

Цель: из лёгкого Piper TTS (Intel N100, без GPU) собрать голос уровня
кино-Джарвиса (Пол Беттани) — тёплый, плотный, кристально чёткий, с едва
уловимым синтетическим лоском «высокотехнологичной связи». Достигается двумя
слоями:

  1. КОГНИТИВНАЯ ПРОСОДИЯ (LLM-side): модель управляет ритмом пунктуацией
     (… = глубокая пауза/анализ, короткие фразы = динамика, точки = разделение
     критических данных) и может задать темп через speak_response(speed, pause)
     или инлайн-тег ``<say speed="1.1" pause="0.2">…</say>``. Просодия также
     зависит от состояния системы (IDLE = размеренно/аристократично,
     ALERT = быстро/по-военному).

  2. СТУДИЙНЫЙ DSP-ТРАКТ: Piper(stdout) → SoX(DSP) → aplay(stdin), без вывода
     сырого звука. Цепочка фильтров «Bettany Signature» (см. _DSP_CHAINS).

Пайплайн неблокирующий: speak() лишь кладёт фразу в очередь, единственный
рабочий поток проигрывает их последовательно (никакого наложения двух Piper).
ALERT прерывает текущую речь. PCM-поток ответвляется в PiperFFTPump → шина
AUDIO_FFT → сфера HUD пульсирует синхронно с голосом.

────────────────────────────────────────────────────────────────────────────
ИНЖЕНЕРНЫЕ ОТКЛОНЕНИЯ ОТ ИСХОДНОГО ТЗ (осознанные, ради сходства с фильмом):

  • highpass 120 → 90. Кино-Джарвис — тёплый БАРИТОН, а не тонкий «комм».
    Срез на 120 Гц убивает основной тон мужского голоса (~100–120 Гц) и делает
    его жестяным. 90 Гц убирает субсоник-гул, сохраняя грудь.
  • Добавлен pitch -80 cents. Голос «dmitry» — средний мужской; лёгкий сдвиг
    вниз даёт баритон-гравитас (центральная черта кино-голоса). Дёшево на N100.
  • equalizer 10000 → ~8000. Модель Piper = 22050 Гц (Найквист 11025). Пик на
    10 кГц поднимает в основном сибилянты/шум у самой границы. 8 кГц — «воздух»,
    который реально слышен на этой частоте дискретизации.
  • reverb: room-scale 100 → ~55, wet-gain 0 → −4 дБ. Цель — near-field «в шлеме»,
    а не размытый зал. Большой room-scale при wet 0 даёт washy-хвост.
  • chorus decay 0.4 → ~0.25. «Менее 1%, едва уловимо» (ТЗ). Сильный хорус =
    робот/Cylon — прочь от живого голоса актёра.
  • Добавлен финальный gain -n. В исходной цепочке не было выходного каскада:
    presence/air-бусты + reverb клиппят пик. Нормализация обязательна.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Optional

from src.audio.fft_analyzer import PiperFFTPump
from src.audio.segmentation import split_sentences
from src.common.event_bus import Event, EventBus, EventType, SystemLoad, SystemState

log = logging.getLogger("jarvis.acoustic")

SOX_BIN: Optional[str] = shutil.which("sox")
APLAY_BIN: Optional[str] = shutil.which("aplay")
# Стриминг TTS: предложения короче этого склеиваются с соседним, чтобы не
# дробить речь на интонационные обрывки («Да.», «Сэр.»). Порог низкий —
# нормальные короткие фразы («Цель захвачена.») стримим отдельно (в этом и смысл),
# склеиваем лишь совсем мелкие огрызки.
_STREAM_MIN_SENTENCE_CHARS = 12
_SAMPLE_RATE = 22050

# Ambient-источник света/proximity (внешний продьюсер пишет JSON). Если файл
# свеж (≤60 с) — используем; иначе fallback на час суток для night-режима.
_AMBIENT_FILE = Path("/tmp/jarvis_ambient.json")
_AMBIENT_FRESH_SEC = 60.0
_NIGHT_LUX_THRESHOLD = 30.0
_NIGHT_HOUR_START = 23
_NIGHT_HOUR_END = 6


class VoiceState(StrEnum):
    """Состояние, определяющее просодию и DSP-профиль."""
    IDLE = "IDLE"      # спокойно, размеренно, аристократично, чуть просторно
    NORMAL = "NORMAL"  # канонический кино-Джарвис по умолчанию
    ALERT = "ALERT"    # быстро, сухо, по-военному, максимально «впереди»
    NIGHT = "NIGHT"    # тёплый, тёмный, тихий — поздний час / темно


# Совместимость со старой tone-системой ядра (say(tone=...)).
_TONE_TO_STATE: dict[str, VoiceState] = {
    "idle": VoiceState.IDLE,
    "normal": VoiceState.NORMAL,
    "alert": VoiceState.ALERT,
    "night_stealth": VoiceState.NIGHT,
}


@dataclass(frozen=True, slots=True)
class Prosody:
    """Параметры Piper для одной фразы."""
    length_scale: float       # >1 медленнее, <1 быстрее (Piper duration multiplier)
    sentence_silence: float   # пауза между предложениями, сек


# Состояние → (speed, pause). speed — натуральный множитель (>1 быстрее);
# конвертируется в Piper length_scale = BASE/speed. Значения IDLE/ALERT — из ТЗ.
_STATE_PROSODY: dict[VoiceState, tuple[float, float]] = {
    VoiceState.IDLE:   (0.95, 0.40),
    VoiceState.NORMAL: (1.00, 0.28),
    VoiceState.ALERT:  (1.15, 0.10),
    VoiceState.NIGHT:  (0.92, 0.45),
}
_BASE_LENGTH_SCALE = 1.0
_LENGTH_SCALE_FLOOR = 0.55
_LENGTH_SCALE_CEIL = 1.5

# Сырой ввод/вывод SoX: S16_LE mono 22050 → S16_LE mono 22050 (raw↔raw).
_DSP_IO_ARGS: tuple[str, ...] = (
    "-q", "-V0",
    "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-",
    "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-",
)

# ── The Bettany Signature ─────────────────────────────────────────────────
# Порядок: rumble-cut → баритон-вес → де-муть/де-назал → presence/air →
# жёсткая компрессия (плотность) → микро-хорус (синтет.лоск) → near-field
# reverb → нормализация. См. шапку модуля про отклонения от исходного ТЗ.
_DSP_CHAINS: dict[VoiceState, tuple[str, ...]] = {
    VoiceState.NORMAL: (
        "highpass", "90",
        "pitch", "-80",
        "equalizer", "250",  "1.2q", "-2.5",   # де-муть/коробка
        "equalizer", "1800", "1.4q", "-1.5",   # де-назализация
        "equalizer", "3500", "0.7q", "4",      # presence — режет фон (ТЗ: 3500)
        "equalizer", "8000", "0.6q", "2",      # «воздух», осмысленный при 22кГц
        "compand", "0.1,0.2", "-60,-60,-30,-15,0,-3", "0", "-90", "0.1",  # жёсткая компрессия (ТЗ)
        "chorus", "0.85", "0.9", "50", "0.25", "0.3", "2", "-t",          # микро-хорус
        "reverb", "18", "50", "55", "100", "12", "-4",                    # near-field
        "gain", "-n", "-2",
    ),
    VoiceState.ALERT: (
        "highpass", "110",
        "pitch", "-50",
        "equalizer", "1800", "1.4q", "-1",
        "equalizer", "3500", "0.7q", "4.5",    # больше presence — срочность
        "equalizer", "8000", "0.6q", "2",
        "compand", "0.05,0.12", "-60,-60,-30,-12,0,-3", "0", "-90", "0.05",  # быстрее/плотнее
        "gain", "-n", "-1",                    # без reverb/хоруса — сухо и сразу
    ),
    VoiceState.IDLE: (
        "highpass", "85",
        "pitch", "-100",
        "equalizer", "250",  "1.2q", "-2.5",
        "equalizer", "1800", "1.4q", "-2",
        "equalizer", "3200", "0.7q", "3",      # presence мягче — отстранённее
        "equalizer", "7500", "0.6q", "1.5",
        "compand", "0.1,0.25", "-60,-60,-30,-15,0,-3", "0", "-90", "0.15",
        "chorus", "0.85", "0.9", "55", "0.22", "0.25", "2", "-t",
        "reverb", "24", "55", "65", "100", "16", "-4",   # чуть больше пространства
        "gain", "-n", "-3",
    ),
    VoiceState.NIGHT: (
        "highpass", "80",
        "pitch", "-90",
        "lowpass", "7000",                     # на ушах ночью — без резкого верха
        "equalizer", "250",  "1.2q", "-2",
        "equalizer", "1800", "1.2q", "-1.5",
        "equalizer", "3300", "0.8q", "2.5",
        "compand", "0.1,0.3", "-60,-60,-30,-15,0,-3", "0", "-90", "0.2",
        "chorus", "0.85", "0.9", "55", "0.2", "0.2", "2", "-t",
        "reverb", "14", "60", "45", "100", "8", "-5",    # крошечная тёплая комната
        "gain", "-n", "-8",                    # заметно тише
    ),
}

# ── Когнитивная просодия: инлайн-тег <say speed="X" pause="Y">…</say> ──────
_SAY_TAG_RE = re.compile(r"<say\b([^>]*)>(.*?)</say>", re.IGNORECASE | re.DOTALL)
_SAY_STRIP_RE = re.compile(r"</?say\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"(\w+)\s*=\s*[\"']([^\"']*)[\"']")


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_prosody(text: str) -> tuple[str, float | None, float | None]:
    """Извлечь просодию из инлайн-тега ``<say speed=".." pause="..">…</say>``.

    Возвращает (чистый_текст, speed|None, pause|None). Даже если тег битый или
    отсутствует — ВСЕГДА возвращает текст без любых ``<say>``-тегов (чтобы они
    никогда не были произнесены буквально). Это та самая «смерть текстового
    парсинга» под капотом: тег опционален, основной канал — параметры tool'а."""
    raw = (text or "").strip()
    speed: float | None = None
    pause: float | None = None
    m = _SAY_TAG_RE.search(raw)
    if m:
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        speed = _to_float(attrs.get("speed"))
        pause = _to_float(attrs.get("pause"))
        raw = m.group(2)
    # Срезаем любые оставшиеся say-теги (открытые/битые) — в речь они не попадут.
    raw = _SAY_STRIP_RE.sub("", raw).strip()
    return raw, speed, pause


def _speed_to_length_scale(speed: float) -> float:
    """speed (натуральный множитель, >1 быстрее) → Piper length_scale."""
    if speed <= 0:
        return _BASE_LENGTH_SCALE
    return max(_LENGTH_SCALE_FLOOR, min(_LENGTH_SCALE_CEIL, _BASE_LENGTH_SCALE / speed))


def resolve_prosody(
    state: VoiceState,
    speed: float | None = None,
    pause: float | None = None,
    load: SystemLoad | None = None,
) -> Prosody:
    """Собрать финальные Piper-параметры: дефолт состояния → переопределения →
    модуляция нагрузкой системы (под нагрузкой говорим быстрее/короче — не
    занимаем эфир, пока N100 занят компиляцией/тяжёлым GUI)."""
    d_speed, d_pause = _STATE_PROSODY.get(state, _STATE_PROSODY[VoiceState.NORMAL])
    eff_speed = d_speed if speed is None else float(speed)
    eff_pause = d_pause if pause is None else float(pause)
    length = _speed_to_length_scale(eff_speed)

    tier = load if load is not None else SystemState().load
    if tier == SystemLoad.HIGH:
        length *= 0.88
        eff_pause *= 0.6
    elif tier == SystemLoad.CRITICAL:
        length *= 0.78
        eff_pause *= 0.4
    length = max(_LENGTH_SCALE_FLOOR, length)
    return Prosody(round(length, 3), round(max(0.0, eff_pause), 3))


class AcousticEngine:
    """Кинематографический TTS-тракт. Владеет очередью речи, рабочим потоком,
    DSP-пайплайном и FFT-перекачкой для HUD.

    Связь с шиной: публикует STATE_CHANGE (SPEAKING/IDLE) и — через PiperFFTPump —
    AUDIO_FFT. Конструктор без тяжёлых side-effect'ов (поднимает только daemon-
    поток воркера), так что движок безопасно создаётся в тестах."""

    def __init__(
        self,
        bus: EventBus,
        *,
        piper_path: str,
        voice_model: str,
        voice_config: str,
        fft_pump: PiperFFTPump | None = None,
    ) -> None:
        self.bus = bus
        self.piper_path = piper_path
        self.voice_model = voice_model
        self.voice_config = voice_config
        self._fft_pump = fft_pump or PiperFFTPump(bus)

        # (text, state, speed|None, pause|None); None = сигнал остановки воркера.
        self._queue: "queue.Queue[tuple[str, VoiceState, float | None, float | None] | None]" = (
            queue.Queue()
        )
        self._active_procs: list[subprocess.Popen[bytes]] = []
        self._active_lock = threading.Lock()
        self._speaking = False
        self._sox_missing_warned = False
        # Стриминг TTS (Tier0 #2): длинную реплику режем на предложения и кладём
        # в очередь по одному — первое короткое предложение звучит раньше, чем
        # синтезируется вся реплика. OFF по умолчанию (между piper-процессами
        # появляется стык; включать осознанно через JARVIS_STREAM_TTS=1).
        self._stream_tts = os.getenv("JARVIS_STREAM_TTS", "0") == "1"

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="jarvis-acoustic"
        )
        self._worker.start()

    # ───────────────────────── public API ─────────────────────────
    def speak(
        self,
        text: str,
        state: VoiceState | str = VoiceState.NORMAL,
        *,
        speed: float | None = None,
        pause: float | None = None,
        interrupt: bool | None = None,
    ) -> None:
        """Поставить фразу в очередь озвучивания (неблокирующе).

        Инлайн-тег ``<say speed pause>`` в тексте парсится и переопределяет
        просодию (если явные speed/pause не заданы). ALERT по умолчанию
        прерывает текущую речь."""
        text = (text or "").strip()
        if not text:
            return
        vs = self._coerce_state(state)

        clean, tag_speed, tag_pause = parse_prosody(text)
        if not clean:
            return
        eff_speed = speed if speed is not None else tag_speed
        eff_pause = pause if pause is not None else tag_pause

        if interrupt is None:
            interrupt = vs == VoiceState.ALERT
        if interrupt:
            self._interrupt()

        # Стриминг: режем на предложения и ставим по одному (первое короткое
        # зазвучит раньше). Просодия/состояние одинаковы для всех кусков одной
        # реплики. min_chars склеивает обрывки, чтобы не дробить на «Да.»/«Сэр.».
        # OFF или одно предложение → прежний единичный путь (байт-в-байт).
        if self._stream_tts:
            segments = split_sentences(clean, min_chars=_STREAM_MIN_SENTENCE_CHARS)
            if len(segments) > 1:
                for seg in segments:
                    self._queue.put((seg, vs, eff_speed, eff_pause))
                return

        self._queue.put((clean, vs, eff_speed, eff_pause))

    def stop(self) -> None:
        """Немедленно оборвать текущую речь (kill piper/sox/aplay)."""
        self._interrupt()

    def is_speaking(self) -> bool:
        return self._speaking

    def shutdown(self) -> None:
        self._interrupt()
        self._queue.put(None)

    # ───────────────────────── internals ─────────────────────────
    @staticmethod
    def _coerce_state(state: VoiceState | str) -> VoiceState:
        if isinstance(state, VoiceState):
            return state
        s = str(state)
        if s in _TONE_TO_STATE:                 # "alert"/"idle"/"normal"
            return _TONE_TO_STATE[s]
        try:
            return VoiceState(s.upper())
        except ValueError:
            return VoiceState.NORMAL

    @staticmethod
    def _read_ambient() -> dict[str, Any] | None:
        try:
            if not _AMBIENT_FILE.is_file():
                return None
            import json
            data = json.loads(_AMBIENT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            ts = float(data.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        if time.time() - ts > _AMBIENT_FRESH_SEC:
            return None
        return data

    def _select_profile(self, state: VoiceState) -> VoiceState:
        """Выбрать DSP-профиль. ALERT никогда не приглушается ночью. Иначе:
        темно (lux<30) / телефон в кармане / поздний час → NIGHT."""
        if state == VoiceState.ALERT:
            return VoiceState.ALERT
        amb = self._read_ambient()
        if amb is not None:
            lux = _to_float(str(amb.get("light_lux", "1e6"))) or 1e6
            if lux < _NIGHT_LUX_THRESHOLD or bool(amb.get("proximity_near", False)):
                return VoiceState.NIGHT
        hour = time.localtime().tm_hour
        if hour >= _NIGHT_HOUR_START or hour < _NIGHT_HOUR_END:
            return VoiceState.NIGHT
        return state

    def _set_active(self, procs: list[subprocess.Popen[bytes]]) -> None:
        with self._active_lock:
            self._active_procs = [p for p in procs if p is not None]

    def _interrupt(self) -> None:
        with self._active_lock:
            procs = self._active_procs
            self._active_procs = []
        for p in procs:
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    def _worker_loop(self) -> None:
        """Единственный воркер: проигрывает фразы строго по одной. Сам управляет
        видимостью сферы HUD (SPEAKING при старте звука, IDLE по опустошению
        очереди) — синхронно с реальным звуком, а не с логикой LLM."""
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                text, state, speed, pause = item
                self._speaking = True
                self.bus.publish_threadsafe(
                    Event(EventType.STATE_CHANGE, ("SPEAKING", text[:60]))
                )
                try:
                    self._play(text, state, speed, pause)
                finally:
                    self._speaking = False
                    if self._queue.empty():
                        self.bus.publish_threadsafe(
                            Event(EventType.STATE_CHANGE, ("IDLE", ""))
                        )
                        self.bus.publish_threadsafe(Event(
                            EventType.AUDIO_FFT,
                            {"bands": [0.0] * 24, "level": 0.0, "ts": time.time()},
                        ))
            except Exception:
                log.exception("acoustic worker loop error")

    def _play(
        self, text: str, state: VoiceState, speed: float | None, pause: float | None
    ) -> None:
        """piper → (sox DSP) → aplay. PCM ответвляется в FFT-памп для сферы HUD."""
        if not Path(self.piper_path).is_file():
            log.error("Piper binary not found at %s — TTS disabled", self.piper_path)
            return
        if not Path(self.voice_model).is_file():
            log.error("Voice model not found at %s — TTS disabled", self.voice_model)
            return

        prosody = resolve_prosody(state, speed, pause)
        profile = self._select_profile(state)
        sox_args = _DSP_CHAINS.get(profile, _DSP_CHAINS[VoiceState.NORMAL])

        piper = sox = aplay = None
        try:
            # 1) Piper: текст → сырой PCM. bufsize=0 — без буферной задержки.
            piper = subprocess.Popen(
                [
                    self.piper_path,
                    "--model", self.voice_model,
                    "--config", self.voice_config,
                    "--output_raw",
                    "--length-scale", f"{prosody.length_scale:.2f}",
                    "--sentence-silence", f"{prosody.sentence_silence:.2f}",
                ],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )

            # 2) aplay: PCM → звуковая карта.
            if APLAY_BIN is not None:
                aplay = subprocess.Popen(
                    [APLAY_BIN, "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                )

            # 3) SoX DSP между piper и aplay (Bettany Signature).
            if SOX_BIN and aplay is not None:
                try:
                    sox = subprocess.Popen(
                        [SOX_BIN, *_DSP_IO_ARGS, *sox_args],
                        stdin=subprocess.PIPE, stdout=aplay.stdin,
                        stderr=subprocess.DEVNULL, bufsize=0,
                    )
                except Exception:
                    log.exception("sox DSP startup failed; raw piper→aplay fallback")
                    sox = None
                else:
                    # sox теперь владеет write-end'ом aplay.stdin. Закрываем
                    # родительскую копию, иначе aplay не получит EOF после выхода
                    # sox и aplay.wait() зависнет навсегда (классическая pipe-ловушка).
                    if aplay.stdin is not None:
                        try:
                            aplay.stdin.close()
                        except Exception:
                            pass
            elif aplay is not None and not SOX_BIN and not self._sox_missing_warned:
                self._sox_missing_warned = True
                log.warning(
                    "sox не найден в PATH — голос идёт БЕЗ кино-постобработки. "
                    "Установите: sudo apt install sox"
                )

            sink: IO[bytes] | None = (
                sox.stdin if sox is not None
                else (aplay.stdin if aplay is not None else None)
            )

            assert piper.stdin is not None and piper.stdout is not None
            self._set_active([p for p in (piper, sox, aplay) if p is not None])

            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()

            # FFT-памп тее'ит piper.stdout → sink и в анализатор (сфера HUD).
            self._fft_pump.start_pump(piper.stdout, sink, label=f"piper[{profile.value}]")

            if sox is not None:
                sox.wait()
            if aplay is not None:
                aplay.wait()
            piper.wait()
        except Exception:
            log.exception("acoustic _play pipeline failed")
        finally:
            self._set_active([])
            for p in (sox, aplay, piper):
                if p is not None and p.poll() is None:
                    try:
                        p.kill()
                    except Exception:
                        pass


__all__ = [
    "AcousticEngine",
    "VoiceState",
    "Prosody",
    "parse_prosody",
    "resolve_prosody",
]
