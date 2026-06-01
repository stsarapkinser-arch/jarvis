from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

# Аудио-стек (sounddevice/vosk) — опционален на уровне импорта: без него модуль
# всё равно импортируется (чистая логика гейта/дедупа тестируема), а слушатель
# уходит в standby. На целевой машине (N100) обе библиотеки установлены.
try:
    import sounddevice as sd
except Exception:  # noqa: BLE001 — ALSA/портаудио могут падать по-разному
    sd = None  # type: ignore[assignment]
try:
    from vosk import KaldiRecognizer, Model
except Exception:  # noqa: BLE001
    KaldiRecognizer = None  # type: ignore[assignment,misc]
    Model = None  # type: ignore[assignment,misc]

from src.audio.wakeword import build_gate
from src.common.event_bus import Event, EventBus, EventType
from src.common.singleton import Singleton

log = logging.getLogger("jarvis.voice")

WAKE_WORDS: Final[tuple[str, ...]] = ("джарвис", "jarvis", "компьютер")
SAMPLE_RATE: Final = 16000
BLOCK_SIZE: Final = 16000
READ_FRAMES: Final = 8000
# Окно дедупликации: одинаковый интент в пределах N секунд считаем
# одним (partial → final одной фразы). 6 с покрывает паузу распознавания.
INTENT_DEDUP_SEC: Final = 6.0
# Эхо-гейт: пока Джарвис говорит, глушим вход микрофона, иначе он слышит сам
# себя и само-триггерится. Закрываем гейт по STATE_CHANGE=SPEAKING и продлеваем
# «хвост» на каждый звуковой AUDIO_FFT-фрейм (~раз в 23 мс). Открываем не по
# IDLE (его шлёт и оркестратор — гонка порядка событий), а по истечении хвоста
# после последнего звукового фрейма. Хвост покрывает паузы между словами и
# затухание/буфер aplay.
ECHO_GATE_TAIL_SEC: Final = 0.5
_FFT_GATE_LEVEL: Final = 1e-3

# Папка с весами Vosk относительно корня проекта.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "model")

# Сообщение оператору, когда модель не загружена. Дублируется в HUD
# (STATE_CHANGE → thinking_text) и в лог.
VOSK_MISSING_MSG = "Сэр, загрузите модель Vosk в папку /model"


class JarvisMain(metaclass=Singleton):
    """Vosk audio listener. Runs sync in a thread, publishes VOICE_INTENT events."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Model | None = None
        self.rec: KaldiRecognizer | None = None
        self.wake_words = WAKE_WORDS
        # Эхо-гейт: monotonic-таймштамп, до которого вход заглушён. Пишется из
        # event-loop (обработчики шины), читается из треда слушателя — для
        # float это атомарно в CPython, блокировка не нужна.
        self._gate_until: float = 0.0
        # Wake-word пред-гейт (Tier0 #5): по умолчанию выключен и прозрачен
        # (feed() → True всегда), поэтому конструктор безопасен и в тестах, и
        # без openwakeword. Включается JARVIS_WAKEWORD=1.
        self._wake_gate = build_gate()
        # Аудио-стек не установлен — тихо в standby (модуль импортируем, но
        # слушать нечем). На N100 сюда не попадаем.
        if Model is None or KaldiRecognizer is None:
            log.warning("vosk/sounddevice недоступны — voice disabled")
            return
        # Ленивая загрузка: если папки нет, не падаем — переходим в режим
        # ожидания. Vosk.Model() при отсутствии бросает Exception('Failed to
        # create a model') и без try-блока валит весь сервис.
        if not Path(model_path).is_dir():
            log.error("Vosk model directory missing at %s — voice disabled", model_path)
            return
        try:
            self.model = Model(model_path)
            self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        except Exception:
            log.exception("Vosk init failed at %s — voice disabled", model_path)
            self.model = None
            self.rec = None

    def _extract_intent(self, text: str) -> str | None:
        for ww in self.wake_words:
            if ww in text:
                return text.replace(ww, "").strip()
        return None

    async def _on_acoustic_state(self, event: Event) -> None:
        """STATE_CHANGE: на SPEAKING мгновенно закрываем гейт. IDLE намеренно
        НЕ открывает гейт (его шлёт и оркестратор из process_intent — порядок
        с SPEAKING не гарантирован); открытие — по истечении хвоста."""
        data = event.data
        state = data[0] if isinstance(data, (tuple, list)) and data else data
        if state == "SPEAKING":
            self._gate_until = time.monotonic() + ECHO_GATE_TAIL_SEC

    async def _on_audio_fft(self, event: Event) -> None:
        """Пока сфера реально звучит (level>0), держим вход закрытым: продлеваем
        хвост на каждый фрейм. Финальный нулевой фрейм хвост не трогает → гейт
        сам открывается через ECHO_GATE_TAIL_SEC после конца речи."""
        data = event.data
        level = 0.0
        if isinstance(data, dict):
            try:
                level = float(data.get("level", 0.0))
            except (TypeError, ValueError):
                level = 0.0
        if level > _FFT_GATE_LEVEL:
            self._gate_until = time.monotonic() + ECHO_GATE_TAIL_SEC

    def _gated(self) -> bool:
        return time.monotonic() < self._gate_until

    def run(self, bus: EventBus | None = None) -> None:
        bus = bus or EventBus()
        # Режим ожидания: модели нет — извещаем HUD и тихо выходим из треда.
        # Сервис при этом остаётся живой, оператор подкладывает веса и
        # перезапускает unit.
        if self.model is None or self.rec is None or sd is None:
            log.warning("voice loop standby: %s", VOSK_MISSING_MSG)
            try:
                bus.publish_threadsafe(
                    Event(EventType.STATE_CHANGE, ("ALERT", VOSK_MISSING_MSG))
                )
            except Exception:
                log.exception("could not publish Vosk-missing alert")
            print(f"⚠ {VOSK_MISSING_MSG}")
            return

        # Эхо-гейт: слушаем состояние голоса и спектр, чтобы глушить вход, пока
        # Джарвис говорит. Подписки безвредны при отсутствии loop (в standby мы
        # сюда не доходим — выходим раньше).
        bus.subscribe(EventType.STATE_CHANGE, self._on_acoustic_state)
        bus.subscribe(EventType.AUDIO_FFT, self._on_audio_fft)

        log.info("voice listener ready")
        print("🎙 Джарвис слушает...")
        # Дедуп: храним последний ОПУБЛИКОВАННЫЙ интент и его время. Без этого
        # одна фраза стреляла дважды — сначала на partial-результате (мы ловим
        # команды без паузы), затем на final-результате с тем же текстом →
        # Джарвис отвечал на одно и то же два раза.
        _last_published: str = ""
        _last_published_ts: float = 0.0

        def _publish_intent(intent: str) -> None:
            nonlocal _last_published, _last_published_ts
            now = time.monotonic()
            # Тот же интент в пределах окна — это partial→final дубль, глушим.
            if intent == _last_published and (now - _last_published_ts) < INTENT_DEDUP_SEC:
                return
            _last_published = intent
            _last_published_ts = now
            bus.publish_threadsafe(Event(EventType.VOICE_INTENT, intent))

        while True:
            try:
                with sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    dtype="int16",
                    channels=1,
                ) as stream:
                    _was_gated = False
                    while True:
                        data, _ = stream.read(READ_FRAMES)
                        # Эхо-гейт: Джарвис говорит — дренируем поток (иначе
                        # переполнится буфер), но в распознаватель НЕ отдаём.
                        if self._gated():
                            _was_gated = True
                            continue
                        if _was_gated:
                            # Сбрасываем накопленные на границе гейта огрызки
                            # (хвост собственной речи/тишины), чтобы не выдать
                            # мусорный partial первым же кадром после гейта.
                            self.rec.Reset()
                            _was_gated = False
                        # Wake-word гейт (Tier0 #5): вне окна активации НЕ кормим
                        # тяжёлый Vosk — экономим CPU N100. Стоит ПОСЛЕ эхо-гейта
                        # (Джарвис не будит сам себя). Выключен/нет детектора →
                        # feed() возвращает True (прозрачно/fail-open).
                        if not self._wake_gate.feed(bytes(data), time.monotonic()):
                            continue
                        if self.rec.AcceptWaveform(bytes(data)):
                            # Финальный результат после паузы
                            try:
                                text = json.loads(self.rec.Result()).get("text", "").strip()
                            except Exception:
                                log.exception("vosk result parse failed")
                                continue
                            if not text:
                                continue
                            intent = self._extract_intent(text)
                            if not intent:
                                continue
                            _publish_intent(intent)
                        else:
                            # Partial result — ловим команды без паузы в конце.
                            # Если partial уже содержит wake word + хвост, и хвост
                            # отличается от предыдущего — публикуем превентивно.
                            try:
                                partial_text = json.loads(self.rec.PartialResult()).get("partial", "").strip()
                            except Exception:
                                continue
                            if not partial_text:
                                continue
                            intent = self._extract_intent(partial_text)
                            if not intent:
                                continue
                            # Ждём достаточно слов (≥2 слова в интенте) чтобы не
                            # срабатывать на каждый фонем
                            if len(intent.split()) < 2:
                                continue
                            _publish_intent(intent)
            except Exception:
                log.exception("voice loop crashed — restarting in 3s")
                time.sleep(3.0)
                # Пересоздаём рекогнайзер после краша
                try:
                    if self.model is not None:
                        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
                except Exception:
                    log.exception("vosk recognizer recreate failed")


if __name__ == "__main__":
    JarvisMain().run()
