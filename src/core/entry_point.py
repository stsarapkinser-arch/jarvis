from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import sounddevice as sd
from vosk import KaldiRecognizer, Model

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

    def run(self, bus: EventBus | None = None) -> None:
        bus = bus or EventBus()
        # Режим ожидания: модели нет — извещаем HUD и тихо выходим из треда.
        # Сервис при этом остаётся живой, оператор подкладывает веса и
        # перезапускает unit.
        if self.model is None or self.rec is None:
            log.warning("voice loop standby: %s", VOSK_MISSING_MSG)
            try:
                bus.publish_threadsafe(
                    Event(EventType.STATE_CHANGE, ("ALERT", VOSK_MISSING_MSG))
                )
            except Exception:
                log.exception("could not publish Vosk-missing alert")
            print(f"⚠ {VOSK_MISSING_MSG}")
            return

        log.info("voice listener ready")
        print("🎙 Джарвис слушает...")
        # Дедуп: храним последний ОПУБЛИКОВАННЫЙ интент и его время. Без этого
        # одна фраза стреляла дважды — сначала на partial-результате (мы ловим
        # команды без паузы), затем на final-результате с тем же текстом →
        # Джарвис отвечал на одно и то же два раза.
        _last_published: str = ""
        _last_published_ts: float = 0.0
        import time as _time

        def _publish_intent(intent: str) -> None:
            nonlocal _last_published, _last_published_ts
            now = _time.monotonic()
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
                    while True:
                        data, _ = stream.read(READ_FRAMES)
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
                import time as _time
                _time.sleep(3.0)
                # Пересоздаём рекогнайзер после краша
                try:
                    if self.model is not None:
                        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
                except Exception:
                    log.exception("vosk recognizer recreate failed")


if __name__ == "__main__":
    JarvisMain().run()
