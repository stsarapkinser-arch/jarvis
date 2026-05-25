from __future__ import annotations

import json
import logging
from typing import Final

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.voice")

WAKE_WORDS: Final[tuple[str, ...]] = ("джарвис", "jarvis", "компьютер")
SAMPLE_RATE: Final = 16000
BLOCK_SIZE: Final = 16000
READ_FRAMES: Final = 8000


class JarvisMain(metaclass=Singleton):
    """Vosk audio listener. Runs sync in a thread, publishes VOICE_INTENT events."""

    def __init__(self, model_path: str = "model") -> None:
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.wake_words = WAKE_WORDS

    def _extract_intent(self, text: str) -> str | None:
        for ww in self.wake_words:
            if ww in text:
                return text.replace(ww, "").strip()
        return None

    def run(self, bus: EventBus | None = None) -> None:
        bus = bus or EventBus()
        log.info("voice listener ready")
        print("🎙 Джарвис слушает...")
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
            ) as stream:
                while True:
                    data, _ = stream.read(READ_FRAMES)
                    if not self.rec.AcceptWaveform(bytes(data)):
                        continue
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
                    bus.publish_threadsafe(Event(EventType.VOICE_INTENT, intent))
        except Exception:
            log.exception("voice loop crashed")


if __name__ == "__main__":
    JarvisMain().run()
