"""Wake-word гейт перед Vosk (Tier0 #5) — меньше постоянной нагрузки N100.

Сейчас слушатель гоняет ПОЛНЫЙ ASR (Vosk AcceptWaveform) на каждом кадре и ищет
wake-слово строкой в тексте — то есть тяжёлый распознаватель крутится всегда.
Лёгкий wake-word детектор (openWakeWord, единицы МБ, CPU) как ПРЕД-гейт даёт
кормить Vosk только в «окне активации» после ключевого слова: меньше нагрев и
меньше ложных срабатываний.

Два слоя (как везде в проекте — рискованное за флагом, с мягкой деградацией):

  * ``WakeWordDetector`` — soft-import обёртка openWakeWord. Нет пакета/модели →
    ``available=False`` (как vosk/sounddevice). Никогда не валит импорт.
  * ``WakeGate`` — ЧИСТАЯ state-machine «окна активации», тестируемая без аудио
    и без самого openWakeWord (детектор инъектируется). Решает лишь, кормить ли
    Vosk этим кадром; аудиопоток и эхо-гейт не трогает.

Семантика деградации — FAIL-OPEN: если гейт включён, но детектор недоступен,
НЕ блокируем вход (ассистент продолжает слышать прежним путём) — потеря экономии
CPU лучше глухого ассистента. Выключенный гейт (по умолчанию) прозрачен:
``should_listen`` всегда True, поведение слушателя байт-в-байт прежнее.

ВАЖНО (калибровка): порог детектора и длина окна зависят от модели wake-word и
акустики; дефолты — консервативный старт. Проверять на железе перед опорой.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger("jarvis.wakeword")

# Детектор: PCM-кадр (int16 bytes, 16 кГц) → score [0..1]. Инъектируется в
# WakeGate; в проде это WakeWordDetector.predict, в тестах — фейк.
DetectFn = Callable[[bytes], float]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


# Окно прослушивания после срабатывания wake-слова: в течение него кадры идут в
# Vosk (захватывает команду-хвост и продлевается, пока оператор говорит).
DEFAULT_WINDOW_SEC = _env_float("JARVIS_WAKEWORD_WINDOW", 8.0)
DEFAULT_THRESHOLD = _env_float("JARVIS_WAKEWORD_THRESHOLD", 0.5)


class WakeWordDetector:
    """openWakeWord, загруженный мягко. ``available`` False без пакета/модели.

    Кадр — int16 PCM 16 кГц (тот же формат, что у Vosk-слушателя), поэтому
    разветвление потока не требует ресемплинга."""

    def __init__(
        self,
        model_paths: Optional[list[str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self._model: Any | None = None
        self.available = False
        try:
            import numpy as np  # noqa: F401  — нужен для буфера int16
            import openwakeword.model  # noqa: F401 — проба наличия пакета
        except Exception:
            log.info("openwakeword недоступен — wake-word детектор в standby")
            return
        try:
            self._model = self._build_model(model_paths)
            self._np = np
            self.available = True
            log.info("wake-word детектор готов (порог %.2f)", threshold)
        except Exception:
            log.exception("openwakeword init не удался — детектор в standby")
            self._model = None

    @staticmethod
    def _build_model(model_paths: Optional[list[str]]) -> Any:
        """Сконструировать openWakeWord.Model переносимо между версиями API.

        Имя аргумента со списком моделей менялось (``wakeword_models`` /
        ``wakeword_model_paths``), а лишний kwarg в новых версиях пробрасывается
        в ``AudioFeatures`` и валит init (``TypeError: ... unexpected keyword
        argument 'wakeword_models'``). Поэтому: пустой список = «встроенные
        предобученные» → зовём конструктор БЕЗ аргумента; явные пути — пробуем
        известные имена kwarg, затем позиционно."""
        from openwakeword.model import Model  # type: ignore

        paths = model_paths or []
        if not paths:
            return Model()  # дефолтные модели (могут требовать download_models())
        for kw in ("wakeword_models", "wakeword_model_paths"):
            try:
                return Model(**{kw: paths})
            except TypeError:
                continue
        return Model(paths)  # последняя попытка — позиционно

    def predict(self, pcm: bytes) -> float:
        """Максимальный score по всем wake-моделям для кадра. 0.0 при сбое."""
        if not self.available or self._model is None:
            return 0.0
        try:
            samples = self._np.frombuffer(pcm, dtype=self._np.int16)
            scores = self._model.predict(samples)
            return float(max(scores.values())) if scores else 0.0
        except Exception:
            log.debug("wake predict failed", exc_info=True)
            return 0.0


class WakeGate:
    """State-machine «окна активации»: кормить ли Vosk этим кадром.

    Чистая логика (детектор инъектируется) — тестируется без аудио. ``enabled``
    False → прозрачна (всегда True). Включена, но детектор не дан/недоступен →
    fail-open (всегда True, с однократным предупреждением)."""

    def __init__(
        self,
        detect_fn: DetectFn | None = None,
        *,
        enabled: bool = False,
        window_sec: float = DEFAULT_WINDOW_SEC,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.enabled = enabled
        self.window_sec = window_sec
        self.threshold = threshold
        self._detect = detect_fn
        self._open_until = 0.0
        self._warned_failopen = False

    @property
    def _fail_open(self) -> bool:
        return self.enabled and self._detect is None

    def feed(self, pcm: bytes, now: float) -> bool:
        """Обработать кадр. True → кадр отдавать в Vosk; False → пропустить.

        Прозрачна при выключенном гейте и fail-open при отсутствии детектора.
        Иначе: срабатывание wake-слова открывает окно на ``window_sec``; пока
        окно открыто — True (и его можно продлевать новым срабатыванием)."""
        if not self.enabled:
            return True
        if self._detect is None:
            if not self._warned_failopen:
                self._warned_failopen = True
                log.warning("wake-word включён, но детектор недоступен — fail-open "
                            "(вход не блокируем; экономии CPU не будет)")
            return True
        try:
            score = self._detect(pcm)
        except Exception:
            log.debug("wake detect raised — fail-open на кадр", exc_info=True)
            return True
        if score >= self.threshold:
            self._open_until = now + self.window_sec
        return now < self._open_until

    def is_listening(self, now: float) -> bool:
        """Открыто ли окно сейчас (для диагностики/HUD), без обработки кадра."""
        if not self.enabled or self._detect is None:
            return True
        return now < self._open_until


def build_gate(detector: WakeWordDetector | None = None) -> WakeGate:
    """Собрать WakeGate из env. JARVIS_WAKEWORD=1 включает; детектор создаётся
    лениво (soft-import). Включён без рабочего детектора → fail-open."""
    enabled = os.getenv("JARVIS_WAKEWORD", "0") == "1"
    if not enabled:
        return WakeGate(None, enabled=False)
    det = detector if detector is not None else WakeWordDetector(threshold=DEFAULT_THRESHOLD)
    detect_fn = det.predict if det.available else None
    return WakeGate(
        detect_fn, enabled=True,
        window_sec=DEFAULT_WINDOW_SEC, threshold=DEFAULT_THRESHOLD,
    )


__all__ = [
    "WakeWordDetector",
    "WakeGate",
    "build_gate",
    "DetectFn",
    "DEFAULT_WINDOW_SEC",
    "DEFAULT_THRESHOLD",
]
