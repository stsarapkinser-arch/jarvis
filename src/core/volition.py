"""ProactiveGate — воля с тактом.

Демоны и проактивный цикл наблюдают постоянно; без фильтра Джарвис превратился
бы в назойливого соседа. Этот гейт решает, ВПРАВЕ ли он сейчас заговорить по
своей инициативе, учитывая:
  * тихие часы (ночью молчим, кроме критичного — это решает вызывающий);
  * минимальный интервал между инициативами (анти-спам);
  * недавнюю речь оператора (не перебиваем «по своей воле» сразу после общения);
  * повтор темы (не бубним одно и то же).

Гейт чистый и синхронный — всё время передаётся аргументами, поэтому тривиально
тестируется без часов и event loop. Критичные тревоги (перегрев, вторжение)
через гейт НЕ проходят — они идут напрямую, у безопасности приоритет над тактом.
"""
from __future__ import annotations

import time
from collections import deque


class ProactiveGate:
    """Решает, можно ли сейчас высказать проактивную (не-критичную) реплику."""

    def __init__(
        self,
        *,
        min_interval_sec: float = 1200.0,
        silence_before_sec: float = 1200.0,
        quiet_start_hour: int = 23,
        quiet_end_hour: int = 7,
        dedup_window: int = 8,
    ) -> None:
        self.min_interval_sec = min_interval_sec
        self.silence_before_sec = silence_before_sec
        self.quiet_start_hour = quiet_start_hour
        self.quiet_end_hour = quiet_end_hour
        self._recent: deque[str] = deque(maxlen=dedup_window)

    def _in_quiet_hours(self, hour: int) -> bool:
        s, e = self.quiet_start_hour, self.quiet_end_hour
        if s == e:
            return False
        if s < e:                       # напр. 1..6
            return s <= hour < e
        return hour >= s or hour < e    # перехлёст через полночь, напр. 23..7

    @staticmethod
    def _topic_key(text: str) -> str:
        return " ".join((text or "").lower().split())[:60]

    def is_repeat(self, text: str) -> bool:
        return self._topic_key(text) in self._recent

    def should_speak(
        self,
        *,
        now: float | None = None,
        last_say_ts: float,
        last_proactive_ts: float,
        load_busy: bool,
        hour: int,
    ) -> bool:
        """True — инициатива уместна прямо сейчас.

        ``last_say_ts`` — время любой последней речи Джарвиса (обновляется в
        say()); ``last_proactive_ts`` — время последней ПРОАКТИВНОЙ реплики."""
        now = time.time() if now is None else now
        if load_busy:
            return False
        if self._in_quiet_hours(hour):
            return False
        if now - last_say_ts < self.silence_before_sec:
            return False
        if now - last_proactive_ts < self.min_interval_sec:
            return False
        return True

    def record(self, text: str) -> None:
        """Запомнить произнесённую тему для подавления повторов."""
        self._recent.append(self._topic_key(text))


__all__ = ["ProactiveGate"]
