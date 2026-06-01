"""Телеметрия лестницы маршрутизации — какой слой обработал интент.

Лестница распознавания (L0 точный алиас → L1 шаблоны → L1a резолвер приложений →
L1b fuzzy → L2 семантика → 3B) построена, но без счётчиков мы НЕ видим, как
часто срабатывает каждый слой на живой речи. А без этого нельзя понять,
окупается ли L2, куда добавлять алиасы и какая доля запросов реально доходит до
дорогого 3B. Этот модуль замыкает цикл «растить детерминированный слой из
данных».

Дизайн: ``process_intent`` уже возвращает тег-строку результата (``[alias_fastpath]``,
``[agent_done]`` …) — единственная точка, где видно, чем кончился интент.
``RoutingStats`` маппит тег в человекочитаемый УРОВЕНЬ и считает. Чистый счётчик
без зависимостей — тестируется в отрыве; периодический лог печатает распределение
и долю, ушедшую на 3B (ключевая метрика «насколько разгрузили мозг»).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

log = logging.getLogger("jarvis.routing")

# Тег результата process_intent → уровень лестницы (человекочитаемо). Теги, не
# относящиеся к слоям (resolve-гейты pending-состояний, пустой ввод), сводим в
# служебные категории, чтобы не засорять распределение по слоям.
_TAG_TO_LEVEL: dict[str, str] = {
    "[alias_fastpath]": "L0_alias",
    "[macro]": "L0_macro",
    "[pattern_fastpath]": "L1_pattern",     # включает L1a (open_app) — один путь
    "[fuzzy_fastpath]": "L1b_fuzzy",
    "[semantic_fastpath]": "L2_semantic",
    "[agent_done]": "L3_model",             # дошло до 3B — дорогой путь
    "[shadow_resolved]": "resolve_gate",
    "[confirmation_handled]": "resolve_gate",
    "[skill_resolved]": "resolve_gate",
    "[learn_resolved]": "resolve_gate",
}
# Уровни, считающиеся «детерминированными» (минули 3B) — для доли разгрузки.
_DETERMINISTIC = frozenset({
    "L0_alias", "L0_macro", "L1_pattern", "L1b_fuzzy", "L2_semantic",
})
_MODEL_LEVEL = "L3_model"


def level_for_tag(tag: str) -> str:
    """Уровень лестницы для тега process_intent. Неизвестное/пустое → 'other'."""
    return _TAG_TO_LEVEL.get((tag or "").strip(), "other")


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    """Неизменяемый срез счётчиков для лога/тестов."""
    counts: dict[str, int]
    total: int

    @property
    def deterministic(self) -> int:
        return sum(n for lvl, n in self.counts.items() if lvl in _DETERMINISTIC)

    @property
    def model(self) -> int:
        return self.counts.get(_MODEL_LEVEL, 0)

    @property
    def routable(self) -> int:
        """Интенты, относящиеся к лестнице (детерминированные + 3B), без
        resolve-гейтов и служебного — знаменатель доли разгрузки."""
        return self.deterministic + self.model

    @property
    def offload_ratio(self) -> float:
        """Доля маршрутизируемых интентов, решённых БЕЗ 3B (главная метрика)."""
        return self.deterministic / self.routable if self.routable else 0.0


class RoutingStats:
    """Счётчик уровней лестницы. Потокобезопасность не нужна: пишется из одного
    event-loop (process_intent сериализован). Логирует распределение каждые
    ``log_every`` маршрутизируемых интентов."""

    def __init__(self, log_every: int = 25) -> None:
        self._counts: Counter[str] = Counter()
        self._log_every = max(0, log_every)
        self._since_log = 0

    def record(self, tag: str) -> str:
        """Учесть результат интента по его тегу. Возвращает уровень (удобно в
        тестах). Пустой ввод/служебное не двигают счётчик периодического лога."""
        level = level_for_tag(tag)
        if level == "other":
            return level
        self._counts[level] += 1
        if level in _DETERMINISTIC or level == _MODEL_LEVEL:
            self._since_log += 1
            if self._log_every and self._since_log >= self._log_every:
                log.info("routing: %s", self.format_summary())
                self._since_log = 0
        return level

    def snapshot(self) -> RoutingSnapshot:
        return RoutingSnapshot(counts=dict(self._counts), total=sum(self._counts.values()))

    def format_summary(self) -> str:
        """Однострочная сводка: доля разгрузки + разбивка по слоям."""
        snap = self.snapshot()
        if snap.routable == 0:
            return "нет маршрутизируемых интентов"
        parts = [
            f"{lvl}={self._counts[lvl]}"
            for lvl in sorted(self._counts)
            if lvl not in ("resolve_gate", "other")
        ]
        return (
            f"разгрузка 3B {snap.offload_ratio:.0%} "
            f"({snap.deterministic}/{snap.routable} без модели) · " + " ".join(parts)
        )


__all__ = ["RoutingStats", "RoutingSnapshot", "level_for_tag"]
