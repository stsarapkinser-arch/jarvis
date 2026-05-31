"""Harness оценки роутера — считает точность и полноту по golden-набору.

Используется и из теста-регрессии (``test_router_eval.py``), и из CLI
(``scripts/eval_router.py``) для человекочитаемого отчёта. Чистый модуль без
side-effect'ов: на вход — роутер и набор, на выход — структура метрик."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.inference.router import IntentCategory, IntentRouter
from tests.eval_dataset import DATASET


@dataclass(frozen=True, slots=True)
class Miss:
    phrase: str
    expected: IntentCategory
    got: IntentCategory
    backend: str


@dataclass(slots=True)
class EvalReport:
    total: int
    correct: int
    misses: list[Miss] = field(default_factory=list)
    per_category_total: dict[IntentCategory, int] = field(default_factory=dict)
    per_category_correct: dict[IntentCategory, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def recall(self, category: IntentCategory) -> float:
        tot = self.per_category_total.get(category, 0)
        return self.per_category_correct.get(category, 0) / tot if tot else 1.0


def evaluate(
    router: IntentRouter | None = None,
    dataset: tuple[tuple[str, IntentCategory], ...] = DATASET,
) -> EvalReport:
    r = router or IntentRouter()
    correct = 0
    misses: list[Miss] = []
    cat_total: dict[IntentCategory, int] = defaultdict(int)
    cat_correct: dict[IntentCategory, int] = defaultdict(int)
    for phrase, expected in dataset:
        cat_total[expected] += 1
        decision = r.route(phrase)
        if decision.category == expected:
            correct += 1
            cat_correct[expected] += 1
        else:
            misses.append(Miss(phrase, expected, decision.category, decision.backend))
    return EvalReport(
        total=len(dataset),
        correct=correct,
        misses=misses,
        per_category_total=dict(cat_total),
        per_category_correct=dict(cat_correct),
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"Router eval: accuracy={report.accuracy:.1%} "
        f"({report.correct}/{report.total})",
        "",
        "Recall по категориям:",
    ]
    for cat in IntentCategory:
        if cat in report.per_category_total:
            lines.append(
                f"  {cat.value:<16} {report.recall(cat):.1%} "
                f"({report.per_category_correct.get(cat, 0)}/{report.per_category_total[cat]})"
            )
    if report.misses:
        lines.append("")
        lines.append(f"Ошибки маршрутизации ({len(report.misses)}):")
        for m in report.misses:
            lines.append(
                f"  «{m.phrase}» → {m.got.value} (ожидалось {m.expected.value}, backend={m.backend})"
            )
    return "\n".join(lines)


__all__ = ["Miss", "EvalReport", "evaluate", "format_report"]
