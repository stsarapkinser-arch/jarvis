"""Регресс-тест качества роутера по golden-набору (фундамент измеримости).

Порог намеренно ниже текущих 100% — набор будет расти, единичные сложные
перефразировки допустимы; но обвал точности или полноты категории поймаем."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.router import IntentCategory
from tests.router_eval import evaluate, format_report

ACCURACY_FLOOR = 0.92
RECALL_FLOOR = 0.85


def test_router_overall_accuracy():
    report = evaluate()
    assert report.accuracy >= ACCURACY_FLOOR, "\n" + format_report(report)


def test_router_per_category_recall():
    report = evaluate()
    for cat in IntentCategory:
        if cat in report.per_category_total:
            assert report.recall(cat) >= RECALL_FLOOR, (
                f"{cat.value} recall просел\n" + format_report(report)
            )


def test_report_formats_cleanly():
    # format_report не должен падать ни на полном, ни на пустом наборе.
    assert "accuracy" in format_report(evaluate())
