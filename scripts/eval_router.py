#!/usr/bin/env python3
"""CLI-отчёт по качеству роутера: точность, recall по категориям, список ошибок.

    python scripts/eval_router.py            # regex-бэкенд (дефолт)

Это инструмент измерения из плана level-up: гоняй на каждое изменение лексикона
роутера или каталога навыков, чтобы видеть, стало лучше или хуже, а не угадывать.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.router_eval import evaluate, format_report  # noqa: E402


def main() -> int:
    report = evaluate()
    print(format_report(report))
    # Ненулевой код выхода, если точность просела ниже порога — удобно для CI.
    return 0 if report.accuracy >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
