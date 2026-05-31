#!/usr/bin/env python3
"""Калибровка порога семантического матчера (L2) на golden-наборе.

Запуск из корня репо (нужен ЖИВОЙ embed-сервер :8090 — он считает векторы):
    python scripts/calibrate_semantic.py

Зачем: порог косинуса семантически зависит от embedding-МОДЕЛИ. Этот скрипт
гоняет ``tests/semantic_dataset`` через реальный эмбеддер, свипает пороги и
печатает таблицу: сколько парафразов поймано верно, сколько ушло мимо, сколько
ЧУЖИХ матчей и ложных приёмов на негативах. Рекомендует МАКСИМАЛЬНЫЙ recall при
НУЛЕ опасных ошибок (wrong=0, false_accept=0) — его и ставьте в
``JARVIS_SEMANTIC_THRESHOLD`` перед включением ``JARVIS_SEMANTIC_MATCH=1``.

Прогоняйте после смены embedding-модели или заметного роста каталога навыков.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.inference.embeddings import EmbeddingClient, EmbeddingServerError  # noqa: E402
from src.inference.semantic import DEFAULT_MARGIN, evaluate  # noqa: E402
from tests.semantic_dataset import NEGATIVES, POSITIVES  # noqa: E402

# Свип порогов: грубая сетка по диапазону косинуса коротких фраз.
_THRESHOLDS = [round(0.30 + 0.02 * i, 2) for i in range(31)]  # 0.30 … 0.90


def main() -> int:
    client = EmbeddingClient()
    if not client.health():
        print("✗ embed-сервер :8090 недоступен. Подними: ./setup_embed_server.sh", file=sys.stderr)
        return 1

    try:
        results = evaluate(
            client.embed, POSITIVES, NEGATIVES,
            thresholds=_THRESHOLDS, margin=DEFAULT_MARGIN,
        )
    except EmbeddingServerError as exc:
        print(f"✗ ошибка эмбеддинга: {exc}", file=sys.stderr)
        return 1

    n_pos, n_neg = len(POSITIVES), len(NEGATIVES)
    print(f"Калибровка L2 — позитивов {n_pos}, негативов {n_neg}, margin {DEFAULT_MARGIN}\n")
    print(f"{'порог':>6} {'верно':>6} {'мимо':>5} {'ЧУЖОЙ':>6} {'ложн.приём':>11} {'recall':>7}  безопасен")
    print("-" * 60)
    best_clean = None
    for r in results:
        flag = "✓" if r.clean else " "
        if r.clean and (best_clean is None or r.recall > best_clean.recall):
            best_clean = r
        print(f"{r.threshold:>6.2f} {r.correct:>6} {r.missed:>5} {r.wrong:>6} "
              f"{r.false_accept:>11} {r.recall:>6.0%}  {flag}")

    print("-" * 60)
    if best_clean is not None:
        print(f"\n→ Рекомендую JARVIS_SEMANTIC_THRESHOLD={best_clean.threshold:.2f} "
              f"(recall {best_clean.recall:.0%}, ноль опасных ошибок).")
        print("  Затем включить: JARVIS_SEMANTIC_MATCH=1")
    else:
        print("\n⚠ Безопасного порога не нашлось (везде есть чужой матч/ложный приём).")
        print("  Расширь экземпляры алиасов или подними margin; модель может плохо")
        print("  разделять близкие навыки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
