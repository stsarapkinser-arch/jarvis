"""Тесты телеметрии лестницы маршрутизации.

Чистый счётчик: маппинг тегов в уровни, доля разгрузки 3B, периодический лог,
и что resolve-гейты/служебное не искажают распределение по слоям.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.routing_stats import RoutingStats, level_for_tag


# ───────────────────────── маппинг тегов ─────────────────────────
def test_tag_to_level_mapping():
    assert level_for_tag("[alias_fastpath]") == "L0_alias"
    assert level_for_tag("[macro]") == "L0_macro"
    assert level_for_tag("[pattern_fastpath]") == "L1_pattern"
    assert level_for_tag("[fuzzy_fastpath]") == "L1b_fuzzy"
    assert level_for_tag("[semantic_fastpath]") == "L2_semantic"
    assert level_for_tag("[agent_done]") == "L3_model"


def test_unknown_and_empty_tag_is_other():
    assert level_for_tag("") == "other"
    assert level_for_tag("[whatever]") == "other"
    assert level_for_tag("[Pixel: none]") == "other"


# ───────────────────────── подсчёт и доля разгрузки ─────────────────────────
def test_offload_ratio_basic():
    s = RoutingStats(log_every=0)
    for tag in ["[alias_fastpath]"] * 7 + ["[pattern_fastpath]"] * 2 + ["[agent_done]"]:
        s.record(tag)
    snap = s.snapshot()
    assert snap.deterministic == 9
    assert snap.model == 1
    assert snap.routable == 10
    assert abs(snap.offload_ratio - 0.9) < 1e-9


def test_resolve_gates_excluded_from_routable():
    s = RoutingStats(log_every=0)
    s.record("[shadow_resolved]")
    s.record("[confirmation_handled]")
    s.record("[alias_fastpath]")
    snap = s.snapshot()
    # resolve-гейты не входят в знаменатель доли разгрузки.
    assert snap.routable == 1
    assert snap.offload_ratio == 1.0


def test_other_tag_does_not_count():
    s = RoutingStats(log_every=0)
    assert s.record("") == "other"
    assert s.record("[Memory: none]") == "other"
    assert s.snapshot().routable == 0


def test_record_returns_level():
    s = RoutingStats(log_every=0)
    assert s.record("[semantic_fastpath]") == "L2_semantic"


# ───────────────────────── периодический лог ─────────────────────────
def test_periodic_log_fires(caplog):
    import logging
    s = RoutingStats(log_every=3)
    with caplog.at_level(logging.INFO, logger="jarvis.routing"):
        s.record("[alias_fastpath]")
        s.record("[alias_fastpath]")
        assert "routing:" not in caplog.text  # ещё рано
        s.record("[agent_done]")               # 3-й маршрутизируемый → лог
    assert "routing:" in caplog.text
    assert "разгрузка" in caplog.text


def test_summary_no_routable():
    s = RoutingStats(log_every=0)
    assert "нет маршрутизируемых" in s.format_summary()


def test_summary_lists_levels():
    s = RoutingStats(log_every=0)
    s.record("[alias_fastpath]")
    s.record("[agent_done]")
    out = s.format_summary()
    assert "L0_alias=1" in out and "L3_model=1" in out
    assert "50%" in out
