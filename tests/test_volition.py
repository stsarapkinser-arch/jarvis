"""Тесты ProactiveGate — такта проактивной речи."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.volition import ProactiveGate


def _gate(**kw) -> ProactiveGate:
    base = dict(min_interval_sec=1200, silence_before_sec=1200,
                quiet_start_hour=23, quiet_end_hour=7)
    base.update(kw)
    return ProactiveGate(**base)


def test_allows_when_quiet_idle_and_intervals_elapsed():
    g = _gate()
    assert g.should_speak(
        now=10_000, last_say_ts=0, last_proactive_ts=0,
        load_busy=False, hour=14,
    ) is True


def test_blocks_when_busy():
    g = _gate()
    assert g.should_speak(
        now=10_000, last_say_ts=0, last_proactive_ts=0,
        load_busy=True, hour=14,
    ) is False


def test_blocks_during_quiet_hours():
    g = _gate()
    # 02:00 — внутри тихих часов 23..7.
    assert g.should_speak(
        now=10_000, last_say_ts=0, last_proactive_ts=0,
        load_busy=False, hour=2,
    ) is False
    # 8:00 — уже вне.
    assert g.should_speak(
        now=10_000, last_say_ts=0, last_proactive_ts=0,
        load_busy=False, hour=8,
    ) is True


def test_blocks_right_after_recent_speech():
    g = _gate()
    assert g.should_speak(
        now=10_000, last_say_ts=9_500, last_proactive_ts=0,
        load_busy=False, hour=14,
    ) is False


def test_blocks_before_min_interval_between_initiatives():
    g = _gate()
    assert g.should_speak(
        now=10_000, last_say_ts=0, last_proactive_ts=9_500,
        load_busy=False, hour=14,
    ) is False


def test_quiet_hours_wraparound_and_non_wrap():
    wrap = _gate(quiet_start_hour=23, quiet_end_hour=7)
    assert wrap._in_quiet_hours(23) and wrap._in_quiet_hours(3) and not wrap._in_quiet_hours(12)
    non = _gate(quiet_start_hour=1, quiet_end_hour=6)
    assert non._in_quiet_hours(3) and not non._in_quiet_hours(0) and not non._in_quiet_hours(8)


def test_repeat_suppression():
    g = _gate()
    assert g.is_repeat("Сэр, диск почти полон") is False
    g.record("Сэр, диск почти полон")
    assert g.is_repeat("сэр,  диск   почти полон") is True  # нормализация регистра/пробелов
