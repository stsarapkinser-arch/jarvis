"""Тесты wake-word гейта (Tier0 #5) — state-machine «окна активации».

openWakeWord/аудио в песочнице нет — детектор инъектируем фейком. Проверяем:
прозрачность при выключенном гейте, fail-open без детектора, открытие/закрытие
окна по порогу, продление окна, и что build_gate уважает env-тумблер.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.wakeword import WakeGate, WakeWordDetector, build_gate

_FRAME = b"\x00\x00" * 1280  # неважно что — детектор фейковый


# ───────────────────────── прозрачность / fail-open ─────────────────────────
def test_disabled_gate_is_transparent():
    g = WakeGate(None, enabled=False)
    # Выключен → всегда кормим Vosk (поведение слушателя байт-в-байт прежнее).
    assert g.feed(_FRAME, now=0.0) is True
    assert g.feed(_FRAME, now=100.0) is True


def test_enabled_without_detector_fails_open():
    g = WakeGate(None, enabled=True)
    # Включён, но детектора нет → не блокируем вход (глухой ассистент хуже).
    assert g.feed(_FRAME, now=0.0) is True


def test_detector_exception_fails_open():
    def boom(_pcm):
        raise RuntimeError("detector crashed")
    g = WakeGate(boom, enabled=True, window_sec=5.0, threshold=0.5)
    assert g.feed(_FRAME, now=0.0) is True  # сбой детектора → fail-open на кадр


# ───────────────────────── окно активации ─────────────────────────
def test_below_threshold_blocks_when_window_closed():
    g = WakeGate(lambda _p: 0.1, enabled=True, window_sec=5.0, threshold=0.5)
    assert g.feed(_FRAME, now=0.0) is False  # тихо — Vosk не кормим


def test_wake_opens_window_and_expires():
    scores = iter([0.9, 0.1, 0.1, 0.1])
    g = WakeGate(lambda _p: next(scores), enabled=True, window_sec=5.0, threshold=0.5)
    assert g.feed(_FRAME, now=0.0) is True    # сработало → окно открыто
    assert g.feed(_FRAME, now=2.0) is True    # внутри окна
    assert g.feed(_FRAME, now=4.9) is True    # ещё внутри
    assert g.feed(_FRAME, now=5.1) is False   # окно истекло


def test_wake_window_extends_on_repeat():
    scores = iter([0.9, 0.1, 0.9, 0.1])
    g = WakeGate(lambda _p: next(scores), enabled=True, window_sec=5.0, threshold=0.5)
    assert g.feed(_FRAME, now=0.0) is True    # окно до 5.0
    assert g.feed(_FRAME, now=4.0) is True    # внутри
    assert g.feed(_FRAME, now=4.5) is True    # снова сработало → окно до 9.5
    assert g.feed(_FRAME, now=9.0) is True    # внутри продлённого окна


def test_is_listening_reflects_window():
    g = WakeGate(lambda _p: 0.9, enabled=True, window_sec=5.0, threshold=0.5)
    assert g.is_listening(0.0) is False       # окно ещё не открывали
    g.feed(_FRAME, now=0.0)
    assert g.is_listening(3.0) is True
    assert g.is_listening(6.0) is False


# ───────────────────────── build_gate / env ─────────────────────────
def test_build_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("JARVIS_WAKEWORD", raising=False)
    g = build_gate()
    assert g.enabled is False
    assert g.feed(_FRAME, now=0.0) is True


def test_build_gate_on_with_flag_failopen_without_pkg(monkeypatch):
    # Включаем флаг; openwakeword в песочнице нет → детектор недоступен →
    # gate включён, но fail-open (вход не блокирует).
    monkeypatch.setenv("JARVIS_WAKEWORD", "1")
    g = build_gate()
    assert g.enabled is True
    assert g.feed(_FRAME, now=0.0) is True


# ───────────────────────── детектор в standby ─────────────────────────
def test_detector_standby_without_openwakeword():
    # openwakeword не установлен → available False, predict даёт 0.0, не падает.
    det = WakeWordDetector()
    assert det.available is False
    assert det.predict(_FRAME) == 0.0
