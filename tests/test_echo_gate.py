"""Тесты эхо-гейта слушателя (анти-само-триггер).

Пока Джарвис говорит, вход микрофона должен быть заглушён, иначе Vosk слышит
собственную речь и срабатывает на неё. Гейт закрывается по STATE_CHANGE=SPEAKING
и продлевается на каждый звуковой AUDIO_FFT-фрейм; открывается по истечении
хвоста (НЕ по IDLE — его шлёт и оркестратор, порядок не гарантирован).

Тестируем чистую логику гейта (обработчики + _gated), не поднимая ни аудио-цикл,
ни реальную модель Vosk: конструируем слушатель с несуществующим путём модели.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.event_bus import Event, EventType
from src.common.singleton import Singleton


@pytest.fixture(autouse=True)
def reset_singletons():
    Singleton.reset()
    yield
    Singleton.reset()


def _listener():
    from src.core.entry_point import JarvisMain
    # Несуществующий путь → model/rec = None, тяжёлый Vosk не грузится.
    return JarvisMain(model_path="/nonexistent/model/path")


def test_not_gated_initially():
    j = _listener()
    assert j._gated() is False


def test_speaking_closes_gate():
    j = _listener()
    asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, ("SPEAKING", "фраза"))))
    assert j._gated() is True


def test_gate_expires_after_tail():
    from src.core.entry_point import ECHO_GATE_TAIL_SEC
    j = _listener()
    asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, ("SPEAKING", "x"))))
    # Симулируем, что последний звук был давно — хвост истёк.
    j._gate_until = time.monotonic() - 0.01
    assert j._gated() is False
    assert ECHO_GATE_TAIL_SEC > 0


def test_audio_fft_with_level_refreshes_gate():
    j = _listener()
    asyncio.run(j._on_audio_fft(Event(EventType.AUDIO_FFT, {"level": 0.5, "bands": []})))
    assert j._gated() is True


def test_audio_fft_zero_level_does_not_refresh():
    j = _listener()
    # нулевой (финальный) фрейм не должен закрывать гейт.
    asyncio.run(j._on_audio_fft(Event(EventType.AUDIO_FFT, {"level": 0.0, "bands": []})))
    assert j._gated() is False


def test_idle_does_not_open_active_gate():
    """IDLE намеренно не трогает гейт: открытие — только по истечении хвоста."""
    j = _listener()
    asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, ("SPEAKING", "x"))))
    asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, ("IDLE", ""))))
    assert j._gated() is True


def test_state_change_plain_string_payload():
    """Обработчик устойчив к payload-строке (а не кортежу)."""
    j = _listener()
    asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, "SPEAKING")))
    assert j._gated() is True


def test_non_speaking_states_ignored():
    j = _listener()
    for st in ("THINKING", "ALERT", "IDLE"):
        asyncio.run(j._on_acoustic_state(Event(EventType.STATE_CHANGE, (st, "x"))))
    assert j._gated() is False
