"""Smoke tests for the FFT analyzer.

These tests run pure-Python (no Qt) and only depend on the stdlib + numpy.
They cover:

* compute_bands on silence → all-zero
* compute_bands on a synthetic 1 kHz tone → energy concentrated in the
  expected log-band
* smooth_bands is monotone-blending and respects length mismatch
* PiperFFTPump.feed() chunks the buffer into per-frame events without
  publishing past empty input
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.audio.fft_analyzer as audio_fft
from src.common.event_bus import EventBus, EventType
from src.common.singleton import Singleton


def _tone_bytes(freq_hz: float, sample_rate: int = 22050, ms: int = 80) -> bytes:
    n = int(sample_rate * ms / 1000)
    samples = [int(0.5 * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate)) for i in range(n)]
    return struct.pack("<" + "h" * n, *samples)


def test_compute_bands_silence_is_zero() -> None:
    frame = audio_fft.compute_bands(b"", n_bands=12)
    assert frame.level == 0.0
    assert all(b == 0.0 for b in frame.bands)
    assert len(frame.bands) == 12


def test_compute_bands_tone_concentrates_energy() -> None:
    pcm = _tone_bytes(1000.0)
    frame = audio_fft.compute_bands(pcm, n_bands=16)
    assert frame.level > 0.05
    # The strongest band should sit somewhere in the lower-middle of the log
    # range (1 kHz between 80 Hz and 8 kHz).
    peak_idx = max(range(len(frame.bands)), key=lambda i: frame.bands[i])
    assert 2 <= peak_idx <= 12


def test_smooth_bands_blends() -> None:
    out = audio_fft.smooth_bands([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], alpha=0.5)
    assert out == (0.5, 0.5, 0.5)


def test_smooth_bands_length_mismatch_returns_current() -> None:
    out = audio_fft.smooth_bands([0.0], [1.0, 1.0], alpha=0.5)
    assert out == (1.0, 1.0)


def test_pump_feed_publishes_events(monkeypatch) -> None:
    Singleton.reset(EventBus)
    Singleton.reset(audio_fft.PiperFFTPump)
    bus = EventBus()
    captured: list = []

    class _FakeLoop:
        def __init__(self):
            self.threadsafe = []

    monkeypatch.setattr(
        bus, "publish_threadsafe",
        lambda e: captured.append(e),
    )
    pump = audio_fft.PiperFFTPump(bus, sample_rate=22050, frame_ms=20, n_bands=8)
    # ~ enough samples for several frames of 20 ms each.
    pump.feed(_tone_bytes(800.0, ms=200))
    assert captured, "pump must publish at least one AUDIO_FFT event"
    for e in captured:
        assert e.type == EventType.AUDIO_FFT
        assert "bands" in e.data and "level" in e.data
        assert len(e.data["bands"]) == 8


def test_pump_reset_zeroes() -> None:
    Singleton.reset(EventBus)
    Singleton.reset(audio_fft.PiperFFTPump)
    bus = EventBus()
    bands_snapshots: list[list[float]] = []

    def _cap(e) -> None:
        if e.type == EventType.AUDIO_FFT:
            bands_snapshots.append(list(e.data["bands"]))

    bus.publish_threadsafe = _cap  # type: ignore[assignment]
    pump = audio_fft.PiperFFTPump(bus, sample_rate=22050, frame_ms=20, n_bands=8)
    pump.feed(_tone_bytes(500.0, ms=80))
    pump.reset()
    assert bands_snapshots, "expected at least one frame before reset"
    # Last published frame after reset should be all-zero.
    assert all(b == 0.0 for b in bands_snapshots[-1])
