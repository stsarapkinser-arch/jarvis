"""Aegis FFT analyzer.

Reads raw S16_LE 22050 Hz PCM coming out of Piper, splits it into bands, and
publishes ``AUDIO_FFT`` events on the bus so the HUD core sphere can pulse in
sync with the voice. The pump is intentionally pure-Python + numpy so it stays
predictable on an Intel N100 (no GStreamer / PulseAudio dependency).

Usage:

    pump = PiperFFTPump(bus, sample_rate=22050)
    piper_stdout = piper_process.stdout   # binary pipe
    pump.start_pump(piper_stdout, aplay_stdin)

The pump tee's the PCM into ``aplay_stdin`` (so audio still plays) and into
its own FFT pipeline that emits one event per ``frame_ms`` window. When numpy
is missing the analyzer degrades gracefully into RMS-only bars (HUD still
animates, just without per-band detail).
"""
from __future__ import annotations

import logging
import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import IO, Sequence

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.audio_fft")

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

# Piper's --output_raw spits S16_LE @ 22050 Hz by default.
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FRAME_MS = 40
DEFAULT_BANDS = 24


@dataclass(frozen=True)
class FFTFrame:
    """One spectral snapshot.

    bands: normalized magnitudes 0..1 (low → high frequency).
    level: RMS amplitude 0..1 for fast pulse calculation on the HUD.
    ts:    monotonic timestamp from time.time().
    """

    bands: tuple[float, ...]
    level: float
    ts: float

    def as_dict(self) -> dict:
        return {"bands": list(self.bands), "level": self.level, "ts": self.ts}


def _bytes_to_samples_numpy(buf: bytes) -> "np.ndarray":
    """S16_LE little-endian → float32 in [-1, 1]."""
    assert _HAS_NUMPY
    # numpy.frombuffer with int16 then scale; tolerate odd byte counts.
    if len(buf) % 2:
        buf = buf[:-1]
    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def _bytes_to_samples_pure(buf: bytes) -> list[float]:
    """Pure-Python fallback. Slow but never crashes."""
    if len(buf) % 2:
        buf = buf[:-1]
    n = len(buf) // 2
    if not n:
        return []
    unpacked = struct.unpack("<" + "h" * n, buf)
    return [s / 32768.0 for s in unpacked]


def compute_bands(
    pcm_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_bands: int = DEFAULT_BANDS,
) -> FFTFrame:
    """Compute one FFT frame from a PCM buffer.

    Always returns a frame (even on empty input) so the HUD never gets stuck
    on a stale pulse. Frequency layout is log-spaced 80 Hz → 8 kHz which
    matches the perceptual range of speech without giving low frequencies
    disproportionate weight."""
    ts = time.time()
    if not pcm_bytes:
        return FFTFrame(tuple([0.0] * n_bands), 0.0, ts)

    if _HAS_NUMPY:
        samples = _bytes_to_samples_numpy(pcm_bytes)
        if samples.size == 0:
            return FFTFrame(tuple([0.0] * n_bands), 0.0, ts)
        # RMS in 0..1
        rms = float(np.sqrt(np.mean(samples * samples)))
        # Hann window + rFFT
        window = np.hanning(samples.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(samples * window))
        if spectrum.size == 0:
            return FFTFrame(tuple([0.0] * n_bands), min(1.0, rms * 4.0), ts)
        # Build log-spaced band edges between 80 Hz and 8 kHz.
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
        f_lo, f_hi = 80.0, min(8000.0, sample_rate / 2.0 - 100.0)
        edges = np.geomspace(f_lo, f_hi, n_bands + 1)
        bands: list[float] = []
        for i in range(n_bands):
            lo, hi = edges[i], edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            chunk = spectrum[mask]
            if chunk.size:
                bands.append(float(chunk.mean()))
            else:
                bands.append(0.0)
        # Normalize bands to 0..1 with mild gamma so quiet whispers still glow.
        peak = max(bands) or 1.0
        scaled = [min(1.0, math.sqrt(b / peak)) for b in bands]
        return FFTFrame(tuple(scaled), min(1.0, rms * 4.0), ts)

    # --- Pure-Python fallback: RMS only ---
    samples = _bytes_to_samples_pure(pcm_bytes)
    if not samples:
        return FFTFrame(tuple([0.0] * n_bands), 0.0, ts)
    sq = sum(s * s for s in samples) / len(samples)
    rms = math.sqrt(sq)
    level = min(1.0, rms * 4.0)
    # Spread RMS into a soft hump in the middle bands to keep the HUD lively.
    centre = n_bands // 2
    bands = [
        level * math.exp(-((i - centre) ** 2) / (2.0 * (n_bands / 6.0) ** 2))
        for i in range(n_bands)
    ]
    return FFTFrame(tuple(bands), level, ts)


def smooth_bands(prev: Sequence[float], curr: Sequence[float], alpha: float = 0.55) -> tuple[float, ...]:
    """Exponential smoothing so the HUD doesn't strobe.

    alpha is the weight of the new frame; with 0.55 the HUD lags ~2 frames
    behind which feels organic for ~25 fps."""
    if len(prev) != len(curr):
        return tuple(curr)
    return tuple(prev[i] * (1.0 - alpha) + curr[i] * alpha for i in range(len(curr)))


class PiperFFTPump(metaclass=Singleton):
    """Owns the binary tee between Piper and aplay.

    Each ``feed(chunk)`` call grows an internal buffer; once it crosses the
    per-frame size we publish a single ``AUDIO_FFT`` event and reset. The
    pump is reentrant — multiple Piper invocations can share it; smoothing
    state survives across utterances so back-to-back acknowledgements blend.
    """

    DEFAULT_PUMP_CHUNK = 1024

    def __init__(
        self,
        bus: EventBus | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        n_bands: int = DEFAULT_BANDS,
    ) -> None:
        self.bus = bus or EventBus()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.frame_bytes = max(2, int(sample_rate * frame_ms / 1000) * 2)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._prev_bands: tuple[float, ...] = tuple([0.0] * n_bands)
        self._last_publish_ts: float = 0.0

    # ---- public API ----
    def feed(self, chunk: bytes) -> None:
        """Push PCM into the analyzer. Safe to call from any thread."""
        if not chunk:
            return
        with self._lock:
            self._buf.extend(chunk)
            while len(self._buf) >= self.frame_bytes:
                frame_bytes = bytes(self._buf[: self.frame_bytes])
                del self._buf[: self.frame_bytes]
                frame = compute_bands(frame_bytes, self.sample_rate, self.n_bands)
                self._prev_bands = smooth_bands(self._prev_bands, frame.bands)
                publishable = FFTFrame(self._prev_bands, frame.level, frame.ts)
                self._publish(publishable)

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._prev_bands = tuple([0.0] * self.n_bands)
        # Final "silence" frame so the HUD glides back to idle. Bypass the
        # rate limiter so the silence frame is guaranteed to land — without
        # this the HUD can get stuck on the last loud frame after a short
        # utterance.
        self._last_publish_ts = 0.0
        self._publish(FFTFrame(self._prev_bands, 0.0, time.time()))

    def start_pump(
        self,
        source: IO[bytes],
        sink: IO[bytes] | None,
        chunk_size: int | None = None,
        label: str = "piper",
    ) -> threading.Thread:
        """Spawn a daemon thread that tee's `source` into `sink` and into the
        FFT analyzer. Closes `sink` when source is exhausted."""
        chunk_size = chunk_size or self.DEFAULT_PUMP_CHUNK

        def _runner() -> None:
            try:
                while True:
                    data = source.read(chunk_size)
                    if not data:
                        break
                    if sink is not None:
                        try:
                            sink.write(data)
                        except (BrokenPipeError, OSError):
                            break
                    self.feed(data)
            except Exception:
                log.exception("FFT pump %s crashed", label)
            finally:
                if sink is not None:
                    try:
                        sink.close()
                    except Exception:
                        pass
                self.reset()

        t = threading.Thread(target=_runner, daemon=True, name=f"fft-pump-{label}")
        t.start()
        return t

    # ---- internals ----
    def _publish(self, frame: FFTFrame) -> None:
        # Cap publish rate at ~50 Hz; the HUD repaints at 60 fps but doesn't
        # need every frame, and the event bus is much cheaper if we don't spam.
        if frame.ts - self._last_publish_ts < 0.018:
            return
        self._last_publish_ts = frame.ts
        try:
            self.bus.publish_threadsafe(Event(EventType.AUDIO_FFT, frame.as_dict()))
        except Exception:
            log.debug("AUDIO_FFT publish failed", exc_info=True)
