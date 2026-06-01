"""Тесты адаптивной громкости (AGC) — чистая логика, без аудио-железа.

Покрываем то, чего требует ТЗ оператора:
  * СТАБИЛЬНОСТЬ: разные по уровню фразы сходятся к одной целевой громкости,
    причём gain меняется плавно (слю-лимит), без скачка между фразами.
  * АДАПТИВНОСТЬ: рост внешнего шума поднимает громкость (и прямо в процессе —
    мид-фраза), падение/тишина — опускает; всё в пределах потолков.
  * Оценка шума: молча — доверяем микрофону; во время речи вычитаем свой голос
    (де-эхо) и пускаем оценку только вверх.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.adaptive_gain import (
    AGCConfig,
    AdaptiveGainController,
    AmbientNoiseEstimator,
    AmbientNoiseHub,
    NoiseConfig,
    amp_to_dbfs,
    apply_gain_pcm16,
    dbfs_to_amp,
    power_subtract_db,
    rms_dbfs_pcm16,
)
from src.common.singleton import Singleton


def _const_pcm(value: int, n: int = 1024) -> bytes:
    return struct.pack("<" + "h" * n, *([value] * n))


# ───────────────────────── dBFS / PCM утилиты ─────────────────────────
def test_amp_dbfs_roundtrip_and_floor():
    assert abs(amp_to_dbfs(1.0)) < 1e-6
    assert abs(amp_to_dbfs(0.5) + 6.0206) < 1e-3
    assert amp_to_dbfs(0.0) == -120.0
    assert abs(dbfs_to_amp(0.0) - 1.0) < 1e-9
    assert abs(dbfs_to_amp(-6.0206) - 0.5) < 1e-3


def test_rms_dbfs_pcm16_levels():
    assert rms_dbfs_pcm16(b"") == -120.0
    assert rms_dbfs_pcm16(_const_pcm(0)) == -120.0
    # Полная шкала (≈32767) → ~0 dBFS.
    assert abs(rms_dbfs_pcm16(_const_pcm(32767))) < 0.01
    # Полшкалы → ~-6 dBFS.
    assert abs(rms_dbfs_pcm16(_const_pcm(16384)) + 6.02) < 0.1


def test_power_subtract_db():
    # Равные мощности → остаток на полу.
    assert power_subtract_db(-20.0, -20.0) == -120.0
    # Вычитаем сильно меньшую → почти без изменений.
    assert abs(power_subtract_db(-20.0, -50.0) - (-20.0)) < 0.1
    # Никогда не паникует и не уходит выше total.
    assert power_subtract_db(-30.0, -10.0) == -120.0


def test_apply_gain_pcm16_passthrough_and_clip():
    pcm = _const_pcm(10000, 4)
    assert apply_gain_pcm16(pcm, 1.0) == pcm  # gain=1 → без изменений
    doubled = struct.unpack("<4h", apply_gain_pcm16(pcm, 2.0))
    assert all(v == 20000 for v in doubled)
    # Клиппинг в int16.
    clipped = struct.unpack("<4h", apply_gain_pcm16(_const_pcm(20000, 4), 2.0))
    assert all(v == 32767 for v in clipped)


# ───────────────────────── контроллер: стабильность ─────────────────────────
def _drive(ctrl: AdaptiveGainController, in_db: float, seconds: float, dt: float = 0.02) -> float:
    """Прогнать постоянный уровень входа и вернуть финальный выходной dBFS."""
    steps = int(seconds / dt)
    for _ in range(steps):
        ctrl.update_gain(in_db, dt)
    return ctrl.output_dbfs


def test_converges_to_target_loudness():
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    out = _drive(ctrl, -30.0, seconds=4.0)
    assert abs(out - cfg.target_rms_dbfs) < 1.0


def test_phrase_to_phrase_consistency():
    """Две разные по уровню «фразы» сходятся к одной громкости (нет скачков)."""
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    quiet_phrase = _drive(ctrl, -34.0, seconds=4.0)
    loud_phrase = _drive(ctrl, -20.0, seconds=4.0)
    assert abs(quiet_phrase - loud_phrase) < 1.0
    assert abs(loud_phrase - cfg.target_rms_dbfs) < 1.0


def test_gain_change_is_slew_limited_no_jump():
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    _drive(ctrl, -34.0, seconds=4.0)   # выходим на стабильный gain тихой фразы
    g_before = ctrl.gain_db
    # Резко громкий вход: за ОДИН маленький шаг gain не должен прыгнуть целиком.
    ctrl.update_gain(-12.0, 0.02)
    assert abs(ctrl.gain_db - g_before) <= cfg.gain_release_db_per_s * 0.02 + 1e-6


def test_silence_holds_gain():
    ctrl = AdaptiveGainController(AGCConfig())
    _drive(ctrl, -30.0, seconds=3.0)
    g = ctrl.gain_db
    # Вход ниже гейта (пауза) — gain держим.
    for _ in range(50):
        ctrl.update_gain(-80.0, 0.02)
    assert ctrl.gain_db == g


# ───────────────────────── контроллер: адаптивность ─────────────────────────
def test_louder_ambient_raises_volume():
    cfg = AGCConfig()
    quiet = AdaptiveGainController(cfg)
    quiet.set_ambient(-60.0)            # тихо
    out_quiet = _drive(quiet, -30.0, seconds=5.0)

    noisy = AdaptiveGainController(cfg)
    noisy.set_ambient(-15.0)            # громкая музыка
    out_noisy = _drive(noisy, -30.0, seconds=5.0)

    assert out_noisy > out_quiet + 5.0          # ощутимо громче
    assert out_noisy <= cfg.target_rms_dbfs + cfg.max_boost_db + 0.5  # в пределах потолка


def test_ambient_boost_is_capped():
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    ctrl.set_ambient(20.0)              # абсурдно громко
    out = _drive(ctrl, -30.0, seconds=6.0)
    assert out <= cfg.target_rms_dbfs + cfg.max_boost_db + 0.5


def test_mid_phrase_adaptation_increases_gain():
    """Громкость растёт ПРЯМО в процессе фразы при внезапном шуме."""
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    ctrl.set_ambient(-55.0)
    _drive(ctrl, -30.0, seconds=3.0)
    g_before = ctrl.gain_db
    # Музыка заиграла — шум подскочил, вход (свой голос) тот же.
    ctrl.set_ambient(-18.0)
    _drive(ctrl, -30.0, seconds=2.0)
    assert ctrl.gain_db > g_before + 3.0


def test_gain_clamped_to_bounds():
    cfg = AGCConfig()
    ctrl = AdaptiveGainController(cfg)
    ctrl.set_ambient(40.0)
    _drive(ctrl, -60.0, seconds=8.0)
    assert cfg.min_gain_db <= ctrl.gain_db <= cfg.max_gain_db


# ───────────────────────── оценка внешнего шума ─────────────────────────
def test_estimator_tracks_mic_when_silent():
    est = AmbientNoiseEstimator(NoiseConfig())
    start = est.level
    est.feed(-30.0, speaking=False, self_output_dbfs=None, dt=1.0)
    assert est.level > start            # пошла вверх к -30


def test_estimator_subtracts_self_echo_while_speaking():
    cfg = NoiseConfig(echo_coupling_db=-10.0)
    est = AmbientNoiseEstimator(cfg)
    # Микрофон = только собственное эхо (выход -30 → эхо ≈ -40 на микрофоне):
    # внешнего шума нет → оценка НЕ растёт.
    est.feed(-40.0, speaking=True, self_output_dbfs=-30.0, dt=1.0)
    assert est.level == cfg.floor_dbfs


def test_estimator_raises_on_external_noise_while_speaking():
    cfg = NoiseConfig(echo_coupling_db=-10.0)
    est = AmbientNoiseEstimator(cfg)
    before = est.level
    # Микрофон громкий (-15), а свой голос лишь -30 → остаток = внешний шум.
    est.feed(-15.0, speaking=True, self_output_dbfs=-30.0, dt=1.0)
    assert est.level > before


def test_estimator_does_not_lower_while_speaking():
    cfg = NoiseConfig()
    est = AmbientNoiseEstimator(cfg)
    est.feed(-20.0, speaking=False, self_output_dbfs=None, dt=2.0)
    high = est.level
    # Под собственную речь оценку вниз не правим (тишину под голосом не слышно).
    est.feed(-90.0, speaking=True, self_output_dbfs=-30.0, dt=2.0)
    assert est.level == high


# ───────────────────────── hub: свежесть данных ─────────────────────────
def test_hub_returns_none_without_mic_feed():
    Singleton.reset(AmbientNoiseHub)
    hub = AmbientNoiseHub(AmbientNoiseEstimator(NoiseConfig()))
    assert hub.ambient_dbfs() is None       # не было кадров микрофона
    hub.feed_mic(-30.0, speaking=False, dt=1.0)
    assert hub.ambient_dbfs() is not None
    Singleton.reset(AmbientNoiseHub)
