"""Адаптивная громкость голоса JARVIS — стабильный уровень + слежение за шумом.

Проблема, которую решаем (две стороны одной монеты):

  1. СТАБИЛЬНОСТЬ. Раньше громкость прыгала от фразы к фразе: финальный
     ``gain -n`` в SoX нормализует КАЖДУЮ реплику по пику независимо, а у разных
     состояний (NORMAL/ALERT/IDLE/NIGHT) ещё и разный итоговый gain. Плюс
     авто-переключение профиля (день→NIGHT) роняло громкость скачком. Итог —
     соседние фразы звучат по-разному громко.

  2. АДАПТИВНОСТЬ. Громкость голоса должна подстраиваться под внешний шум, причём
     ПРЯМО в процессе фразы: внезапно громко заиграла музыка — Джарвис тут же
     говорит громче; стало тихо — плавно убавляет.

Решение — потоковый авто-регулятор громкости (AGC) ПОСЛЕ DSP, перед aplay:

  piper → FFT-памп → sox(DSP) → AdaptiveGainPump → aplay

(AGC включается только когда есть И sox, И aplay; иначе тракт прежний. SoX-цепочки
сохраняют свой финальный ``gain -n`` — он лишь выравнивает ПИК внутри фразы, а AGC
ниже по потоку перенормирует громкость по RMS к общей цели, нейтрализуя разнобой
пиковой нормировки и состояний между фразами.)

``AdaptiveGainController`` (чистая логика, без I/O — тестируется детерминированно):
гонит выходную громкость к ЦЕЛИ ``target_rms`` (отсюда стабильность между фразами:
цель одна и та же, состояние gain переносится между репликами и слю-лимитируется,
без скачков), а саму цель смещает по оценке внешнего шума (отсюда адаптивность).
Все изменения ограничены по скорости (attack/release) → плавно, «стабильно».

``AmbientNoiseEstimator`` оценивает уровень внешнего шума по микрофону. Пока
Джарвис молчит — доверяем замеру напрямую. Пока говорит — вычитаем по мощности
собственный голос (его уровень известен из AGC) и позволяем оценке лишь РАСТИ
(резкий скачок = внешний звук); вниз во время речи не правим — под своей речью
тишину не «слышно». Коэффициент связи микрофон↔динамик и пороги — калибруются под
железо (см. JARVIS_AGC_* ниже); дефолты — консервативный старт.

``AmbientNoiseHub`` — потокобезопасный мост: слушатель (entry_point) пишет уровень
микрофона, AGC-памп — свой выходной уровень и читает оценку шума. Без него движку
и слушателю пришлось бы знать друг о друге (цикл импортов).

numpy — мягкая зависимость: есть → быстрый путь, нет → честный pure-python fallback
(как в fft_analyzer). Регулятор остаётся тестируемым без аудио-железа.
"""
from __future__ import annotations

import math
import os
import struct
import threading
import time
from dataclasses import dataclass

from src.common.singleton import Singleton

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy есть в рантайме и в CI
    np = None  # type: ignore
    _HAS_NUMPY = False

DBFS_FLOOR = -120.0  # «тишина» в dBFS — ниже не опускаемся (защита от log10(0))
_INT16_MAX = 32767
_INT16_MIN = -32768


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def amp_to_dbfs(rms_lin: float) -> float:
    """RMS-амплитуда [0..1] → dBFS. 0/отрицательное → пол (DBFS_FLOOR)."""
    if rms_lin <= 1e-9:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(rms_lin))


def dbfs_to_amp(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def rms_dbfs_pcm16(buf: bytes) -> float:
    """RMS уровня сырого S16_LE PCM в dBFS. Пустой буфер → пол."""
    if not buf:
        return DBFS_FLOOR
    if len(buf) % 2:
        buf = buf[:-1]
    if not buf:
        return DBFS_FLOOR
    if _HAS_NUMPY:
        a = np.frombuffer(buf, dtype=np.int16).astype(np.float64) / 32768.0
        if a.size == 0:
            return DBFS_FLOOR
        rms = math.sqrt(float(np.mean(a * a)))
    else:
        n = len(buf) // 2
        vals = struct.unpack("<" + "h" * n, buf)
        acc = 0.0
        for v in vals:
            x = v / 32768.0
            acc += x * x
        rms = math.sqrt(acc / n) if n else 0.0
    return amp_to_dbfs(rms)


def power_subtract_db(total_db: float, part_db: float) -> float:
    """Вычесть мощность part из total в dB-домене (для де-эха).

    Возвращает dB остатка: 10·log10(10^(total/10) − 10^(part/10)), пол при ≤0.
    Согласовано с RMS-dBFS (амплитуда), т.к. 10·log10(amp²)=20·log10(amp)."""
    total_p = 10.0 ** (total_db / 10.0)
    part_p = 10.0 ** (part_db / 10.0)
    residual = total_p - part_p
    if residual <= 1e-12:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 10.0 * math.log10(residual))


def _slew(current: float, target: float, up_per_s: float, down_per_s: float, dt: float) -> float:
    """Подтянуть current к target с раздельным ограничением скорости вверх/вниз."""
    if dt <= 0.0:
        return current
    if target > current:
        return min(target, current + up_per_s * dt)
    return max(target, current - down_per_s * dt)


# ───────────────────────────── контроллер ──────────────────────────────────
@dataclass
class AGCConfig:
    """Параметры авто-регулятора громкости. Все — переопределяемы через env
    (JARVIS_AGC_*), чтобы калибровать под железо без правки кода."""
    target_rms_dbfs: float = -16.0   # целевая громкость голоса в тихой комнате
    noise_ref_dbfs: float = -50.0    # уровень шума, считающийся «тихо» (нет добавки)
    noise_track: float = 0.9         # +dB цели на каждый +1 dB шума сверх ref
    max_boost_db: float = 12.0       # потолок прибавки в шумной комнате
    max_cut_db: float = 8.0          # потолок убавки в очень тихой
    gate_dbfs: float = -45.0         # тише этого на входе = пауза/тишина → держим gain
    min_gain_db: float = -15.0
    max_gain_db: float = 18.0
    gain_attack_db_per_s: float = 20.0   # как быстро РАСТЁТ громкость (громче)
    gain_release_db_per_s: float = 10.0  # как быстро УБАВЛЯЕТ (плавнее)
    target_slew_db_per_s: float = 12.0   # скорость слежения цели за шумом
    env_attack_db_per_s: float = 220.0   # огибающая входа вверх — почти мгновенно
    env_release_db_per_s: float = 70.0   # вниз — мягче (не дёргаться на согласных)

    @classmethod
    def from_env(cls) -> "AGCConfig":
        c = cls()
        c.target_rms_dbfs = _env_float("JARVIS_AGC_TARGET_DBFS", c.target_rms_dbfs)
        c.noise_ref_dbfs = _env_float("JARVIS_AGC_NOISE_REF_DBFS", c.noise_ref_dbfs)
        c.noise_track = _env_float("JARVIS_AGC_NOISE_TRACK", c.noise_track)
        c.max_boost_db = _env_float("JARVIS_AGC_MAX_BOOST_DB", c.max_boost_db)
        c.max_cut_db = _env_float("JARVIS_AGC_MAX_CUT_DB", c.max_cut_db)
        c.gate_dbfs = _env_float("JARVIS_AGC_GATE_DBFS", c.gate_dbfs)
        c.min_gain_db = _env_float("JARVIS_AGC_MIN_GAIN_DB", c.min_gain_db)
        c.max_gain_db = _env_float("JARVIS_AGC_MAX_GAIN_DB", c.max_gain_db)
        return c


class AdaptiveGainController:
    """Потоковый AGC: вход (PCM-чанки) → выход с плавно подстраиваемым gain.

    Состояние (gain, огибающая, смещение цели) переживает фразы → между репликами
    громкость непрерывна (нет скачков). ``set_ambient`` подаёт текущую оценку
    внешнего шума; цель громкости смещается вверх в шуме и вниз в тишине, всё —
    со слю-лимитом (attack/release). Чистая логика, без потоков/процессов."""

    def __init__(self, config: AGCConfig | None = None) -> None:
        self.cfg = config or AGCConfig()
        self._gain_db = 0.0
        self._env_db = DBFS_FLOOR
        self._target_off_db = 0.0    # текущее (слю-лимитированное) смещение цели
        self._ambient_db: float | None = None
        self._state_bias_db = 0.0    # смещение цели по состоянию (NIGHT тише и т.п.)
        self._last_out_db = DBFS_FLOOR
        self._lock = threading.Lock()

    # --- входы ---
    def set_ambient(self, ambient_dbfs: float | None) -> None:
        with self._lock:
            self._ambient_db = ambient_dbfs

    def set_state_bias(self, bias_db: float) -> None:
        """Целевое смещение по состоянию голоса (напр. NIGHT тише). Переходы
        сглаживает слю-лимит gain — резкого скачка между фразами не будет."""
        with self._lock:
            self._state_bias_db = bias_db

    @property
    def gain_db(self) -> float:
        return self._gain_db

    @property
    def output_dbfs(self) -> float:
        """Оценка фактического уровня выхода (вход + применённый gain)."""
        return self._last_out_db

    # --- ядро ---
    def _desired_target_offset(self) -> float:
        amb = self._ambient_db
        if amb is None:
            return 0.0  # нет данных о шуме → база (тихая комната)
        raw = self.cfg.noise_track * (amb - self.cfg.noise_ref_dbfs)
        return max(-self.cfg.max_cut_db, min(self.cfg.max_boost_db, raw))

    def update_gain(self, in_dbfs: float, dt: float) -> float:
        """Пересчитать gain (dB) по уровню входа in_dbfs за интервал dt. Возвращает
        gain в dB. Выделено из process() ради детерминированных юнит-тестов."""
        with self._lock:
            cfg = self.cfg
            # 1) Огибающая входа (быстро вверх, мягче вниз).
            self._env_db = _slew(
                self._env_db, in_dbfs,
                cfg.env_attack_db_per_s, cfg.env_release_db_per_s, dt,
            )
            # 2) Цель: база + смещение по шуму, слю-лимитированное.
            desired_off = self._desired_target_offset()
            self._target_off_db = _slew(
                self._target_off_db, desired_off,
                cfg.target_slew_db_per_s, cfg.target_slew_db_per_s, dt,
            )
            desired_loudness = cfg.target_rms_dbfs + self._state_bias_db + self._target_off_db
            # 3) Тишина/пауза — gain НЕ трогаем (иначе раскачаем шум в паузах и
            # дадим громкий «всплеск» на возврате речи). Гейтим по СЫРОМУ входу,
            # а не по огибающей: огибающая спадает плавно и на спуске успела бы
            # задрать gain до возврата голоса. Сырой уровень падает в паузу сразу.
            if in_dbfs > cfg.gate_dbfs:
                desired_gain = desired_loudness - self._env_db
                desired_gain = max(cfg.min_gain_db, min(cfg.max_gain_db, desired_gain))
                self._gain_db = _slew(
                    self._gain_db, desired_gain,
                    cfg.gain_attack_db_per_s, cfg.gain_release_db_per_s, dt,
                )
            self._last_out_db = self._env_db + self._gain_db
            return self._gain_db

    def process(self, pcm: bytes, dt: float) -> bytes:
        """Применить адаптивный gain к чанку S16_LE PCM. Возвращает новый PCM."""
        if not pcm:
            return pcm
        in_db = rms_dbfs_pcm16(pcm)
        gain_db = self.update_gain(in_db, dt)
        return apply_gain_pcm16(pcm, dbfs_to_amp(gain_db))


def apply_gain_pcm16(pcm: bytes, gain_lin: float) -> bytes:
    """Умножить S16_LE PCM на линейный gain с жёстким клиппингом в int16."""
    if not pcm or abs(gain_lin - 1.0) < 1e-6:
        return pcm
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if _HAS_NUMPY:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * gain_lin
        np.clip(a, _INT16_MIN, _INT16_MAX, out=a)
        return a.astype(np.int16).tobytes()
    n = len(pcm) // 2
    vals = struct.unpack("<" + "h" * n, pcm)
    out = [max(_INT16_MIN, min(_INT16_MAX, int(v * gain_lin))) for v in vals]
    return struct.pack("<" + "h" * n, *out)


# ───────────────────────── оценка внешнего шума ─────────────────────────────
@dataclass
class NoiseConfig:
    attack_db_per_s: float = 40.0   # рост оценки (новый шум) — быстро
    release_db_per_s: float = 6.0   # спад — медленно (не дёргать на паузах)
    echo_coupling_db: float = -10.0  # насколько голос Джарвиса тише на микрофоне,
    # чем на выходе AGC (динамик→микрофон). Калибруется; консервативно −10 dB.
    floor_dbfs: float = -90.0

    @classmethod
    def from_env(cls) -> "NoiseConfig":
        c = cls()
        c.attack_db_per_s = _env_float("JARVIS_AGC_NOISE_ATTACK", c.attack_db_per_s)
        c.release_db_per_s = _env_float("JARVIS_AGC_NOISE_RELEASE", c.release_db_per_s)
        c.echo_coupling_db = _env_float("JARVIS_AGC_ECHO_COUPLING_DB", c.echo_coupling_db)
        return c


class AmbientNoiseEstimator:
    """Оценка уровня внешнего шума (dBFS) по кадрам микрофона. Чистая логика."""

    def __init__(self, config: NoiseConfig | None = None) -> None:
        self.cfg = config or NoiseConfig()
        self._level = self.cfg.floor_dbfs

    @property
    def level(self) -> float:
        return self._level

    def feed(
        self,
        mic_dbfs: float,
        speaking: bool,
        self_output_dbfs: float | None,
        dt: float,
    ) -> float:
        """Обновить оценку. ``speaking`` — говорит ли сейчас Джарвис (эхо-гейт)."""
        cfg = self.cfg
        if not speaking:
            # Джарвис молчит — микрофон = чистый внешний фон, доверяем напрямую.
            self._level = _slew(
                self._level, max(cfg.floor_dbfs, mic_dbfs),
                cfg.attack_db_per_s, cfg.release_db_per_s, dt,
            )
            return self._level
        # Джарвис говорит: вычитаем по мощности собственный голос и позволяем
        # оценке лишь РАСТИ (внезапный внешний звук). Вниз под своей речью не
        # правим — там тишину не различить (вернёмся к спаду в ближайшей паузе).
        if self_output_dbfs is not None:
            residual = power_subtract_db(mic_dbfs, self_output_dbfs + cfg.echo_coupling_db)
        else:
            residual = mic_dbfs
        residual = max(cfg.floor_dbfs, residual)
        if residual > self._level:
            self._level = _slew(self._level, residual, cfg.attack_db_per_s, 0.0, dt)
        return self._level


class AmbientNoiseHub(metaclass=Singleton):
    """Потокобезопасный мост между слушателем (микрофон) и AGC-пампом (голос).

    Слушатель: ``feed_mic(mic_dbfs, speaking)`` каждый кадр. AGC: ``set_self_output``
    каждый чанк и ``ambient_dbfs()`` для текущей оценки. Свежесть self-output
    ограничена — устаревший (Джарвис давно молчит) не вычитаем."""

    _SELF_OUTPUT_TTL = 1.0  # с: старше — считаем, что Джарвис уже не звучит
    _MIC_TTL = 3.0          # с: нет свежих кадров микрофона → оценка недоступна

    def __init__(self, estimator: AmbientNoiseEstimator | None = None) -> None:
        self._est = estimator or AmbientNoiseEstimator(NoiseConfig.from_env())
        self._lock = threading.Lock()
        self._self_out_db = DBFS_FLOOR
        self._self_out_ts = 0.0
        self._last_mic_ts = 0.0

    def set_self_output(self, dbfs: float) -> None:
        with self._lock:
            self._self_out_db = dbfs
            self._self_out_ts = time.monotonic()

    def feed_mic(self, mic_dbfs: float, speaking: bool, dt: float | None = None) -> float:
        with self._lock:
            now = time.monotonic()
            if dt is None:
                dt = (now - self._last_mic_ts) if self._last_mic_ts else 0.5
            self._last_mic_ts = now
            fresh = (now - self._self_out_ts) < self._SELF_OUTPUT_TTL
            self_out = self._self_out_db if (speaking and fresh) else None
            return self._est.feed(mic_dbfs, speaking, self_out, dt)

    def ambient_dbfs(self) -> float | None:
        """Текущая оценка шума, либо None если давно нет кадров микрофона —
        чтобы AGC не принял «нет данных» за «очень тихо» и не срезал громкость."""
        with self._lock:
            if not self._last_mic_ts or (time.monotonic() - self._last_mic_ts) > self._MIC_TTL:
                return None
            return self._est.level


__all__ = [
    "AGCConfig",
    "NoiseConfig",
    "AdaptiveGainController",
    "AmbientNoiseEstimator",
    "AmbientNoiseHub",
    "amp_to_dbfs",
    "dbfs_to_amp",
    "rms_dbfs_pcm16",
    "power_subtract_db",
    "apply_gain_pcm16",
]
