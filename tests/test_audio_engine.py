"""Тесты кинематографического голосового тракта (AcousticEngine).

Проверяем чистую логику: парсинг когнитивной просодии (<say speed pause>),
конвертацию speed→length_scale, дефолты состояний, модуляцию нагрузкой,
валидность DSP-цепочек «Bettany Signature» и маршрутизацию speak()→_play.
Реальный звук не воспроизводим."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.audio_engine import (
    _DSP_CHAINS,
    _STATE_PROSODY,
    _TONE_TO_STATE,
    AcousticEngine,
    VoiceState,
    parse_prosody,
    resolve_prosody,
)
from src.common.event_bus import EventBus, SystemLoad


# ───────────────────────── когнитивная просодия ─────────────────────────
def test_parse_prosody_extracts_and_strips_tag():
    clean, speed, pause = parse_prosody('<say speed="1.1" pause="0.2">Сэр, готово.</say>')
    assert clean == "Сэр, готово."
    assert speed == 1.1
    assert pause == 0.2


def test_parse_prosody_no_tag_returns_text():
    clean, speed, pause = parse_prosody("Просто фраза без тега.")
    assert clean == "Просто фраза без тега."
    assert speed is None and pause is None


def test_parse_prosody_strips_stray_tags_so_never_spoken():
    # Битый/незакрытый тег не должен быть произнесён буквально.
    clean, _, _ = parse_prosody('Текст <say speed="2"> и хвост')
    assert "<say" not in clean and "say>" not in clean
    assert "Текст" in clean and "хвост" in clean


def test_parse_prosody_single_quotes_and_partial_attrs():
    clean, speed, pause = parse_prosody("<say speed='0.9'>Размеренно…</say>")
    assert clean == "Размеренно…"
    assert speed == 0.9
    assert pause is None


# ───────────────────── speed → length_scale (инверсия) ─────────────────────
def test_faster_speed_means_shorter_length_scale():
    fast = resolve_prosody(VoiceState.NORMAL, speed=1.5, load=SystemLoad.NORMAL)
    slow = resolve_prosody(VoiceState.NORMAL, speed=0.7, load=SystemLoad.NORMAL)
    assert fast.length_scale < 1.0 < slow.length_scale


def test_length_scale_clamped():
    extreme = resolve_prosody(VoiceState.NORMAL, speed=99.0, load=SystemLoad.NORMAL)
    assert extreme.length_scale >= 0.55


# ─────────────────────── дефолты состояний ───────────────────────
def test_idle_is_measured_alert_is_fast():
    idle = resolve_prosody(VoiceState.IDLE, load=SystemLoad.NORMAL)
    alert = resolve_prosody(VoiceState.ALERT, load=SystemLoad.NORMAL)
    # IDLE медленнее (length>1), ALERT быстрее (length<1); ТЗ: 0.95 vs 1.15.
    assert idle.length_scale > 1.0
    assert alert.length_scale < 1.0
    # Паузы: ALERT короткие, IDLE длинные.
    assert alert.sentence_silence < idle.sentence_silence


def test_explicit_params_override_state_defaults():
    p = resolve_prosody(VoiceState.IDLE, speed=2.0, pause=0.0, load=SystemLoad.NORMAL)
    assert p.length_scale < 1.0          # speed=2 → быстро, несмотря на IDLE
    assert p.sentence_silence == 0.0


def test_system_load_speeds_speech_up():
    normal = resolve_prosody(VoiceState.NORMAL, load=SystemLoad.NORMAL)
    crit = resolve_prosody(VoiceState.NORMAL, load=SystemLoad.CRITICAL)
    # Под нагрузкой говорим быстрее (меньше length_scale) и короче паузы.
    assert crit.length_scale < normal.length_scale
    assert crit.sentence_silence < normal.sentence_silence


def test_state_prosody_table_matches_spec():
    # ТЗ: IDLE speed=0.95 pause=0.4; ALERT speed=1.15 pause=0.1.
    assert _STATE_PROSODY[VoiceState.IDLE] == (0.95, 0.40)
    assert _STATE_PROSODY[VoiceState.ALERT] == (1.15, 0.10)


# ─────────────────────── DSP-цепочки «Bettany Signature» ───────────────────────
def test_all_states_have_dsp_chain():
    for state in VoiceState:
        chain = _DSP_CHAINS[state]
        assert chain and all(isinstance(a, str) for a in chain)


def test_normal_chain_has_signature_filters():
    chain = _DSP_CHAINS[VoiceState.NORMAL]
    for effect in ("highpass", "equalizer", "compand", "chorus", "reverb", "gain"):
        assert effect in chain, f"NORMAL chain missing {effect}"


def test_alert_chain_is_dry_no_reverb():
    # ALERT — сухо и немедленно: без reverb и без хоруса.
    chain = _DSP_CHAINS[VoiceState.ALERT]
    assert "reverb" not in chain
    assert "chorus" not in chain
    assert "compand" in chain          # но плотность сохраняем


def test_every_chain_normalizes_output():
    # gain -n в конце — защита от клиппинга после presence/air-бустов и reverb.
    for state in VoiceState:
        assert "gain" in _DSP_CHAINS[state]


# ─────────────────────── tone/state coercion ───────────────────────
def test_tone_to_state_mapping():
    assert _TONE_TO_STATE["alert"] == VoiceState.ALERT
    assert _TONE_TO_STATE["idle"] == VoiceState.IDLE
    assert _TONE_TO_STATE["normal"] == VoiceState.NORMAL


def test_coerce_state_accepts_tone_strings_and_enum():
    assert AcousticEngine._coerce_state("alert") == VoiceState.ALERT
    assert AcousticEngine._coerce_state(VoiceState.IDLE) == VoiceState.IDLE
    assert AcousticEngine._coerce_state("NORMAL") == VoiceState.NORMAL
    assert AcousticEngine._coerce_state("garbage") == VoiceState.NORMAL


# ─────────────────────── speak() маршрутизация (без аудио) ───────────────────────
def _make_engine() -> AcousticEngine:
    return AcousticEngine(
        EventBus(),
        piper_path="/nonexistent/piper",
        voice_model="/nonexistent/model.onnx",
        voice_config="/nonexistent/model.json",
    )


def test_speak_routes_parsed_prosody_to_play():
    eng = _make_engine()
    captured: list = []
    done = threading.Event()

    def fake_play(text, state, speed, pause):
        captured.append((text, state, speed, pause))
        done.set()

    eng._play = fake_play  # type: ignore[assignment]
    try:
        eng.speak('<say speed="1.2" pause="0.15">Внимание, сэр.</say>', state="alert")
        assert done.wait(timeout=3.0), "worker не вызвал _play"
    finally:
        eng.shutdown()

    text, state, speed, pause = captured[0]
    assert text == "Внимание, сэр."
    assert state == VoiceState.ALERT
    assert speed == 1.2 and pause == 0.15


def test_speak_empty_is_noop():
    eng = _make_engine()
    called = threading.Event()
    eng._play = lambda *a, **k: called.set()  # type: ignore[assignment]
    try:
        eng.speak("   ")
        eng.speak('<say speed="1.0"></say>')
        assert not called.wait(timeout=0.5), "пустая фраза не должна играться"
    finally:
        eng.shutdown()
