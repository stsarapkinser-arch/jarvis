"""Тесты JSON-схем инструментов и типизированных аргументов (Phase 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import tools as t


def test_all_tools_have_valid_openai_shape():
    names = set()
    for s in t.TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict) and params["properties"]
        for req in params.get("required", []):
            assert req in params["properties"], f"{fn['name']}: required {req} not in properties"
        json.dumps(s)  # должна быть JSON-сериализуемой
        names.add(fn["name"])
    assert names == {
        "internal_monologue", "speak_response", "set_hud_state",
        "read_telemetry", "execute_bash",
    }


def test_enum_constraints_match_python_enums():
    speak = t.TOOLS_BY_NAME["speak_response"]["function"]["parameters"]["properties"]["mood"]
    assert set(speak["enum"]) == {m.value for m in t.SpeakMood}
    hud = t.TOOLS_BY_NAME["set_hud_state"]["function"]["parameters"]["properties"]["animation"]
    assert set(hud["enum"]) == {a.value for a in t.HudAnimation}
    tel = t.TOOLS_BY_NAME["read_telemetry"]["function"]["parameters"]["properties"]["sensor"]
    assert set(tel["enum"]) == {s.value for s in t.TelemetrySensor}


def test_conversation_category_cannot_execute_bash():
    """Безопасность: чистый разговор не должен иметь доступа к системе."""
    convo = {s["function"]["name"] for s in t.tools_for_category("CONVERSATION")}
    assert "execute_bash" not in convo
    assert {"speak_response", "internal_monologue", "set_hud_state"} <= convo


def test_system_ops_has_full_toolset():
    ops = {s["function"]["name"] for s in t.tools_for_category("SYSTEM_OPS")}
    assert {"execute_bash", "read_telemetry", "speak_response", "set_hud_state"} <= ops


def test_unknown_category_returns_all_tools():
    assert len(t.tools_for_category("???")) == len(t.TOOL_SCHEMAS)


def test_parse_arguments_robust():
    assert t.parse_arguments('{"a": 1}') == {"a": 1}
    assert t.parse_arguments({"a": 1}) == {"a": 1}
    assert t.parse_arguments("") == {}
    assert t.parse_arguments("   ") == {}
    assert t.parse_arguments("not json at all") == {}
    assert t.parse_arguments(None) == {}
    assert t.parse_arguments("[1, 2, 3]") == {}  # валидный JSON, но не объект
    assert t.parse_arguments(42) == {}


def test_execute_bash_args_coercion():
    a = t.ExecuteBashArgs.from_dict(
        {"command": "  ls -la  ", "requires_sudo": "true", "background": 1}
    )
    assert a.command == "ls -la"
    assert a.requires_sudo is True
    assert a.background is True
    # пропущенные поля → дефолты
    b = t.ExecuteBashArgs.from_dict({"command": "pwd"})
    assert b.requires_sudo is False and b.background is False


def test_speak_args_mood_fallback_and_case_insensitive():
    assert t.SpeakArgs.from_dict({"text": "hi", "mood": "banana"}).mood == t.SpeakMood.PROFESSIONAL
    assert t.SpeakArgs.from_dict({"text": "hi", "mood": "ALERT"}).mood == t.SpeakMood.ALERT
    assert t.SpeakArgs.from_dict({"text": "hi"}).mood == t.SpeakMood.PROFESSIONAL


def test_hud_args_defaults_and_normalisation():
    a = t.HudArgs.from_dict({"color": "RED"})
    assert a.color == "red"
    assert a.animation == t.HudAnimation.PULSE
    b = t.HudArgs.from_dict({"color": "amber", "animation": "GLITCH"})
    assert b.animation == t.HudAnimation.GLITCH


def test_telemetry_args_fallback():
    assert t.TelemetryArgs.from_dict({"sensor": "cpu"}).sensor == t.TelemetrySensor.CPU
    assert t.TelemetryArgs.from_dict({"sensor": "weird"}).sensor == t.TelemetrySensor.CPU
    assert t.TelemetryArgs.from_dict({"sensor": "pixel_phone"}).sensor == t.TelemetrySensor.PIXEL_PHONE
