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
    assert {"speak_response", "set_hud_state"} <= convo


def test_internal_monologue_excluded_from_hot_path():
    """На слабом железе internal_monologue не в боевых подмножествах (лишний
    раунд диалога при декоде ~1 т/с), хотя схема инструмента сохранена."""
    assert "internal_monologue" in t.TOOLS_BY_NAME
    for cat in ("SYSTEM_OPS", "UI_CONTROL", "PENTEST_RECON", "CONVERSATION"):
        names = {s["function"]["name"] for s in t.tools_for_category(cat)}
        assert "internal_monologue" not in names, f"{cat} всё ещё тянет internal_monologue"


def test_action_categories_offer_run_skill_and_bash():
    """Архитектурный переворот: action-категории дают run_skill (готовые навыки,
    предпочтительно) + execute_bash (gated-fallback для длинного хвоста)."""
    ops = [s["function"]["name"] for s in t.tools_for_category("SYSTEM_OPS")]
    assert ops == ["run_skill", "execute_bash"]


def test_action_categories_share_identical_toolset():
    """Кэш-стабильность: три action-категории дают ИДЕНТИЧНЫЙ набор инструментов
    (run_skill с ГЛОБАЛЬНЫМ enum + execute_bash) → сервер переиспользует
    KV-префикс tool-блока на переключении категории."""
    ops = t.tools_for_category("SYSTEM_OPS")
    ui = t.tools_for_category("UI_CONTROL")
    pen = t.tools_for_category("PENTEST_RECON")
    assert ops == ui == pen, "action-категории должны делить идентичный tool-набор"


def test_unknown_category_falls_back_to_action_set():
    names = [s["function"]["name"] for s in t.tools_for_category("???")]
    assert names == ["run_skill", "execute_bash"]


def test_run_skill_enum_matches_registry():
    """skill_id enum в run_skill = все зарегистрированные навыки."""
    from src import skills
    tool = t.tools_for_category("UI_CONTROL")[0]
    assert tool["function"]["name"] == "run_skill"
    enum = tool["function"]["parameters"]["properties"]["skill_id"]["enum"]
    assert set(enum) == set(skills.skill_ids())
    assert "enable_night_mode" in enum  # пример из ТЗ оператора
    # required: skill_id + reply (модель обязана и выбрать навык, и ответить).
    assert set(tool["function"]["parameters"]["required"]) == {"skill_id", "reply"}


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


def test_run_skill_args_coercion():
    a = t.RunSkillArgs.from_dict(
        {"skill_id": "  open_files  ", "reply": " открываю ", "mood": "IRONIC",
         "args": {"percent": 30}}
    )
    assert a.skill_id == "open_files"
    assert a.reply == "открываю"
    assert a.mood == t.SpeakMood.IRONIC
    assert a.args == {"percent": 30}
    # мусорные/пропущенные поля → дефолты, args не-объект → пустой dict
    b = t.RunSkillArgs.from_dict({"skill_id": "x", "mood": "banana", "args": "nope"})
    assert b.mood == t.SpeakMood.PROFESSIONAL
    assert b.args == {}
    assert b.reply == ""
