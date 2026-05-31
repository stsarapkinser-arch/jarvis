"""Тесты макросов/сценариев (композиция навыков одной фразой).

Юнит реестра макросов + целостность встроенных (шаги ссылаются на реальные,
неразрушительные навыки) + поток _try_macro в оркестраторе (последовательность
навыков, сбор телеметрии, короткое замыкание process_intent мимо 3B).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import skills
from src.skills import macros
from src.common.singleton import Singleton
from src.inference.router import IntentCategory


@pytest.fixture(autouse=True)
def reset_singletons():
    Singleton.reset()
    yield
    Singleton.reset()


# ───────────────────────── реестр / целостность ─────────────────────────
def test_builtin_macros_registered():
    ids = {m.id for m in macros.all_macros()}
    assert {"pentest_workspace", "focus_mode", "system_briefing", "start_day"} <= ids


def test_match_macro_exact_and_normalized():
    assert macros.match_macro("режим фокуса").id == "focus_mode"
    assert macros.match_macro("  Режим Фокуса!  ").id == "focus_mode"
    assert macros.match_macro("рабочее место для пентеста").id == "pentest_workspace"


def test_match_macro_no_partial():
    assert macros.match_macro("режим фокуса на минималках") is None
    assert macros.match_macro("") is None


def test_every_macro_step_references_real_nondestructive_skill():
    for m in macros.all_macros():
        assert m.steps, f"макрос {m.id} без шагов"
        for skill_id, args in m.steps:
            sk = skills.get(skill_id)
            assert sk is not None, f"макрос {m.id}: несуществующий навык {skill_id}"
            assert not sk.destructive, f"макрос {m.id}: разрушительный шаг {skill_id}"
            assert isinstance(args, dict)


def test_macro_aliases_do_not_collide_with_skill_aliases():
    """Фраза не должна одновременно быть и макросом, и навыком."""
    for m in macros.all_macros():
        for alias in m.aliases:
            assert skills.match_alias(alias) is None, (
                f"алиас {alias!r} макроса {m.id} конфликтует с навыком"
            )


# ───────────────────────── поток в оркестраторе ─────────────────────────
class _NullMemory:
    def remember(self, *a, **kw):
        return None

    def recall(self, *a, **kw):
        return []


def _make_jarvis():
    from src.core.orchestrator import Jarvis
    j = Jarvis()
    j.memory = _NullMemory()
    return j


def _fake_skill(id, *, speaks_result=False, destructive=False):
    from src.skills.registry import Skill
    return Skill(id=id, category=IntentCategory.SYSTEM_OPS, description=id,
                 handler=None, speaks_result=speaks_result, destructive=destructive)


def test_try_macro_runs_steps_and_briefs_telemetry():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    macro = macros.Macro(
        id="t", description="d", reply="Интро.",
        steps=(("rep", {}), ("act", {})), aliases=("тест сценарий",),
    )
    fakes = {"rep": _fake_skill("rep", speaks_result=True),
             "act": _fake_skill("act", speaks_result=False)}
    invoked: list = []

    async def fake_invoke(sk, args):
        invoked.append(sk.id)
        return "Загрузка 50 процентов." if sk.id == "rep" else "launched"

    with patch("src.core.orchestrator.macros.match_macro", return_value=macro), \
         patch("src.core.orchestrator.skills.get", side_effect=lambda sid: fakes.get(sid)), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        j._invoke_skill = fake_invoke  # type: ignore[assignment]
        handled = asyncio.run(j._try_macro("тест сценарий"))

    assert handled is True
    assert invoked == ["rep", "act"]          # все шаги по порядку
    assert "Интро." in spoken                 # интро озвучено
    assert any("50 процентов" in s for s in spoken)  # телеметрия speaks_result
    # запись реплики в буфер диалога.
    assert j._dialogue and j._dialogue[-1][0] == "тест сценарий"


def test_try_macro_skips_destructive_step():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    macro = macros.Macro(id="t", description="d", reply="",
                         steps=(("safe", {}), ("danger", {})), aliases=("x",))
    fakes = {"safe": _fake_skill("safe"),
             "danger": _fake_skill("danger", destructive=True)}
    invoked: list = []

    async def fake_invoke(sk, args):
        invoked.append(sk.id)
        return "ok"

    with patch("src.core.orchestrator.macros.match_macro", return_value=macro), \
         patch("src.core.orchestrator.skills.get", side_effect=lambda sid: fakes.get(sid)), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        j._invoke_skill = fake_invoke  # type: ignore[assignment]
        asyncio.run(j._try_macro("x"))

    assert invoked == ["safe"], "разрушительный шаг должен быть пропущен"


def test_try_macro_no_match_returns_false():
    j = _make_jarvis()
    with patch("src.core.orchestrator.macros.match_macro", return_value=None):
        assert asyncio.run(j._try_macro("просто фраза")) is False


def test_process_intent_macro_skips_agent():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    macro = macros.Macro(id="t", description="d", reply="r", steps=(), aliases=("c",))

    async def boom(*a, **kw):
        raise AssertionError("агент не должен вызываться при макросе")

    j._run_agent_for_intent = boom  # type: ignore[assignment]
    with patch("src.core.orchestrator.macros.match_macro", return_value=macro), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        res = asyncio.run(j.process_intent("c"))
    assert res == "[macro]"
