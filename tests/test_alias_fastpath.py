"""Тесты alias fast-path оркестратора.

Точная фраза-алиас должна вызывать навык напрямую — без маршрутизатора и без
3B (главный рычаг латентности на N100). Проверяем: прямой вызов хендлера,
озвучку (ack для обычных, живой результат для speaks_result), подтверждение
разрушительных, отказ на не-совпадении и то, что process_intent при матче
вообще не зовёт агентный цикл.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.singleton import Singleton
from src.inference.router import IntentCategory


@pytest.fixture(autouse=True)
def reset_singletons():
    Singleton.reset()
    yield
    Singleton.reset()


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


def _fake_skill(**kw):
    from src.skills.registry import Skill
    defaults = dict(
        id="open_terminal", category=IntentCategory.UI_CONTROL,
        description="открыть терминал", handler=None,
    )
    defaults.update(kw)
    return Skill(**defaults)


_ACKS = {"Готово.", "Сделано.", "Принято.", "Выполнено.", "Есть."}


def test_fastpath_invokes_skill_and_speaks_ack():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "raw-result"

    fake = _fake_skill(handler=handler)
    with patch("src.core.orchestrator.skills.match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_alias_fastpath("терминал"))

    assert handled is True
    assert ran == [{}], "хендлер должен вызваться с пустыми args"
    # обычный навык (не speaks_result) → короткий ack, а НЕ сырой результат.
    assert spoken and spoken[0] in _ACKS
    assert "raw-result" not in spoken


def test_fastpath_speaks_result_for_telemetry():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "Процессор загружен на 50 процентов."

    fake = _fake_skill(id="report_cpu", category=IntentCategory.SYSTEM_OPS,
                       handler=handler, speaks_result=True)
    with patch("src.core.orchestrator.skills.match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        asyncio.run(j._try_alias_fastpath("нагрузка процессора"))

    assert any("50 процентов" in s for s in spoken)


def test_fastpath_destructive_queues_confirmation():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "done"

    fake = _fake_skill(id="wipe", category=IntentCategory.SYSTEM_OPS,
                       handler=handler, destructive=True)
    with patch("src.core.orchestrator.skills.match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_alias_fastpath("снеси всё"))

    assert handled is True
    assert ran == [], "разрушительный навык не исполняется до подтверждения"
    assert j._pending_skill is not None
    assert j._pending_skill["skill"].id == "wipe"
    assert j._pending_skill["args"] == {}


def test_fastpath_no_match_returns_false():
    j = _make_jarvis()
    with patch("src.core.orchestrator.skills.match_alias", return_value=None):
        handled = asyncio.run(j._try_alias_fastpath("в чём смысл жизни"))
    assert handled is False


def test_process_intent_fastpath_skips_agent_loop():
    """При точном алиасе process_intent отвечает мгновенно, не трогая 3B."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "ok"

    fake = _fake_skill(handler=handler)
    agent_called = {"hit": False}

    async def fake_agent(*a, **kw):
        agent_called["hit"] = True
        raise AssertionError("агентный цикл не должен вызываться при fast-path")

    j._run_agent_for_intent = fake_agent  # type: ignore[assignment]
    with patch("src.core.orchestrator.skills.match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        res = asyncio.run(j.process_intent("терминал"))

    assert res == "[alias_fastpath]"
    assert agent_called["hit"] is False
