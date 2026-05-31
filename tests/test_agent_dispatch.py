"""Тесты привязки tool-вызовов к подсистемам Jarvis (Phase 3 + 5).

Проверяем, что каждый инструмент бьёт в нужную подсистему: speak_response →
TTS, set_hud_state → шина HUD_STATE, read_telemetry → SystemState,
execute_bash → конвейер ShadowExec (со стопом цикла на подтверждении/реджекте),
internal_monologue → лог (без речи).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.event_bus import Event, EventType
from src.common.singleton import Singleton
from src.inference.agent import ToolCall
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


def _state():
    from src.core.orchestrator import _IntentState
    return _IntentState(category=IntentCategory.SYSTEM_OPS)


def _call(name, args):
    return ToolCall(id="c1", name=name, arguments=args)


def test_internal_monologue_is_silent():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda *a, **kw: spoken.append(a)
    res = asyncio.run(
        j._dispatch_tool(_call("internal_monologue", {"thought": "подумаю"}), "intent", {}, _state())
    )
    assert res.content == "logged"
    assert spoken == []


def test_speak_response_calls_say_with_mapped_tone():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append((text, tone))
    st = _state()
    res = asyncio.run(
        j._dispatch_tool(
            _call("speak_response", {"text": "Сэр, готово", "mood": "alert"}), "intent", {}, st
        )
    )
    assert spoken and spoken[0] == ("Сэр, готово", "alert")
    assert st.spoke is True
    assert res.content == "spoken"
    # speak_response терминален — иначе при tool_choice=required цикл не остановится.
    assert res.stop is True


def test_set_hud_state_publishes_hud_event():
    j = _make_jarvis()
    events: list = []

    async def _capture(ev):
        events.append(ev)

    j.bus.publish = _capture  # type: ignore[assignment]
    res = asyncio.run(
        j._dispatch_tool(
            _call("set_hud_state", {"color": "red", "animation": "glitch"}), "intent", {}, _state()
        )
    )
    assert res.content == "hud updated"
    hud = [e for e in events if e.type == EventType.HUD_STATE]
    assert hud and hud[0].data == {"color": "red", "animation": "glitch"}


def test_read_telemetry_returns_live_reading():
    j = _make_jarvis()
    res = asyncio.run(
        j._dispatch_tool(_call("read_telemetry", {"sensor": "cpu"}), "intent", {}, _state())
    )
    assert "cpu=" in res.content


def test_execute_bash_stops_loop_on_shadow_reject():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None
    st = _state()
    with patch.object(j, "_shadow_then_real", new=AsyncMock(return_value="rejected")):
        res = asyncio.run(
            j._dispatch_tool(
                _call("execute_bash", {"command": "nmap -sV 10.0.0.1"}), "intent", {}, st
            )
        )
    assert res.stop is True
    assert "nmap -sV 10.0.0.1" in st.bash_commands


def test_execute_bash_stops_loop_on_queued_confirmation():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None
    with patch.object(j, "_shadow_then_real", new=AsyncMock(return_value="queued")):
        res = asyncio.run(
            j._dispatch_tool(
                _call("execute_bash", {"command": "apt update", "requires_sudo": True}),
                "intent", {}, _state(),
            )
        )
    assert res.stop is True


def test_unknown_tool_is_handled_gracefully():
    j = _make_jarvis()
    res = asyncio.run(j._dispatch_tool(_call("frobnicate", {}), "intent", {}, _state()))
    assert "unknown tool" in res.content


# ───────────────────────── run_skill (Skill Registry) ─────────────────────────
def _fake_skill(**kw):
    from src.skills.registry import Skill
    defaults = dict(
        id="open_files", category=IntentCategory.UI_CONTROL,
        description="открыть файлы", handler=None,
    )
    defaults.update(kw)
    return Skill(**defaults)


def test_run_skill_speaks_reply_and_executes():
    """run_skill: говорит reply модели И зовёт хендлер за один терминальный вызов."""
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append((text, tone))

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "executed"

    fake = _fake_skill(handler=handler)
    st = _state()
    with patch("src.core.orchestrator.skills.get", return_value=fake):
        res = asyncio.run(j._dispatch_tool(
            _call("run_skill", {"skill_id": "open_files", "reply": "Открываю, сэр",
                                "mood": "professional"}),
            "открой файлы", {}, st,
        ))
    assert ("Открываю, сэр", "normal") in spoken
    assert ran == [{}]
    assert st.spoke is True
    assert res.stop is True


def test_run_skill_speaks_result_for_telemetry():
    """speaks_result навык озвучивает СВОЙ результат (живые цифры), а не reply."""
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
    st = _state()
    with patch("src.core.orchestrator.skills.get", return_value=fake):
        asyncio.run(j._dispatch_tool(
            _call("run_skill", {"skill_id": "report_cpu", "reply": "сейчас гляну"}),
            "нагрузка", {}, st,
        ))
    assert any("50 процентов" in s for s in spoken)
    assert "сейчас гляну" not in spoken  # reply модели проигнорирован — данные у хендлера


def test_run_skill_destructive_queues_confirmation():
    """Разрушительный навык не исполняется сразу — уходит на подтверждение."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "ran"

    fake = _fake_skill(id="wipe", category=IntentCategory.SYSTEM_OPS,
                       handler=handler, destructive=True)
    st = _state()
    with patch("src.core.orchestrator.skills.get", return_value=fake):
        res = asyncio.run(j._dispatch_tool(
            _call("run_skill", {"skill_id": "wipe", "reply": "r"}), "intent", {}, st,
        ))
    assert res.stop is True
    assert ran == [], "разрушительный навык не должен исполниться до подтверждения"
    assert j._pending_skill is not None
    assert j._pending_skill["skill"].id == "wipe"


def test_run_skill_unknown_is_graceful():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)
    st = _state()
    with patch("src.core.orchestrator.skills.get", return_value=None):
        res = asyncio.run(j._dispatch_tool(
            _call("run_skill", {"skill_id": "nope", "reply": "ладно"}), "intent", {}, st,
        ))
    assert res.stop is True
    assert "unknown skill" in res.content
    assert spoken == ["ладно"]  # озвучили reply модели как graceful-degrade


def test_skill_confirmation_runs_handler_on_yes():
    """Подтверждение разрушительного навыка («да») переисполняет хендлер."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "done"

    fake = _fake_skill(id="wipe", category=IntentCategory.SYSTEM_OPS,
                       handler=handler, destructive=True)
    j._pending_skill = {
        "skill": fake, "args": {"a": 1}, "intent": "intent", "snap": {}, "ts": 9e18,
    }
    handled = asyncio.run(j._try_resolve_skill("да, подтверждаю"))
    assert handled is True
    assert ran == [{"a": 1}]
    assert j._pending_skill is None


def test_warmup_sends_tiny_request_with_tools():
    """Прогрев должен слать один крошечный запрос С tools (греет реальный
    префикс + компилирует Vulkan-шейдеры), max_tokens мал."""
    from src.inference.agent import ChatResponse

    j = _make_jarvis()
    captured: dict = {}

    async def fake_chat(messages, tools=None, **kw):
        captured["tools"] = tools
        captured["max_tokens"] = kw.get("max_tokens")
        return ChatResponse("", (), "stop")

    j._llm.chat = fake_chat  # type: ignore[assignment]
    assert asyncio.run(j.warmup()) is True
    assert captured["tools"], "warmup обязан слать tools"
    assert any("execute_bash" in str(t) for t in captured["tools"])
    assert captured["max_tokens"] and captured["max_tokens"] <= 8


def test_warmup_returns_false_on_server_error():
    from src.inference.openai_client import LlamaServerError

    j = _make_jarvis()

    async def boom(*a, **kw):
        raise LlamaServerError("server down")

    j._llm.chat = boom  # type: ignore[assignment]
    assert asyncio.run(j.warmup()) is False


def test_voice_intents_coalesce_to_latest():
    """Сериализация + latest-wins: пока обрабатывается команда, бурст новых
    голосовых интентов не плодит конкурентные запросы к одно-слотовому серверу —
    выполняется первый (уже стартовавший) и самый свежий; промежуточные
    вытесняются."""
    j = _make_jarvis()
    processed: list = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_process(text):
        processed.append(text)
        if text == "first":
            started.set()
            await release.wait()

    j.process_intent = slow_process  # type: ignore[assignment]

    async def scenario():
        t1 = asyncio.create_task(j.on_voice_intent(Event(EventType.VOICE_INTENT, "first")))
        await started.wait()
        t2 = asyncio.create_task(j.on_voice_intent(Event(EventType.VOICE_INTENT, "second")))
        t3 = asyncio.create_task(j.on_voice_intent(Event(EventType.VOICE_INTENT, "third")))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(t1, t2, t3)

    asyncio.run(scenario())
    assert "first" in processed
    assert "third" in processed
    assert "second" not in processed  # вытеснен более свежим
