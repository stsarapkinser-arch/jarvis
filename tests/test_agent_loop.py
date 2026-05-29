"""Тесты обобщённого агентного цикла (Phase 4) и парсинга OpenAI-ответа.

Клиент мокается фейком — никакой сети/httpx не требуется."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.agent import ChatResponse, ToolCall, ToolResult, run_agent
from src.inference.openai_client import _parse_completion


class FakeClient:
    """Отдаёт заранее заготовленные ChatResponse по очереди; пишет историю
    переданных messages в self.calls для проверки фидбэка."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list] = []

    async def chat(self, messages, tools=None, tool_choice="auto",
                   temperature=0.3, max_tokens=512):
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _tc(name, args, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=args)


def _run(coro):
    return asyncio.run(coro)


def test_agent_dispatches_then_stops_on_no_tools():
    responses = [
        ChatResponse("", (_tc("speak_response", {"text": "привет"}),), "tool_calls"),
        ChatResponse("", (), "stop"),
    ]
    client = FakeClient(responses)
    seen: list[str] = []

    async def dispatch(call):
        seen.append(call.name)
        return ToolResult(call.id, "ok")

    run = _run(run_agent(client, [{"role": "user", "content": "hi"}], [], dispatch, max_steps=4))
    assert seen == ["speak_response"]
    assert run.stopped == "no_tools"
    assert run.tool_calls_made == 1
    roles = [m["role"] for m in run.messages]
    assert "assistant" in roles and "tool" in roles


def test_agent_feeds_tool_result_back_to_model():
    responses = [
        ChatResponse("", (_tc("read_telemetry", {"sensor": "cpu"}),), "tool_calls"),
        ChatResponse("", (), "stop"),
    ]
    client = FakeClient(responses)

    async def dispatch(call):
        return ToolResult(call.id, "cpu=42% thermal=55C")

    _run(run_agent(client, [{"role": "user", "content": "temp?"}], [], dispatch, max_steps=4))
    # второй запрос к модели обязан содержать tool-результат
    second = client.calls[1]
    assert any(m.get("role") == "tool" and "cpu=42%" in m.get("content", "") for m in second)


def test_agent_stop_flag_halts_loop():
    responses = [
        ChatResponse("", (_tc("execute_bash", {"command": "rm -rf /"}),), "tool_calls"),
        ChatResponse("недостижимо", (), "stop"),
    ]
    client = FakeClient(responses)

    async def dispatch(call):
        return ToolResult(call.id, "awaiting confirmation", stop=True)

    run = _run(run_agent(client, [{"role": "user", "content": "x"}], [], dispatch, max_steps=4))
    assert run.stopped == "tool_stop"
    assert len(client.calls) == 1  # модель не зовём повторно


def test_agent_max_steps_cap():
    loop_resp = [
        ChatResponse("", (_tc("internal_monologue", {"thought": "loop"}),), "tool_calls")
        for _ in range(10)
    ]
    client = FakeClient(loop_resp)

    async def dispatch(call):
        return ToolResult(call.id, "ok")

    run = _run(run_agent(client, [{"role": "user", "content": "x"}], [], dispatch, max_steps=3))
    assert run.stopped == "max_steps"
    assert len(client.calls) == 3


def test_dispatch_exception_becomes_error_result():
    responses = [
        ChatResponse("", (_tc("execute_bash", {"command": "x"}),), "tool_calls"),
        ChatResponse("", (), "stop"),
    ]
    client = FakeClient(responses)

    async def dispatch(call):
        raise RuntimeError("boom")

    run = _run(run_agent(client, [{"role": "user", "content": "x"}], [], dispatch, max_steps=4))
    assert run.stopped == "no_tools"
    assert any(
        m.get("role") == "tool" and "error" in m.get("content", "")
        for m in run.messages
    )


def test_multiple_parallel_tool_calls_in_one_step():
    responses = [
        ChatResponse("", (
            _tc("set_hud_state", {"color": "cyan", "animation": "pulse"}, "a"),
            _tc("speak_response", {"text": "готово"}, "b"),
        ), "tool_calls"),
        ChatResponse("", (), "stop"),
    ]
    client = FakeClient(responses)
    seen: list[str] = []

    async def dispatch(call):
        seen.append(call.name)
        return ToolResult(call.id, "ok")

    run = _run(run_agent(client, [{"role": "user", "content": "x"}], [], dispatch, max_steps=4))
    assert seen == ["set_hud_state", "speak_response"]
    assert run.tool_calls_made == 2


# ─────────── openai_client._parse_completion (pure, без httpx) ───────────
def test_parse_completion_tool_calls():
    data = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None,
        "tool_calls": [{
            "id": "x1", "type": "function",
            "function": {"name": "speak_response", "arguments": '{"text":"hi","mood":"alert"}'},
        }],
    }}]}
    resp = _parse_completion(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "speak_response"
    assert resp.tool_calls[0].arguments == {"text": "hi", "mood": "alert"}


def test_parse_completion_plain_text():
    data = {"choices": [{"finish_reason": "stop", "message": {"content": "привет, сэр"}}]}
    resp = _parse_completion(data)
    assert resp.content == "привет, сэр"
    assert resp.tool_calls == ()


def test_parse_completion_handles_dict_arguments_and_missing_id():
    data = {"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"type": "function", "function": {
            "name": "read_telemetry", "arguments": {"sensor": "ram"},
        }}],
    }}]}
    resp = _parse_completion(data)
    assert resp.tool_calls[0].name == "read_telemetry"
    assert resp.tool_calls[0].arguments == {"sensor": "ram"}
    assert resp.tool_calls[0].id  # сгенерирован фолбэк-id


def test_parse_completion_empty_choices():
    resp = _parse_completion({"choices": []})
    assert resp.content == "" and resp.tool_calls == ()
