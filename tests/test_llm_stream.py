"""Тесты SSE-стриминга LlamaServerClient.chat(stream=True).

Зачем стрим: read-timeout httpx считается между чанками, а не на весь ответ —
медленная-но-живая 3B на N100 перестаёт рваться по «all-or-nothing» 180с. Клиент
обязан пересобрать ТОТ ЖЕ ChatResponse (контент + tool_calls из дельт), а контент
прокинуть в on_content по кускам (стык для пофразной озвучки). Сеть не нужна —
мокаем httpx.AsyncClient.stream фейком, отдающим заранее заданные SSE-строки."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.openai_client import LlamaServerClient, LlamaServerError


class _FakeStreamResp:
    def __init__(self, lines: list[str], status: int = 200):
        self._lines = lines
        self.status_code = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"boom"


class _FakeHttpx:
    """Имитирует httpx.AsyncClient.stream(...) — отдаёт заготовленные SSE-строки."""

    def __init__(self, lines: list[str], status: int = 200):
        self._lines = lines
        self._status = status
        self.stream_calls = 0
        self.last_body: dict | None = None

    def stream(self, _method, _url, json=None):
        self.stream_calls += 1
        self.last_body = json
        return _FakeStreamResp(self._lines, self._status)


def _client_with(fake) -> LlamaServerClient:
    c = LlamaServerClient(endpoint="http://x")
    c._client = fake
    return c


def _run(coro):
    return asyncio.run(coro)


def test_stream_reassembles_content_and_invokes_callback():
    lines = [
        'data: {"choices":[{"delta":{"content":"Привет"}}]}',
        "",  # пустые строки SSE игнорируем
        'data: {"choices":[{"delta":{"content":", сэр"}}]}',
        'data: {"choices":[{"delta":{"content":"."},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    fake = _FakeHttpx(lines)
    client = _client_with(fake)
    pieces: list[str] = []
    resp = _run(client.chat(
        [{"role": "user", "content": "hi"}], stream=True, on_content=pieces.append,
    ))
    assert resp.content == "Привет, сэр."
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == ()
    assert pieces == ["Привет", ", сэр", "."]          # отдан по кускам, по порядку
    assert fake.last_body and fake.last_body.get("stream") is True


def test_stream_reassembles_fragmented_tool_call():
    # Имя в первом фрагменте, JSON-аргументы — по кускам (как у реального сервера).
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"open_app","arguments":"{\\"app\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"\\"browser\\"}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    client = _client_with(_FakeHttpx(lines))
    resp = _run(client.chat(
        [{"role": "user", "content": "открой браузер"}],
        tools=[{"type": "function", "function": {"name": "open_app"}}],
        tool_choice="required", stream=True,
    ))
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "open_app"
    assert call.arguments == {"app": "browser"}
    assert resp.finish_reason == "tool_calls"


def test_stream_non_200_raises():
    client = _client_with(_FakeHttpx([], status=503))
    try:
        _run(client.chat([{"role": "user", "content": "hi"}], stream=True))
        assert False, "ожидался LlamaServerError"
    except LlamaServerError as e:
        assert "503" in str(e)


def test_stream_skips_garbage_lines():
    # Битый JSON и не-data строки не валят стрим — просто пропускаются.
    lines = [
        ": keep-alive comment",
        "data: not-json",
        'data: {"choices":[{"delta":{"content":"ок"}}]}',
        "data: [DONE]",
    ]
    client = _client_with(_FakeHttpx(lines))
    resp = _run(client.chat([{"role": "user", "content": "hi"}], stream=True))
    assert resp.content == "ок"
