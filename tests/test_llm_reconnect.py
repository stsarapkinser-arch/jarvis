"""Тест устойчивости LlamaServerClient к рестарту сервера посреди запроса.

llama-server — отдельный systemd-юнит с Restart=always: краш бэкенда убивает
демон, systemd поднимает его за ~2с. Клиент должен пережить ОДИН дисконнект
(дождаться /health и повторить), а не сразу рапортовать «мозг недоступен».
Сеть/httpx не нужны — мокаем POST/health фейком."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import openai_client as oc
from src.inference.openai_client import LlamaServerClient, LlamaServerError


class _Resp:
    status_code = 200

    def json(self):
        return {
            "choices": [
                {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
            ]
        }


class _FakeHttpx:
    """Имитирует httpx.AsyncClient: первый POST роняет соединение, дальше — 200.
    health() начинает отвечать True только после «рестарта»."""

    def __init__(self, fail_times: int):
        self._left = fail_times
        self.posts = 0

    async def post(self, _url, json=None):
        self.posts += 1
        if self._left > 0:
            self._left -= 1
            raise _RemoteProtocolError("Server disconnected without sending a response.")
        return _Resp()


class _RemoteProtocolError(Exception):
    pass


# Имя класса должно совпасть с тем, что клиент считает «дисконнектом».
_RemoteProtocolError.__name__ = "RemoteProtocolError"


def _client_with(fake, healthy_after_calls=0):
    c = LlamaServerClient(endpoint="http://x")
    c._client = fake
    # health: True после исчерпания «лага рестарта».
    state = {"n": 0}

    async def fake_health():
        state["n"] += 1
        return state["n"] > healthy_after_calls

    c.health = fake_health  # type: ignore
    return c


def _run(coro):
    return asyncio.run(coro)


def test_retries_once_after_disconnect(monkeypatch):
    monkeypatch.setattr(oc, "RECONNECT_WAIT", 5.0)
    fake = _FakeHttpx(fail_times=1)
    client = _client_with(fake, healthy_after_calls=0)
    resp = _run(client.chat([{"role": "user", "content": "hi"}]))
    assert resp.content == "ok"
    assert fake.posts == 2  # упал раз, повторился и прошёл


def test_gives_up_after_window(monkeypatch):
    monkeypatch.setattr(oc, "RECONNECT_WAIT", 2.0)
    fake = _FakeHttpx(fail_times=99)  # сервер так и не вернулся
    client = _client_with(fake, healthy_after_calls=0)
    try:
        _run(client.chat([{"role": "user", "content": "hi"}]))
        assert False, "ожидался LlamaServerError"
    except LlamaServerError as e:
        assert "не вернулся" in str(e)


def test_no_retry_when_disabled(monkeypatch):
    monkeypatch.setattr(oc, "RECONNECT_WAIT", 0.0)  # ретрай выключен
    fake = _FakeHttpx(fail_times=1)
    client = _client_with(fake)
    try:
        _run(client.chat([{"role": "user", "content": "hi"}]))
        assert False, "ожидался LlamaServerError"
    except LlamaServerError:
        assert fake.posts == 1  # повтора не было
