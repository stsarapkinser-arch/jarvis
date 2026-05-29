"""Agentic tool-calling loop — обобщённый цикл общения с llama-server.

Транспорт-агностичный движок Function Calling: получает массив сообщений и
инструментов, гоняет ``client.chat()``, исполняет вернувшиеся tool-вызовы через
переданный ``dispatch`` и скармливает результаты обратно модели — пока та не
перестанет звать инструменты или не упрётся в лимит шагов.

Никакого знания о Jarvis здесь нет: ``dispatch`` — это замыкание оркестратора,
которое и связывает абстрактные ToolCall с реальными подсистемами (TTS, HUD,
ShadowExec). Так слой инференса остаётся переиспользуемым и тестируемым.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("jarvis.agent")

Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Один вызов инструмента, распарсенный из ответа модели."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Результат исполнения инструмента, который уходит обратно в модель.

    ``stop=True`` останавливает цикл после фидбэка (например, execute_bash ушёл
    на голосовое подтверждение — продолжать диалог в этом интенте незачем)."""
    tool_call_id: str
    content: str
    stop: bool = False


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Разобранный ответ /v1/chat/completions."""
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRun:
    """Итог прогона агентного цикла."""
    final_content: str
    steps: int
    tool_calls_made: int
    stopped: str                      # "no_tools" | "tool_stop" | "max_steps"
    messages: list[Message] = field(default_factory=list)


class ChatClient(Protocol):
    """Минимальный контракт клиента, который нужен циклу (duck-typing)."""
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = ...,
        tool_choice: str = ...,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> ChatResponse: ...


Dispatcher = Callable[[ToolCall], Awaitable[ToolResult | None]]

DEFAULT_MAX_STEPS = 4


def _assistant_tool_call_message(resp: ChatResponse) -> Message:
    """Эхо assistant-сообщения с tool_calls — обязательно для протокола,
    иначе следующий запрос с role:tool вернёт 400 от сервера."""
    return {
        "role": "assistant",
        "content": resp.content or None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for c in resp.tool_calls
        ],
    }


async def run_agent(
    client: ChatClient,
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]],
    dispatch: Dispatcher,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> AgentRun:
    """Прогнать агентный цикл. Исключения транспорта пробрасываются наружу
    (оркестратор их ловит и озвучивает сбой)."""
    convo: list[Message] = list(messages)
    tool_calls_made = 0

    for step in range(max_steps):
        resp = await client.chat(
            convo,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not resp.tool_calls:
            # Модель ответила обычным текстом (не должна, но бывает) —
            # возвращаем его, оркестратор решит, что делать.
            return AgentRun(
                final_content=resp.content or "",
                steps=step + 1,
                tool_calls_made=tool_calls_made,
                stopped="no_tools",
                messages=convo,
            )

        convo.append(_assistant_tool_call_message(resp))
        stop = False
        for call in resp.tool_calls:
            tool_calls_made += 1
            try:
                result = await dispatch(call)
            except Exception:
                log.exception("dispatch failed for tool %s", call.name)
                result = ToolResult(call.id, "error: tool execution failed")
            if result is None:
                result = ToolResult(call.id, "ok")
            convo.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })
            stop = stop or result.stop

        if stop:
            return AgentRun(
                final_content="",
                steps=step + 1,
                tool_calls_made=tool_calls_made,
                stopped="tool_stop",
                messages=convo,
            )

    log.info("agent loop hit max_steps=%d", max_steps)
    return AgentRun(
        final_content="",
        steps=max_steps,
        tool_calls_made=tool_calls_made,
        stopped="max_steps",
        messages=convo,
    )


__all__ = [
    "Message",
    "ToolCall",
    "ToolResult",
    "ChatResponse",
    "AgentRun",
    "ChatClient",
    "Dispatcher",
    "run_agent",
    "DEFAULT_MAX_STEPS",
]
