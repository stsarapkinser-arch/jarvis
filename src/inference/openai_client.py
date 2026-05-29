"""Async HTTP client to the native ``llama-server`` (OpenAI-compatible REST).

Фаза 1+4 ТЗ: Python больше НЕ грузит веса модели. Инференс вынесен в отдельный
системный процесс (бинарь ``llama-server`` из llama.cpp, скомпилированный с
Vulkan, см. ``setup_server.sh`` / ``config/jarvis-llm.service``). Здесь — тонкий
асинхронный клиент поверх ``httpx`` к ``/v1/chat/completions``.

Почему это быстро на N100:
  * сервер держит системный промпт в KV-кэше (``--cache-reuse``) — нет
    пересчёта 100-словного микро-промпта на каждый запрос;
  * запрос неблокирующий (httpx.AsyncClient) — EventBus, HUD и голос не
    замирают, пока модель думает.

Импорт ``httpx`` намеренно «мягкий» (как llama_cpp/psutil в проекте): отсутствие
пакета не валит импорт оркестратора — клиент просто рапортует unhealthy, а
конструктор не делает сетевых side-effect'ов (важно для unit-тестов).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - окружения без httpx (CI без рантайма)
    httpx = None  # type: ignore
    _HAS_HTTPX = False

from .agent import ChatResponse, ToolCall
from .tools import parse_arguments

log = logging.getLogger("jarvis.llm_http")

DEFAULT_ENDPOINT = os.getenv("JARVIS_LLM_ENDPOINT", "http://127.0.0.1:8080")
DEFAULT_MODEL = os.getenv("JARVIS_LLM_MODEL", "llama-3.2-3b-instruct")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


# Потолок ожидания ответа. На N100 декод ~1 т/с — даже ограниченный ответ
# (max_tokens) может занять до ~2.5 мин в худшем случае, поэтому запас 180с
# (это не «костыль-чтобы-не-падало», а соответствие реальному bounded-времени:
# генерация ограничена max_tokens, висеть бесконечно не может). Оператор может
# подстроить через JARVIS_LLM_TIMEOUT.
DEFAULT_REQUEST_TIMEOUT = _env_float("JARVIS_LLM_TIMEOUT", 180.0)


class LlamaServerError(RuntimeError):
    """Сервер инференса недоступен или вернул ошибку. Оркестратор ловит это и
    озвучивает оператору сбой мозга вместо немого ухода в IDLE."""


class LlamaServerClient:
    """Тонкий async-клиент к llama-server (OpenAI Tool-Use API).

    httpx.AsyncClient создаётся лениво при первом запросе и переиспользует
    keep-alive соединение — на localhost это почти нулевой overhead на запрос."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        request_timeout: float | None = None,
        connect_timeout: float = 5.0,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self._timeout = request_timeout if request_timeout is not None else DEFAULT_REQUEST_TIMEOUT
        self._connect_timeout = connect_timeout
        self._client: Any | None = None  # httpx.AsyncClient | None

    # ───────────────────────── lifecycle ─────────────────────────
    def _ensure_client(self) -> Any:
        if not _HAS_HTTPX:
            raise LlamaServerError("httpx не установлен — pip install httpx")
        if self._client is None:
            timeout = httpx.Timeout(self._timeout, connect=self._connect_timeout)
            self._client = httpx.AsyncClient(base_url=self.endpoint, timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ───────────────────────── health ─────────────────────────
    async def health(self) -> bool:
        """Жив ли сервер и загружена ли модель. Никогда не бросает."""
        if not _HAS_HTTPX:
            return False
        try:
            client = self._ensure_client()
            resp = await client.get("/health", timeout=self._connect_timeout)
            if resp.status_code != 200:
                return False
            data = resp.json()
            # llama.cpp: {"status":"ok"}; во время загрузки — 503/"loading model"
            return str(data.get("status", "")).lower() in ("ok", "no slot available")
        except Exception as exc:
            log.debug("health check failed: %s", exc)
            return False

    # ───────────────────────── chat completion ─────────────────────────
    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResponse:
        """Один раунд /v1/chat/completions (без стриминга — детерминизм tool-use).

        Возвращает ChatResponse с распарсенными tool_calls. На сетевой/HTTP сбой
        бросает LlamaServerError."""
        client = self._ensure_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = tool_choice

        try:
            resp = await client.post("/v1/chat/completions", json=body)
        except Exception as exc:  # httpx.ConnectError/ReadTimeout/...
            # У httpx.ReadTimeout пустой str(), поэтому добавляем имя типа —
            # иначе в логе было голое «llama-server недоступен: ».
            detail = str(exc) or "превышен таймаут ответа (модель слишком долго думает?)"
            raise LlamaServerError(
                f"llama-server недоступен: {type(exc).__name__}: {detail}"
            ) from exc

        if resp.status_code != 200:
            raise LlamaServerError(
                f"llama-server вернул {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise LlamaServerError(f"невалидный JSON от сервера: {exc}") from exc

        return _parse_completion(data)


def _parse_completion(data: dict[str, Any]) -> ChatResponse:
    """Разобрать OpenAI-ответ в ChatResponse. Терпим к отсутствию полей."""
    choices = data.get("choices") or []
    if not choices:
        return ChatResponse(content="", tool_calls=(), finish_reason="empty", raw=data)

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    finish_reason = str(choice.get("finish_reason") or "")

    tool_calls: list[ToolCall] = []
    for i, raw_call in enumerate(message.get("tool_calls") or []):
        fn = raw_call.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        call_id = str(raw_call.get("id") or f"call_{i}")
        tool_calls.append(
            ToolCall(id=call_id, name=name, arguments=parse_arguments(fn.get("arguments")))
        )

    return ChatResponse(
        content=str(content),
        tool_calls=tuple(tool_calls),
        finish_reason=finish_reason,
        raw=data,
    )


__all__ = [
    "LlamaServerClient",
    "LlamaServerError",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
]
