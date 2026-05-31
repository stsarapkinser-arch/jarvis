"""Локальный эмбеддер через нативный llama-server (/v1/embeddings) — смерть Ollama.

Эмбеддинги памяти (ChronoMemory) больше НЕ требуют демона ollama. Их считает
второй лёгкий экземпляр ``llama-server --embedding`` с маленькой МУЛЬТИЯЗЫЧНОЙ
моделью (русский у оператора — англоцентричный MiniLM не годится). Профит на N100:

  * один стек llama.cpp на всё — ноль нового рантайма и лишних установок;
  * out-of-process — Python весов в RAM не держит (принцип фазы 1 ТЗ);
  * крошечная модель (~120M) — копейки RAM рядом с 3B-мозгом;
  * на одну точку отказа (демон ollama) меньше.

Chroma зовёт ``EmbeddingFunction`` СИНХРОННО внутри add/query, поэтому здесь
синхронный ``httpx.Client`` (не Async). Импорт httpx «мягкий» — без пакета
конструктор не падает (важно для unit-тестов), а вызов рапортует понятной
ошибкой. Размерность вектора задаёт модель сервера; код модель-агностичен.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover — окружения без httpx (CI без рантайма)
    httpx = None  # type: ignore
    _HAS_HTTPX = False

log = logging.getLogger("jarvis.embed")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


# Отдельный порт от chat-сервера (8080) — это ВТОРОЙ экземпляр llama-server,
# поднятый с --embedding и маленькой embedding-моделью.
DEFAULT_EMBED_ENDPOINT = os.getenv("JARVIS_EMBED_ENDPOINT", "http://127.0.0.1:8090")
# Имя-алиас модели (llama-server отдаёт под ним загруженную модель). Модель
# подбирает оператор при сетапе; код к конкретной модели не привязан.
DEFAULT_EMBED_MODEL = os.getenv("JARVIS_EMBED_MODEL", "multilingual-e5-small")
DEFAULT_EMBED_TIMEOUT = _env_float("JARVIS_EMBED_TIMEOUT", 60.0)


class EmbeddingServerError(RuntimeError):
    """Сервер эмбеддингов недоступен или вернул ошибку."""


class EmbeddingClient:
    """Синхронный клиент к ``llama-server /v1/embeddings`` (OpenAI-формат).

    Весь батч текстов уходит ОДНИМ HTTP-запросом (сервер принимает массив
    ``input``). ``httpx.Client`` создаётся лениво и переиспользует keep-alive —
    на localhost это почти нулевой overhead."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_EMBED_ENDPOINT).rstrip("/")
        self.model = model or DEFAULT_EMBED_MODEL
        self._timeout = timeout if timeout is not None else DEFAULT_EMBED_TIMEOUT
        self._connect_timeout = connect_timeout
        self._client: Any | None = None  # httpx.Client | None

    def _ensure_client(self) -> Any:
        # Уже есть (в т.ч. пред-инъекция в тестах) — отдаём, не требуя httpx.
        if self._client is not None:
            return self._client
        if not _HAS_HTTPX:
            raise EmbeddingServerError("httpx не установлен — pip install httpx")
        timeout = httpx.Timeout(self._timeout, connect=self._connect_timeout)
        self._client = httpx.Client(base_url=self.endpoint, timeout=timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def health(self) -> bool:
        """Жив ли embed-сервер. Никогда не бросает."""
        if not _HAS_HTTPX:
            return False
        try:
            client = self._ensure_client()
            resp = client.get("/health", timeout=self._connect_timeout)
            if resp.status_code != 200:
                return False
            return str(resp.json().get("status", "")).lower() in ("ok", "no slot available")
        except Exception as exc:
            log.debug("embed health failed: %s", exc)
            return False

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Векторы для списка текстов (порядок входа сохраняется)."""
        items = [str(t) for t in texts]
        if not items:
            return []
        client = self._ensure_client()
        body = {"model": self.model, "input": items}
        try:
            resp = client.post("/v1/embeddings", json=body)
        except Exception as exc:  # ConnectError/ReadTimeout/...
            detail = str(exc) or "нет ответа"
            raise EmbeddingServerError(
                f"embed-сервер недоступен: {type(exc).__name__}: {detail}"
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingServerError(
                f"embed-сервер вернул {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise EmbeddingServerError(f"невалидный JSON от embed-сервера: {exc}") from exc
        return _parse_embeddings(data, len(items))


def _parse_embeddings(data: dict[str, Any], expected: int) -> list[list[float]]:
    """Разобрать OpenAI-ответ /v1/embeddings → векторы в порядке index."""
    rows = data.get("data")
    if not isinstance(rows, list):
        raise EmbeddingServerError("ответ embed-сервера без поля data[]")
    by_index: dict[int, list[float]] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        vec = row.get("embedding")
        if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
            raise EmbeddingServerError("элемент data[] без embedding-вектора")
        try:
            idx = int(row.get("index", i))
        except (TypeError, ValueError):
            idx = i
        by_index[idx] = [float(x) for x in vec]
    out = [by_index[i] for i in sorted(by_index)]
    if len(out) != expected:
        raise EmbeddingServerError(
            f"сервер вернул {len(out)} векторов вместо {expected}"
        )
    return out


__all__ = [
    "EmbeddingClient",
    "EmbeddingServerError",
    "DEFAULT_EMBED_ENDPOINT",
    "DEFAULT_EMBED_MODEL",
]
