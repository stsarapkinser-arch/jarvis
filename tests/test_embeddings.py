"""Тесты локального эмбеддера (llama-server /v1/embeddings) — замена Ollama.

Без сети: подсовываем фейковый HTTP-клиент. Проверяем батч-запрос, разбор
OpenAI-ответа (включая переупорядочивание по index), ошибки сервера и
Chroma-обёртку LlamaServerEmbedding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.embeddings import (
    EmbeddingClient,
    EmbeddingServerError,
    _parse_embeddings,
)


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _FakeHttp:
    """Фейковый httpx.Client: пишет последний body, отдаёт заготовленный ответ."""

    def __init__(self, resp):
        self._resp = resp
        self.last_json = None
        self.posted_url = None

    def post(self, url, json=None):
        self.posted_url = url
        self.last_json = json
        return self._resp


def _client_with(resp):
    c = EmbeddingClient(endpoint="http://x", model="test-model")
    c._client = _FakeHttp(resp)  # пред-инъекция — httpx не нужен
    return c


# ───────────────────────── _parse_embeddings ─────────────────────────
def test_parse_preserves_index_order():
    data = {"data": [
        {"index": 1, "embedding": [0.1, 0.2]},
        {"index": 0, "embedding": [0.9, 0.8]},
    ]}
    out = _parse_embeddings(data, expected=2)
    assert out == [[0.9, 0.8], [0.1, 0.2]]  # отсортировано по index


def test_parse_rejects_count_mismatch():
    data = {"data": [{"index": 0, "embedding": [0.1]}]}
    with pytest.raises(EmbeddingServerError):
        _parse_embeddings(data, expected=2)


def test_parse_rejects_missing_data():
    with pytest.raises(EmbeddingServerError):
        _parse_embeddings({"oops": 1}, expected=1)


def test_parse_rejects_non_vector():
    data = {"data": [{"index": 0, "embedding": "not-a-list"}]}
    with pytest.raises(EmbeddingServerError):
        _parse_embeddings(data, expected=1)


# ───────────────────────── EmbeddingClient.embed ─────────────────────────
def test_embed_batches_all_texts_in_one_request():
    resp = _FakeResp(payload={"data": [
        {"index": 0, "embedding": [1.0, 1.0]},
        {"index": 1, "embedding": [2.0, 2.0]},
    ]})
    c = _client_with(resp)
    out = c.embed(["раз", "два"])
    assert out == [[1.0, 1.0], [2.0, 2.0]]
    # один запрос, весь батч в input, модель проброшена.
    assert c._client.posted_url == "/v1/embeddings"
    assert c._client.last_json == {"model": "test-model", "input": ["раз", "два"]}


def test_embed_empty_returns_empty_without_request():
    c = _client_with(_FakeResp(payload={"data": []}))
    assert c.embed([]) == []
    assert c._client.posted_url is None  # запрос не отправлялся


def test_embed_raises_on_http_error():
    c = _client_with(_FakeResp(status_code=503, text="loading model"))
    with pytest.raises(EmbeddingServerError):
        c.embed(["x"])


def test_embed_wraps_transport_exception():
    class _Boom:
        def post(self, *a, **kw):
            raise ConnectionError("refused")

    c = EmbeddingClient()
    c._client = _Boom()
    with pytest.raises(EmbeddingServerError):
        c.embed(["x"])


# ───────────────────────── Chroma-обёртка ─────────────────────────
def test_llama_server_embedding_delegates_and_has_name():
    from src.memory.engine import LlamaServerEmbedding
    ef = LlamaServerEmbedding(endpoint="http://x", model="m")
    ef._client._client = _FakeHttp(_FakeResp(payload={"data": [
        {"index": 0, "embedding": [0.5, 0.5, 0.5]},
    ]}))
    out = ef(["документ"])
    # chromadb оборачивает __call__ и нормализует вывод в numpy — сравниваем
    # поэлементно после приведения к float.
    rows = [[float(x) for x in vec] for vec in list(out)]
    assert rows == [[0.5, 0.5, 0.5]]
    assert ef.name() == "jarvis-llama-embed"
