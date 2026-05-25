from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Mapping

import chromadb
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings

from singleton import Singleton

log = logging.getLogger("jarvis.memory")


class OllamaEmbedding(EmbeddingFunction):
    def __init__(self, model: str = "all-minilm") -> None:
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        out: Embeddings = []
        for text in input:
            res = ollama.embeddings(model=self.model, prompt=text)
            out.append(res["embedding"])
        return out


def _flatten_snapshot(snap: Mapping[str, Any] | None) -> dict[str, Any]:
    if not snap:
        return {}
    return {
        "snap_window": str(snap.get("window", ""))[:128],
        "snap_cpu": float(snap.get("cpu_pct", 0.0) or 0.0),
        "snap_ram": float(snap.get("ram_pct", 0.0) or 0.0),
        "snap_load": float(snap.get("load_avg", 0.0) or 0.0),
        "snap_hour": int(snap.get("hour", 0) or 0),
    }


class ChronoMemory(metaclass=Singleton):
    def __init__(self, path: str = "./jarvis_memory", embed_model: str = "all-minilm") -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = OllamaEmbedding(embed_model)
        self.collection = self.client.get_or_create_collection(
            name="chrono", embedding_function=self.embed_fn
        )

    def remember(
        self,
        intent: str,
        command: str = "",
        result: str = "",
        kind: str = "task",
        snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        document = f"INTENT: {intent}\nCMD: {command}\nRESULT: {result}".strip()
        metadata: dict[str, Any] = {
            "ts": time.time(),
            "kind": kind,
            "intent": intent[:512],
            "command": command[:512],
        }
        metadata.update(_flatten_snapshot(snapshot))
        try:
            self.collection.add(documents=[document], metadatas=[metadata], ids=[uuid.uuid4().hex])
        except Exception:
            log.exception("ChromaDB add failed")

    def recall(self, query: str, n_results: int = 3, since_days: int = 14) -> list[dict]:
        try:
            since_ts = time.time() - since_days * 86400
            res = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"ts": {"$gte": since_ts}},
            )
        except Exception:
            log.exception("ChromaDB query failed")
            return []
        out: list[dict] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for doc, meta in zip(docs, metas):
            out.append({"document": doc, "metadata": meta or {}})
        return out
