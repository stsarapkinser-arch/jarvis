"""Three-tier persistent memory for Jarvis.

Pillar #1 of the Symbiote architecture: a ChromaDB-backed memory whose
retention scales with significance, so the database never bloats into
multi-gigabyte territory on the N100 box.

Three Chroma collections share one ``PersistentClient`` and one embedding
function:

* **Core Identity** (``chrono_core``) — never-expiring. Who Jarvis is,
  who the operator is, project paths, principles, contacts. Written via
  :meth:`remember_core` or by the auto-promoter when significance hits 2.

* **Warm Context** (``chrono_warm``) — 5-day TTL. Above-baseline events:
  intrusion alerts, critical OS events, OTPs, named entities the operator
  asked about, sessions where work was done. Written by callers passing
  ``significance=1`` to :meth:`remember`.

* **Lite Context** (``chrono_lite``) — 2-day TTL. The firehose: every
  executed command, every routine clipboard snapshot, every daemon ping.
  Default tier for :meth:`remember` (``significance=0``).

Recall queries all three tiers and returns a merged, deduplicated list
sorted core-first, then warm, then lite — within each tier newest first.

Migration: any pre-existing ``chrono`` (legacy) or ``chrono_rolling`` (v2
two-tier) collections are drained into ``chrono_lite`` on first boot."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Final

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.errors import InternalError as ChromaDBInternalError

from src.common.singleton import Singleton
from src.inference.embeddings import EmbeddingClient

log = logging.getLogger("jarvis.memory")

# ---- Retention policy ------------------------------------------------------
# Three tiers. The operator picks a tier by passing ``significance`` to
# :meth:`remember`; if they don't, the lite tier (2 days) catches it.
LITE_TTL_SEC: Final = 2 * 86_400
WARM_TTL_SEC: Final = 5 * 86_400
# Sweep cadence. Hourly is plenty — Chroma's delete-by-ids is O(n) over the
# matched set, and the matched set grows by at most a few hundred per hour
# on a normal workload.
GC_PERIOD_SEC: Final = 3600.0
# Hard ceilings per tier even if TTL hasn't expired them yet — protects
# against runaway producers (e.g. a chatty daemon stuck in a loop).
LITE_HARD_CAP: Final = 25_000
WARM_HARD_CAP: Final = 10_000
# Recall windows when the caller doesn't specify one. Each tier is already
# capped at its TTL; these constants only let recall reach less far back.
DEFAULT_RECALL_DAYS: Final = 14
# Significance levels passed to :meth:`remember`.
SIG_LITE: Final = 0    # routine — 2-day TTL
SIG_WARM: Final = 1    # above-baseline — 5-day TTL
SIG_CORE: Final = 2    # identity-level — permanent (promoted to chrono_core)
# Core identity kinds we accept without warning. Free-form kinds still work,
# they just get a log line.
CORE_KINDS: Final = frozenset({"identity", "project", "preference", "principle", "contact"})


class LlamaServerEmbedding(EmbeddingFunction):
    """Chroma EmbeddingFunction поверх ``llama-server /v1/embeddings`` (без Ollama).

    Весь батч уходит одним HTTP-запросом (см. ``EmbeddingClient``). ``name()``
    задан явно — иначе свежий Chroma предупреждает о будущем требовании к
    EmbeddingFunction."""

    def __init__(self, endpoint: str | None = None, model: str | None = None) -> None:
        self._client = EmbeddingClient(endpoint=endpoint, model=model)
        self.model = self._client.model

    def name(self) -> str:
        return "jarvis-llama-embed"

    def __call__(self, input: Documents) -> Embeddings:
        return self._client.embed(list(input))


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
    """Three-tier persistent memory. Thread-safe by way of Chroma's own
    client serialisation; the only async-native surface is :meth:`start_gc`."""

    def __init__(
        self,
        path: str = "./jarvis_memory",
        embed_endpoint: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = LlamaServerEmbedding(endpoint=embed_endpoint, model=embed_model)
        self.lite = self.client.get_or_create_collection(
            name="chrono_lite", embedding_function=self.embed_fn
        )
        self.warm = self.client.get_or_create_collection(
            name="chrono_warm", embedding_function=self.embed_fn
        )
        self.core = self.client.get_or_create_collection(
            name="chrono_core", embedding_function=self.embed_fn
        )
        # Backward-compat alias: external callers / tests that still touch
        # .rolling get the lite tier (its successor in role).
        self.rolling = self.lite
        self._migrate_legacy()
        self._gc_task: asyncio.Task | None = None
        self._gc_lock = asyncio.Lock()

    # ---- One-time migration ------------------------------------------------
    def _migrate_legacy(self) -> None:
        """Drain pre-existing ``chrono`` (v1) or ``chrono_rolling`` (v2) into
        ``chrono_lite``, then drop the obsolete collections. Safe to call on
        every boot — no-op when no legacy collection is present."""
        try:
            existing = {c.name for c in self.client.list_collections()}
        except Exception:
            log.exception("list_collections failed")
            return
        for legacy_name in ("chrono", "chrono_rolling"):
            if legacy_name not in existing or legacy_name == "chrono_lite":
                continue
            try:
                legacy = self.client.get_collection(
                    legacy_name, embedding_function=self.embed_fn
                )
                data = legacy.get()
                ids = data.get("ids") or []
                docs = data.get("documents") or []
                metas = data.get("metadatas") or []
                if ids and docs:
                    fixed_metas: list[dict] = []
                    for m in metas:
                        m = dict(m or {})
                        m["tier"] = "lite"
                        m.setdefault("ts", time.time())
                        m.setdefault("significance", SIG_LITE)
                        fixed_metas.append(m)
                    self.lite.add(ids=ids, documents=docs, metadatas=fixed_metas)
                self.client.delete_collection(legacy_name)
                log.info(
                    "migrated %d entries from %s -> chrono_lite",
                    len(ids), legacy_name,
                )
            except Exception:
                log.exception("legacy %s migration failed (non-fatal)", legacy_name)

    # ---- Lite / Warm tiers ------------------------------------------------
    def remember(
        self,
        intent: str,
        command: str = "",
        result: str = "",
        kind: str = "task",
        snapshot: Mapping[str, Any] | None = None,
        significance: int = SIG_LITE,
    ) -> None:
        """Write a context entry. The ``significance`` argument selects tier:

        * SIG_LITE (0, default) — 2-day TTL, lite firehose
        * SIG_WARM (1)         — 5-day TTL, above-baseline events
        * SIG_CORE (2)         — promoted to permanent core via :meth:`remember_core`

        Unknown significance values silently clamp to SIG_LITE."""
        if significance == SIG_CORE:
            self.remember_core(
                fact=f"{intent} — {result}".strip(" —"),
                kind=kind if kind in CORE_KINDS else "identity",
                tags=(kind,),
            )
            return

        if significance not in (SIG_LITE, SIG_WARM):
            significance = SIG_LITE

        tier_name = "warm" if significance == SIG_WARM else "lite"
        target = self.warm if significance == SIG_WARM else self.lite

        document = f"INTENT: {intent}\nCMD: {command}\nRESULT: {result}".strip()
        metadata: dict[str, Any] = {
            "ts": time.time(),
            "tier": tier_name,
            "kind": kind,
            "significance": significance,
            "intent": intent[:512],
            "command": command[:512],
        }
        metadata.update(_flatten_snapshot(snapshot))
        try:
            target.add(
                documents=[document],
                metadatas=[metadata],
                ids=[uuid.uuid4().hex],
            )
        except Exception:
            log.exception("%s.add failed", tier_name)

    # ---- Core tier ---------------------------------------------------------
    def remember_core(
        self,
        fact: str,
        kind: str = "identity",
        tags: Iterable[str] | None = None,
    ) -> str | None:
        """Write a permanent core-identity fact. Never GC'd.

        Returns the assigned id on success so the caller can later
        :meth:`forget_core` it if needed."""
        text = (fact or "").strip()
        if not text:
            return None
        if kind not in CORE_KINDS:
            log.info("remember_core: non-standard kind %r (accepted)", kind)
        meta: dict[str, Any] = {
            "ts": time.time(),
            "tier": "core",
            "kind": kind,
            "tags": ",".join(tags or ())[:256],
        }
        new_id = uuid.uuid4().hex
        try:
            self.core.add(documents=[text], metadatas=[meta], ids=[new_id])
            return new_id
        except Exception:
            log.exception("core.add failed")
            return None

    def forget_core(self, entry_id: str) -> bool:
        try:
            self.core.delete(ids=[entry_id])
            return True
        except Exception:
            log.exception("core.delete %s failed", entry_id)
            return False

    def list_core(self) -> list[dict]:
        try:
            data = self.core.get()
        except Exception:
            log.exception("core.get failed")
            return []
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [
            {"id": i, "document": d, "metadata": m or {}}
            for i, d, m in zip(ids, docs, metas, strict=False)
        ]

    # ---- Recall (queries all three tiers) ---------------------------------
    def recall(
        self,
        query: str,
        n_results: int = 3,
        since_days: int = DEFAULT_RECALL_DAYS,
    ) -> list[dict]:
        """Return up to ``n_results`` matches per tier, merged core→warm→lite.

        Core results are returned regardless of age. Warm and lite results
        are each constrained to ``since_days`` (and to their respective TTLs
        — there's nothing older than that to find)."""
        results: list[dict] = []
        now = time.time()
        results.extend(
            self._query_collection(self.core, query, n_results, where=None)
        )
        warm_cutoff = now - min(since_days * 86_400, WARM_TTL_SEC)
        results.extend(
            self._query_collection(
                self.warm, query, n_results, where={"ts": {"$gte": warm_cutoff}}
            )
        )
        lite_cutoff = now - min(since_days * 86_400, LITE_TTL_SEC)
        results.extend(
            self._query_collection(
                self.lite, query, n_results, where={"ts": {"$gte": lite_cutoff}}
            )
        )
        # Dedup by document prefix.
        seen: set[str] = set()
        unique: list[dict] = []
        for item in results:
            sig = (item.get("document") or "")[:160]
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(item)
        # Tier rank: core → warm → lite; within each, newest first.
        rank = {"core": 0, "warm": 1, "lite": 2, "rolling": 2}  # legacy alias
        unique.sort(
            key=lambda r: (
                rank.get(r["metadata"].get("tier", "lite"), 2),
                -float(r["metadata"].get("ts", 0.0)),
            )
        )
        return unique[: n_results * 3]

    def _query_collection(
        self,
        col,
        query: str,
        n_results: int,
        where: Mapping[str, Any] | None,
    ) -> list[dict]:
        try:
            count = col.count()
            if count == 0:
                return []
            kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(n_results, count)}
            if where:
                kwargs["where"] = dict(where)
            res = col.query(**kwargs)
        except ChromaDBInternalError:
            # ChromaDB бросает "Error finding id" при рассинхроне HNSW-сегмента
            # с WAL — чаще всего это провоцирует именно where-фильтр по ts.
            # Пробуем ещё раз БЕЗ фильтра (вернём свежие записи, отфильтруем
            # по ts уже в Python), чтобы не терять воспоминания целиком.
            if where:
                try:
                    res = col.query(query_texts=[query], n_results=min(n_results, col.count()))
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    out: list[dict] = []
                    for d, m in zip(docs, metas, strict=False):
                        m = m or {}
                        # Воспроизводим семантику where={"ts": {"$gte": cutoff}}.
                        cutoff = None
                        ts_cond = where.get("ts") if isinstance(where, Mapping) else None
                        if isinstance(ts_cond, Mapping):
                            cutoff = ts_cond.get("$gte")
                        if cutoff is None or float(m.get("ts", 0.0)) >= float(cutoff):
                            out.append({"document": d, "metadata": m})
                    log.warning("ChromaDB internal error on %s — fallback без where, %d записей",
                                getattr(col, "name", "?"), len(out))
                    return out
                except Exception:
                    log.warning("ChromaDB internal error on %s — пустой результат",
                                getattr(col, "name", "?"))
                    return []
            log.warning("ChromaDB internal error on %s (likely empty collection)",
                        getattr(col, "name", "?"))
            return []
        except Exception:
            log.exception("query failed on %s", getattr(col, "name", "?"))
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        return [
            {"document": d, "metadata": m or {}} for d, m in zip(docs, metas, strict=False)
        ]

    # ---- Garbage collection ------------------------------------------------
    def gc(self) -> int:
        """Drop entries older than each tier's TTL and trim above hard cap.

        Sweeps lite (2d) and warm (5d); core is permanent. Returns total
        entries deleted across both tiers."""
        deleted = 0
        deleted += self._gc_tier(self.lite, "lite", LITE_TTL_SEC, LITE_HARD_CAP)
        deleted += self._gc_tier(self.warm, "warm", WARM_TTL_SEC, WARM_HARD_CAP)
        if deleted:
            log.info("memory GC dropped %d entries (lite+warm)", deleted)
        return deleted

    def _gc_tier(self, col, name: str, ttl_sec: int, hard_cap: int) -> int:
        """Sweep one TTL'd tier. Returns deletion count."""
        deleted = 0
        cutoff = time.time() - ttl_sec
        try:
            stale = col.get(where={"ts": {"$lt": cutoff}})
            ids = stale.get("ids") or []
            if ids:
                col.delete(ids=ids)
                deleted += len(ids)
        except Exception:
            log.exception("%s TTL sweep failed", name)

        try:
            total = col.count()
            overflow = total - hard_cap
            if overflow > 0:
                everything = col.get()
                ids = everything.get("ids") or []
                metas = everything.get("metadatas") or []
                pairs = list(
                    zip(
                        ids,
                        (float((m or {}).get("ts", 0.0)) for m in metas),
                        strict=False,
                    )
                )
                pairs.sort(key=lambda p: p[1])
                victim_ids = [p[0] for p in pairs[:overflow]]
                if victim_ids:
                    col.delete(ids=victim_ids)
                    deleted += len(victim_ids)
        except Exception:
            log.exception("%s hard-cap sweep failed", name)
        return deleted

    async def start_gc(self, period_sec: float = GC_PERIOD_SEC) -> asyncio.Task:
        """Schedule the periodic GC sweep on the running event loop.

        First sweep fires immediately so a session that's been off for days
        catches up on startup; subsequent sweeps run every ``period_sec``."""
        if self._gc_task and not self._gc_task.done():
            return self._gc_task
        self._gc_task = asyncio.create_task(
            self._gc_loop(period_sec), name="memory-gc"
        )
        return self._gc_task

    async def _gc_loop(self, period_sec: float) -> None:
        log.info(
            "memory GC loop started (lite_ttl=%ds warm_ttl=%ds period=%ds "
            "lite_cap=%d warm_cap=%d)",
            LITE_TTL_SEC, WARM_TTL_SEC, int(period_sec),
            LITE_HARD_CAP, WARM_HARD_CAP,
        )
        try:
            async with self._gc_lock:
                await asyncio.to_thread(self.gc)
            while True:
                await asyncio.sleep(period_sec)
                async with self._gc_lock:
                    await asyncio.to_thread(self.gc)
        except asyncio.CancelledError:
            log.info("memory GC loop cancelled")
            raise
