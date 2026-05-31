"""Тесты L2 — семантический матч навыка через эмбеддинги.

Эмбеддинги в песочнице не считаем (нет embed-сервера) — инъектируем
ДЕТЕРМИНИРОВАННЫЙ фейк-эмбеддер с заранее заданными векторами. Это проверяет
ЛОГИКУ матчера и харнесса (порог/зазор/группировка/исключения/калибровка), а
не качество модели. Плюс инвариант: все навыки golden-набора реально
индексируемы. Плюс интеграция и gating в оркестраторе.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.singleton import Singleton
from src.inference.semantic import (
    SemanticSkillMatcher,
    evaluate,
    indexable_skill_ids,
)
from src.inference.router import IntentCategory


def _embed_from(vmap: dict[str, list[float]], dim: int = 2):
    """Фейк-эмбеддер: известная фраза → заданный вектор, иначе нулевой (cos=-1)."""
    def embed(texts):
        return [list(vmap.get(t, [0.0] * dim)) for t in texts]
    return embed


# ───────────────────────── 1. Механика матчера ─────────────────────────
def test_match_clear_winner():
    ex = {"term": ["t-alias"], "brow": ["b-alias"]}
    embed = _embed_from({"t-alias": [1, 0], "b-alias": [0, 1], "q": [0.95, 0.05]})
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.8, margin=0.1)
    assert m.match("q") == "term"


def test_match_below_threshold_none():
    ex = {"term": ["t-alias"]}
    embed = _embed_from({"t-alias": [1, 0], "q": [1, 1]})  # cos ~0.707
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.8, margin=0.1)
    assert m.match("q") is None


def test_match_ambiguous_margin_none():
    ex = {"a": ["x"], "b": ["y"]}
    embed = _embed_from({"x": [1, 0], "y": [1, 0], "q": [1, 0]})  # обе цели cos=1.0
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.5, margin=0.1)
    assert m.match("q") is None


def test_match_same_skill_multiple_aliases_ok():
    ex = {"a": ["x1", "x2"]}
    embed = _embed_from({"x1": [1, 0], "x2": [0, 1], "q": [0.9, 0.1]})
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.8, margin=0.1)
    assert m.match("q") == "a"


def test_rank_returns_per_skill_sorted():
    ex = {"a": ["x"], "b": ["y"]}
    embed = _embed_from({"x": [1, 0], "y": [0, 1], "q": [0.9, 0.4]})
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.0, margin=0.0)
    ranked = m.rank("q")
    assert [sid for sid, _ in ranked] == ["a", "b"]


# ───────────────────────── 2. Graceful degradation ─────────────────────────
def test_build_failure_disables_quietly():
    def embed(texts):
        raise RuntimeError("embed server down")
    m = SemanticSkillMatcher(embed, exemplars={"a": ["x"]})
    assert m.match("любой запрос") is None  # не бросает, тихо None


def test_query_embed_failure_returns_none():
    ex = {"a": ["x"]}

    def embed(texts):
        # экземпляры эмбеддятся (build ок), а запрос — падает.
        if list(texts) == ["x"]:
            return [[1.0, 0.0]]
        raise RuntimeError("transient")
    m = SemanticSkillMatcher(embed, exemplars=ex, threshold=0.5, margin=0.1)
    assert m.match("q") is None


# ───────────────────────── 3. Индекс: исключения ─────────────────────────
def test_indexable_excludes_destructive_and_param_skills():
    idx = indexable_skill_ids()
    assert "log_out" not in idx, "разрушительный навык не должен быть в L2-индексе"
    assert "set_volume" not in idx, "параметрический навык (слот) не в L2-индексе"
    assert "nmap_quick" not in idx
    assert "open_terminal" in idx, "безаргументный навык с алиасом должен быть в индексе"
    assert "report_cpu" in idx


# ───────────────────────── 4. Харнесс калибровки (синтетика) ─────────────────────────
def test_evaluate_counts_false_accepts_and_recommends_threshold():
    ex = {"term": ["t"], "brow": ["b"]}
    embed = _embed_from({
        "t": [1, 0], "b": [0, 1],
        "qt": [1, 0.1], "qb": [0.1, 1],   # позитивы — близко к своим целям
        "qn": [1, 0.6],                    # негатив — но довольно близко к term
    })
    positives = [("qt", "term"), ("qb", "brow")]
    negatives = ["qn"]
    res = evaluate(embed, positives, negatives, thresholds=[0.80, 0.90], margin=0.1, exemplars=ex)
    low, high = res[0], res[1]
    # низкий порог ловит негатив (ложный приём), высокий — нет.
    assert low.false_accept == 1 and not low.clean
    assert high.false_accept == 0 and high.clean
    assert high.recall == 1.0


# ───────────────────────── 5. Инвариант golden-набора ─────────────────────────
def test_dataset_positives_are_all_indexable():
    from tests.semantic_dataset import POSITIVES
    idx = indexable_skill_ids()
    missing = sorted({sid for _, sid in POSITIVES} - idx)
    assert not missing, f"в датасете навыки вне L2-индекса: {missing}"


# ───────────────────────── 6. Интеграция в оркестратор ─────────────────────────
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


def _fake_skill(**kw):
    from src.skills.registry import Skill
    defaults = dict(
        id="open_terminal", category=IntentCategory.UI_CONTROL,
        description="открыть терминал", handler=None,
    )
    defaults.update(kw)
    return Skill(**defaults)


class _FakeMatcher:
    def __init__(self, sid):
        self._sid = sid

    def match(self, text):
        return self._sid


def test_semantic_fastpath_disabled_is_noop():
    j = _make_jarvis()
    j._semantic = None
    handled = asyncio.run(j._try_semantic_fastpath("запусти терминал"))
    assert handled is False


def test_semantic_fastpath_invokes_skill():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "ok"

    j._semantic = _FakeMatcher("open_terminal")
    fake = _fake_skill(handler=handler)
    with patch("src.core.orchestrator.skills.get", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_semantic_fastpath("запусти терминал для работы"))
    assert handled is True
    assert ran == [{}]


def test_semantic_fastpath_refuses_destructive_or_param_skill():
    """Страховка: даже если матчер вернул опасный id, навык не запускается."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "boom"

    j._semantic = _FakeMatcher("dangerous")
    fake = _fake_skill(id="dangerous", destructive=True, handler=handler)
    with patch("src.core.orchestrator.skills.get", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_semantic_fastpath("снеси всё"))
    assert handled is False
    assert ran == []


def test_process_intent_semantic_skips_agent():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "ok"

    j._semantic = _FakeMatcher("open_terminal")
    fake = _fake_skill(handler=handler)

    async def fake_agent(*a, **kw):
        raise AssertionError("агент не должен вызываться при semantic fast-path")

    j._run_agent_for_intent = fake_agent  # type: ignore[assignment]
    with patch("src.core.orchestrator.skills.match_alias", return_value=None), \
         patch("src.core.orchestrator.macros.match_macro", return_value=None), \
         patch("src.core.orchestrator.patterns.match", return_value=None), \
         patch("src.core.orchestrator.macros.fuzzy_match_macro", return_value=None), \
         patch("src.core.orchestrator.skills.fuzzy_match_alias", return_value=None), \
         patch("src.core.orchestrator.skills.get", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        res = asyncio.run(j.process_intent("запусти терминал для работы"))
    assert res == "[semantic_fastpath]"


# ───────────────────────── 7. Gating через env ─────────────────────────
def test_init_semantic_matcher_off_by_default(monkeypatch):
    from src.core.orchestrator import Jarvis
    monkeypatch.delenv("JARVIS_SEMANTIC_MATCH", raising=False)
    assert Jarvis._init_semantic_matcher() is None


def test_init_semantic_matcher_on_with_flag(monkeypatch):
    from src.core.orchestrator import Jarvis
    monkeypatch.setenv("JARVIS_SEMANTIC_MATCH", "1")
    m = Jarvis._init_semantic_matcher()
    assert m is not None and hasattr(m, "match")
