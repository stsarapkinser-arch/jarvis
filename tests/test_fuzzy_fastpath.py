"""Тесты L1b — fuzzy-матч алиасов/макросов (дрейф распознавания Vosk).

Плоскости:
  1. Движок fuzzy_best: порог, минимальная длина, группировка по цели и зазор
     над вторым кандидатом (анти-неоднозначность).
  2. fuzzy_match_alias / fuzzy_match_macro на реальном каталоге: ловит дрейф,
     исключает разрушительные, молчит на разговоре и коротких фразах.
  3. Интеграция: _try_fuzzy_fastpath (макрос выше навыка) и process_intent,
     минующий 3B при fuzzy-матче.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import skills
from src.common.singleton import Singleton
from src.inference.router import IntentCategory
from src.skills import macros
from src.skills.registry import fuzzy_best


# ───────────────────────── 1. Движок fuzzy_best ─────────────────────────
def test_fuzzy_best_clear_winner():
    assert fuzzy_best("терминал", [("терминал", "a"), ("браузер", "b")]) == "a"


def test_fuzzy_best_below_threshold_none():
    assert fuzzy_best("абвгдеж", [("яэюьъыщ", "a")]) is None


def test_fuzzy_best_short_input_none():
    # короче FUZZY_MIN_LEN (6) — только точный матч на верхних слоях, тут None.
    assert fuzzy_best("да", [("дай", "a")]) is None


def test_fuzzy_best_ambiguous_returns_none():
    # два РАЗНЫХ кандидата равно близки → зазор < margin → неоднозначно → None.
    assert fuzzy_best("abcdefgh", [("abcdefgx", "a"), ("abcdefgy", "b")]) is None


def test_fuzzy_best_same_group_not_ambiguous():
    # два близких алиаса ОДНОЙ цели — не неоднозначность (сравнение по группам).
    assert fuzzy_best("abcdefgh", [("abcdefgx", "a"), ("abcdefgy", "a")]) == "a"


# ───────────────────────── 2. Реальный каталог ─────────────────────────
@pytest.mark.parametrize("phrase,skill_id", [
    ("терминэл", "open_terminal"),    # дрейф гласной
    ("скриншод", "open_screenshot_tool"),
    ("калькулятар", "open_calculator"),
])
def test_fuzzy_match_alias_absorbs_drift(phrase, skill_id):
    sk = skills.fuzzy_match_alias(phrase)
    assert sk is not None and sk.id == skill_id


@pytest.mark.parametrize("phrase", [
    "в чём смысл жизни",   # разговор
    "расскажи анекдот",    # разговор
])
def test_fuzzy_match_alias_ignores_conversation(phrase):
    assert skills.fuzzy_match_alias(phrase) is None


def test_fuzzy_match_alias_excludes_destructive():
    """Разрушительный навык (log_out) не должен ловиться по созвучию — даже
    близкая ослышка не имеет права тянуть destructive."""
    # «разлогинся» — дрейф точного destructive-алиаса «разлогинься».
    sk = skills.fuzzy_match_alias("разлогинса")
    assert sk is None or not sk.destructive


def test_fuzzy_match_alias_short_phrase_none():
    assert skills.fuzzy_match_alias("ок") is None


@pytest.mark.parametrize("phrase,macro_id", [
    ("режим фокус", "focus_mode"),
    ("доброе утра", "start_day"),
])
def test_fuzzy_match_macro_absorbs_drift(phrase, macro_id):
    m = macros.fuzzy_match_macro(phrase)
    assert m is not None and m.id == macro_id


def test_fuzzy_match_macro_ignores_conversation():
    assert macros.fuzzy_match_macro("что ты думаешь о свободе воли") is None


# ───────────────────────── 3. Интеграция ─────────────────────────
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


def test_fuzzy_fastpath_alias_invokes_skill():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "ok"

    fake = _fake_skill(handler=handler)
    with patch("src.core.orchestrator.macros.fuzzy_match_macro", return_value=None), \
         patch("src.core.orchestrator.skills.fuzzy_match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_fuzzy_fastpath("терминэл"))

    assert handled is True
    assert ran == [{}]


def test_fuzzy_fastpath_macro_takes_priority():
    """Макрос проверяется первым — навык не должен даже опрашиваться."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    fake_macro = macros.Macro(id="focus_mode", description="d", reply="Включаю.", steps=())
    alias_probed = {"hit": False}

    def _alias(_text):
        alias_probed["hit"] = True
        return None

    with patch("src.core.orchestrator.macros.fuzzy_match_macro", return_value=fake_macro), \
         patch("src.core.orchestrator.skills.fuzzy_match_alias", side_effect=_alias), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        handled = asyncio.run(j._try_fuzzy_fastpath("режим фокус"))

    assert handled is True
    assert alias_probed["hit"] is False, "при матче макроса навык не опрашиваем"


def test_fuzzy_fastpath_no_match_returns_false():
    j = _make_jarvis()
    with patch("src.core.orchestrator.macros.fuzzy_match_macro", return_value=None), \
         patch("src.core.orchestrator.skills.fuzzy_match_alias", return_value=None):
        handled = asyncio.run(j._try_fuzzy_fastpath("абракадабра"))
    assert handled is False


def test_process_intent_fuzzy_skips_agent_loop():
    """Fuzzy-матч (после промахов L0/L1/L1a) отвечает без 3B."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "ok"

    fake = _fake_skill(handler=handler)

    async def fake_agent(*a, **kw):
        raise AssertionError("агент не должен вызываться при fuzzy fast-path")

    j._run_agent_for_intent = fake_agent  # type: ignore[assignment]
    with patch("src.core.orchestrator.skills.match_alias", return_value=None), \
         patch("src.core.orchestrator.macros.match_macro", return_value=None), \
         patch("src.core.orchestrator.patterns.match", return_value=None), \
         patch("src.core.orchestrator.macros.fuzzy_match_macro", return_value=None), \
         patch("src.core.orchestrator.skills.fuzzy_match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        res = asyncio.run(j.process_intent("терминэл"))

    assert res == "[fuzzy_fastpath]"
