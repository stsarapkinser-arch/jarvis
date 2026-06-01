"""Тесты pattern fast-path (L1) — параметрические команды без 3B.

Проверяем три плоскости:
  1. Чистый матчинг шаблонов (patterns.match) — извлечение слота, числительные
     словами, отказ на фразах-алиасах без аргумента и на разговоре.
  2. Инвариант каталога: каждый шаблон ссылается на существующий навык, а имена
     групп ⊆ params навыка (дрейф каталога ловится в CI, а не в проде).
  3. Интеграция в оркестратор: _try_pattern_fastpath зовёт навык с args и
     озвучивает шаблонную reply; process_intent при матче не трогает 3B.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.singleton import Singleton
from src.inference.router import IntentCategory
from src.skills import patterns


# ───────────────────────── 1. Чистый матчинг ─────────────────────────
def test_digitize_folds_numeral_words():
    assert patterns.digitize("громкость двадцать пять") == "громкость 25"
    assert patterns.digitize("яркость сто") == "яркость 100"
    assert patterns.digitize("громкость 30") == "громкость 30"  # цифры — насквозь
    assert patterns.digitize("быстрый скан") == "быстрый скан"  # без числительных


def test_match_volume_digits():
    m = patterns.match("громкость 30")
    assert m is not None
    assert m.skill_id == "set_volume"
    assert m.args == {"percent": "30"}
    assert "30" in m.reply


@pytest.mark.parametrize("phrase,percent", [
    ("сделай громкость двадцать пять", "25"),
    ("поставь звук на 50", "50"),
    ("громкость 80 процентов", "80"),
])
def test_match_volume_variants(phrase, percent):
    m = patterns.match(phrase)
    assert m is not None and m.skill_id == "set_volume"
    assert m.args == {"percent": percent}


def test_match_brightness():
    m = patterns.match("яркость на 70 процентов")
    assert m is not None
    assert m.skill_id == "set_brightness"
    assert m.args == {"percent": "70"}


@pytest.mark.parametrize("phrase,skill_id,target", [
    ("быстрый скан 192.168.1.1", "nmap_quick", "192.168.1.1"),
    ("скан сервисов 10.0.0.5", "nmap_service_scan", "10.0.0.5"),
    ("кто в сети 10.0.0.0/24", "nmap_ping_sweep", "10.0.0.0/24"),
    ("топ порты example.com", "nmap_top_ports", "example.com"),
    ("просканируй порты 192.168.1.1.", "nmap_quick", "192.168.1.1"),  # крайняя точка снимается
])
def test_match_scan_targets(phrase, skill_id, target):
    m = patterns.match(phrase)
    assert m is not None, f"шаблон не поймал {phrase!r}"
    assert m.skill_id == skill_id
    assert m.args == {"target": target}


@pytest.mark.parametrize("phrase", [
    "громкость",            # точный алиас без аргумента → отдать L0/3B, не L1
    "быстрый скан",         # то же — алиас без цели
    "расскажи как дела",    # разговор
    "в чём смысл жизни",    # разговор
    "сделай погромче",      # не шаблон со слотом
    "",                     # пусто
])
def test_no_match_passes_through(phrase):
    assert patterns.match(phrase) is None


# ───────────────────────── 2. Инвариант каталога ─────────────────────────
def test_every_pattern_points_to_real_skill_with_matching_slots():
    """Каждый шаблон → существующий навык; имена групп ⊆ params навыка."""
    from src import skills  # импорт регистрирует каталог
    for pat in patterns.all_patterns():
        sk = skills.get(pat.skill_id)
        assert sk is not None, f"шаблон {pat.id!r} ссылается на неизвестный навык"
        slot_names = set(pat.regex.groupindex.keys())
        param_names = set(sk.params.keys())
        assert slot_names <= param_names, (
            f"шаблон {pat.id!r} извлекает {slot_names - param_names}, "
            f"которых нет в params навыка {sk.id} ({param_names})"
        )


# ───────────────────────── 3. Интеграция в оркестратор ─────────────────────────
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
        id="set_volume", category=IntentCategory.UI_CONTROL,
        description="установить громкость", handler=None,
    )
    defaults.update(kw)
    return Skill(**defaults)


def test_pattern_fastpath_invokes_skill_with_args_and_speaks_reply():
    j = _make_jarvis()
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "volume=30%"

    fake = _fake_skill(handler=handler)
    pm = patterns.PatternMatch(
        skill_id="set_volume", args={"percent": "30"},
        reply="Громкость на 30 процентов, сэр.",
    )
    with patch("src.core.ladder.patterns.match", return_value=pm), \
         patch("src.core.ladder.skills.get", return_value=fake), \
         patch("src.core.ladder.snapshot", return_value={}):
        handled = asyncio.run(j._try_pattern_fastpath("громкость 30"))

    assert handled is True
    assert ran == [{"percent": "30"}], "хендлер должен получить извлечённый слот"
    # навык не speaks_result → озвучиваем шаблонную reply, а не сырой результат.
    assert any("30 процентов" in s for s in spoken)
    assert "volume=30%" not in spoken


def test_pattern_fastpath_destructive_queues_confirmation_with_args():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]
    ran: list = []

    async def handler(ctx, args):
        ran.append(args)
        return "done"

    fake = _fake_skill(id="dangerous", destructive=True, handler=handler)
    pm = patterns.PatternMatch(skill_id="dangerous", args={"percent": "30"}, reply="ok")
    with patch("src.core.ladder.patterns.match", return_value=pm), \
         patch("src.core.ladder.skills.get", return_value=fake), \
         patch("src.core.ladder.snapshot", return_value={}):
        handled = asyncio.run(j._try_pattern_fastpath("снеси 30"))

    assert handled is True
    assert ran == [], "разрушительный навык не исполняется до подтверждения"
    assert j._pending_skill is not None
    assert j._pending_skill["args"] == {"percent": "30"}


def test_pattern_fastpath_no_match_returns_false():
    j = _make_jarvis()
    with patch("src.core.ladder.patterns.match", return_value=None):
        handled = asyncio.run(j._try_pattern_fastpath("в чём смысл жизни"))
    assert handled is False


def test_pattern_fastpath_unknown_skill_falls_through():
    """Шаблон ссылается на снятый навык → не падаем, отдаём агенту (False)."""
    j = _make_jarvis()
    pm = patterns.PatternMatch(skill_id="ghost", args={}, reply="ok")
    with patch("src.core.ladder.patterns.match", return_value=pm), \
         patch("src.core.ladder.skills.get", return_value=None):
        handled = asyncio.run(j._try_pattern_fastpath("призрак 5"))
    assert handled is False


def test_process_intent_pattern_skips_agent_loop():
    """При матче шаблона process_intent отвечает без 3B (агент не зовётся)."""
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "ok"

    fake = _fake_skill(handler=handler)
    pm = patterns.PatternMatch(skill_id="set_volume", args={"percent": "30"}, reply="ok")

    async def fake_agent(*a, **kw):
        raise AssertionError("агентный цикл не должен вызываться при pattern fast-path")

    j._run_agent_for_intent = fake_agent  # type: ignore[assignment]
    with patch("src.core.ladder.patterns.match", return_value=pm), \
         patch("src.core.ladder.skills.get", return_value=fake), \
         patch("src.core.ladder.skills.match_alias", return_value=None), \
         patch("src.core.ladder.macros.match_macro", return_value=None), \
         patch("src.core.ladder.snapshot", return_value={}):
        res = asyncio.run(j.process_intent("громкость 30"))

    assert res == "[pattern_fastpath]"
