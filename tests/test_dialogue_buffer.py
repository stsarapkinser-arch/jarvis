"""Тесты буфера диалога (связный контекст последних реплик).

Буфер держит последние N пар (оператор → Джарвис) и подаёт их в build_context
отдельной секцией перед интентом — чтобы модель разрешала ссылки («его», «то
же», «второй»). Проверяем: запись только полных пар, формат секции, попадание
в контекст, вытеснение по maxlen и фиксацию реплики на fast-path.
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


def test_record_turn_keeps_only_full_pairs():
    j = _make_jarvis()
    j._record_turn("открой терминал", "Готово.")
    j._record_turn("", "пусто")           # нет ввода — пропуск
    j._record_turn("команда", "")          # немой ответ — пропуск
    j._record_turn("   ", "   ")            # только пробелы — пропуск
    assert list(j._dialogue) == [("открой терминал", "Готово.")]


def test_record_turn_skips_synthetic_daemon_intents():
    j = _make_jarvis()
    j._record_turn("[DAEMON_ALERT] thermal 85C", "Снижаю частоту.")
    j._record_turn("[SYSTEM_EVENT cpu]", "Ок.")
    assert list(j._dialogue) == [], "синтетические интенты не идут в диалог"


def test_fmt_dialogue_none_when_empty():
    j = _make_jarvis()
    assert j._fmt_dialogue() == "[Dialogue: none]"


def test_fmt_dialogue_renders_turns():
    j = _make_jarvis()
    j._record_turn("сделай тише", "Готово.")
    out = j._fmt_dialogue()
    assert "[Dialogue: last 1]" in out
    assert "оператор: сделай тише" in out
    assert "ты: Готово." in out


def test_build_context_includes_dialogue_section():
    j = _make_jarvis()
    # пусто → секция none, но присутствует и стоит ПЕРЕД интентом.
    ctx0 = j.build_context("привет", [])
    assert "[Dialogue: none]" in ctx0
    assert ctx0.index("[Dialogue:") < ctx0.index("[User Intent:")

    j._record_turn("открой браузер", "Открываю.")
    ctx1 = j.build_context("закрой его", [])
    assert "открой браузер" in ctx1
    assert "[User Intent: закрой его]" in ctx1


def test_dialogue_buffer_evicts_oldest():
    from src.core.orchestrator import DIALOGUE_TURNS
    j = _make_jarvis()
    for i in range(DIALOGUE_TURNS + 3):
        j._record_turn(f"команда {i}", f"ответ {i}")
    assert len(j._dialogue) == DIALOGUE_TURNS
    # самый старый вытеснен, самый свежий на месте.
    assert j._dialogue[0][0] == f"команда {3}"
    assert j._dialogue[-1][0] == f"команда {DIALOGUE_TURNS + 2}"


def test_fastpath_records_dialogue_turn():
    j = _make_jarvis()
    j.say = lambda *a, **kw: None

    async def _noop(ev):
        return None

    j.bus.publish = _noop  # type: ignore[assignment]

    async def handler(ctx, args):
        return "raw"

    from src.skills.registry import Skill
    fake = Skill(id="open_terminal", category=IntentCategory.UI_CONTROL,
                 description="терминал", handler=handler)
    with patch("src.core.orchestrator.skills.match_alias", return_value=fake), \
         patch("src.core.orchestrator.snapshot", return_value={}):
        asyncio.run(j._try_alias_fastpath("терминал"))

    assert len(j._dialogue) == 1
    user, assistant = j._dialogue[0]
    assert user == "терминал"
    assert assistant  # озвученный ack зафиксирован
