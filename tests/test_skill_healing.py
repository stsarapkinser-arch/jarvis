"""Тесты самолечения команд навыков (автоматический «# VERIFY on target»).

Юнит SkillHealer: стор замен (load/save/forget) и генерация детерминированных
кандидатов (варианты бинаря qdbus). Интеграция: _run_skill_command чинит
падение qdbus6 → qdbus, запоминает замену и озвучивает.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.singleton import Singleton
from src.skills.healing import SkillHealer, first_binary


@pytest.fixture(autouse=True)
def reset_singletons():
    Singleton.reset()
    yield
    Singleton.reset()


# ───────────────────────── SkillHealer (unit) ─────────────────────────
def test_candidates_swap_qdbus_binary(tmp_path):
    h = SkillHealer(store_path=tmp_path / "ov.json")
    cmd = "qdbus6 org.kde.KWin /KWin org.kde.KWin.NightLight.setEnabled true"
    cands = h.candidates(cmd)
    assert any(c.startswith("qdbus ") for c in cands)
    assert any("qdbus-qt6" in c for c in cands)
    # сам qdbus6-вариант (исходный) в кандидаты не попадает.
    assert cmd not in cands


def test_candidates_empty_for_non_qdbus(tmp_path):
    h = SkillHealer(store_path=tmp_path / "ov.json")
    assert h.candidates("brightnessctl set 70%") == []


def test_remember_and_override_roundtrip(tmp_path):
    p = tmp_path / "ov.json"
    h = SkillHealer(store_path=p)
    h.remember("qdbus6 a b c", "qdbus a b c")
    assert h.override_for("qdbus6 a b c") == "qdbus a b c"
    # нормализация пробелов: ключ устойчив к форматированию.
    assert h.override_for("qdbus6   a b   c") == "qdbus a b c"
    # персистится на диск.
    assert json.loads(p.read_text(encoding="utf-8"))


def test_remember_ignores_noop(tmp_path):
    h = SkillHealer(store_path=tmp_path / "ov.json")
    h.remember("same", "same")
    assert h.override_for("same") is None


def test_forget_drops_override(tmp_path):
    h = SkillHealer(store_path=tmp_path / "ov.json")
    h.remember("qdbus6 x", "qdbus x")
    h.forget("qdbus6 x")
    assert h.override_for("qdbus6 x") is None


def test_load_reads_existing_store(tmp_path):
    p = tmp_path / "ov.json"
    p.write_text(json.dumps({"qdbus6 y": "qdbus y"}), encoding="utf-8")
    h = SkillHealer(store_path=p)
    assert h.override_for("qdbus6 y") == "qdbus y"


def test_first_binary():
    assert first_binary("qdbus6 org.kde.KWin foo") == "qdbus6"
    assert first_binary("/usr/bin/qdbus -x") == "qdbus"


# ───────────────────────── интеграция в оркестраторе ─────────────────────────
class _NullMemory:
    def remember(self, *a, **kw):
        return None

    def recall(self, *a, **kw):
        return []


def _make_jarvis(tmp_path):
    from src.core.orchestrator import Jarvis
    j = Jarvis()
    j.memory = _NullMemory()
    # Изолируем стор от реального jarvis_memory/.
    j._healer = SkillHealer(store_path=tmp_path / "ov.json")
    return j


def test_run_skill_command_heals_qdbus_and_remembers(tmp_path):
    j = _make_jarvis(tmp_path)
    spoken: list = []
    j.say = lambda text, tone=None, *a, **kw: spoken.append(text)

    calls: list = []

    async def fake_run(cmd):
        calls.append(cmd)
        # qdbus6 отсутствует (как на машине без Qt6-обёртки), qdbus — работает.
        if cmd.startswith("qdbus6"):
            return 1, "", "qdbus6: command not found"
        if cmd.startswith("qdbus "):
            return 0, "ok", ""
        return 127, "", "nope"

    j._run = fake_run  # type: ignore[assignment]

    cmd = "qdbus6 org.kde.KWin /KWin invokeShortcut 'Toggle Night Color'"
    rc, out, err = asyncio.run(j._run_skill_command(cmd))

    assert rc == 0 and out == "ok"
    # запомнил рабочую замену → впредь сразу qdbus.
    assert j._healer.override_for(cmd).startswith("qdbus ")
    # озвучил самопочинку.
    assert spoken and any("qdbus" in s for s in spoken)


def test_run_skill_command_uses_stored_override_first(tmp_path):
    j = _make_jarvis(tmp_path)
    j.say = lambda *a, **kw: None
    cmd = "qdbus6 a b c"
    j._healer.remember(cmd, "qdbus a b c")

    calls: list = []

    async def fake_run(c):
        calls.append(c)
        return (0, "ok", "") if c.startswith("qdbus ") else (1, "", "fail")

    j._run = fake_run  # type: ignore[assignment]
    rc, out, _ = asyncio.run(j._run_skill_command(cmd))
    assert rc == 0
    # первой же командой пошла выученная замена, не оригинал.
    assert calls[0].startswith("qdbus ")


def test_run_skill_command_success_is_passthrough(tmp_path):
    j = _make_jarvis(tmp_path)
    j.say = lambda *a, **kw: None

    async def fake_run(c):
        return 0, "done", ""

    j._run = fake_run  # type: ignore[assignment]
    rc, out, _ = asyncio.run(j._run_skill_command("wpctl set-volume @DEFAULT_SINK@ 50%"))
    assert (rc, out) == (0, "done")


def test_run_skill_command_no_heal_returns_failure(tmp_path):
    j = _make_jarvis(tmp_path)
    j.say = lambda *a, **kw: None

    async def fake_run(c):
        return 1, "", "boom"  # всё падает, чинить нечем

    j._run = fake_run  # type: ignore[assignment]
    rc, _, err = asyncio.run(j._run_skill_command("brightnessctl set 70%"))
    assert rc == 1 and "boom" in err
