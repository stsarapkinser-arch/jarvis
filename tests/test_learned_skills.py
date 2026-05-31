"""Тесты загрузчика выученных навыков (self-authoring, слой хранения).

Проверяем валидацию записей, сборку рабочего хендлера над фиксированной
командой и round-trip записи в файл. Глобальный реестр не трогаем — тестируем
чистые функции и файловый ввод-вывод во временном пути."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.router import IntentCategory
from src.skills import learned


class _FakeCtx:
    def __init__(self, run_result=(0, "ok-out", "")):
        self.spawned: list[str] = []
        self.ran: list[str] = []
        self._run_result = run_result

    async def spawn(self, cmd: str) -> int:
        self.spawned.append(cmd)
        return 777

    async def run(self, cmd: str):
        self.ran.append(cmd)
        return self._run_result


def test_valid_entry_requires_prefix_and_command():
    assert learned._valid_entry({"id": "learned_x", "command": "echo hi"})
    assert not learned._valid_entry({"id": "x", "command": "echo hi"})       # нет префикса
    assert not learned._valid_entry({"id": "learned_", "command": "echo"})   # пустой суффикс
    assert not learned._valid_entry({"id": "learned_x", "command": "  "})     # пустая команда
    assert not learned._valid_entry("not a dict")


def test_build_skill_run_kind_invokes_ctx_run():
    sk = learned._build_skill({
        "id": "learned_show_ip", "command": "ip a", "kind": "run",
        "description": "показать адреса", "category": IntentCategory.SYSTEM_OPS.value,
    })
    assert sk is not None and sk.id == "learned_show_ip"
    assert sk.destructive is False
    ctx = _FakeCtx()
    out = asyncio.run(sk.handler(ctx, {}))
    assert ctx.ran == ["ip a"]
    assert "rc=0" in out


def test_build_skill_spawn_kind_invokes_ctx_spawn():
    sk = learned._build_skill({"id": "learned_open_x", "command": "xterm", "kind": "spawn"})
    ctx = _FakeCtx()
    out = asyncio.run(sk.handler(ctx, {}))
    assert ctx.spawned == ["xterm"] and "pid=" in out


def test_append_learned_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "learned_skills.json"
    monkeypatch.setattr(learned, "LEARNED_PATH", path)
    monkeypatch.setattr(learned, "_CONFIG_DIR", tmp_path)

    entry = {"id": "learned_foo", "command": "echo foo", "kind": "run",
             "description": "foo", "category": "SYSTEM_OPS", "aliases": []}
    assert learned.append_learned(entry) is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data and data[0]["id"] == "learned_foo"

    # Дубль не добавляется второй раз.
    assert learned.append_learned(entry) is False
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1


def test_append_learned_rejects_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(learned, "LEARNED_PATH", tmp_path / "x.json")
    monkeypatch.setattr(learned, "_CONFIG_DIR", tmp_path)
    assert learned.append_learned({"id": "bad", "command": "echo"}) is False


def test_read_raw_tolerates_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "learned_skills.json"
    path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(learned, "LEARNED_PATH", path)
    assert learned._read_raw() == []
