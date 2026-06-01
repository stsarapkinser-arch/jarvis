"""Тесты загрузчика пользовательских команд (config/commands.toml).

Проверяем валидацию записей, сборку рабочего хендлера над фиксированной командой
(spawn/run + озвучка ``say``), генерацию уникального id и устойчивость к битому
TOML. Глобальный реестр навыков не трогаем — тестируем чистые функции и файловый
ввод-вывод во временном пути (как в test_learned_skills)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.router import IntentCategory
from src.skills import custom


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


def test_valid_entry_requires_phrases_and_run():
    assert custom._valid_entry({"phrases": ["открой загрузки"], "run": "xdg-open ~"})
    assert custom._valid_entry({"phrases": "одной строкой", "run": "echo hi"})  # str → [str]
    assert not custom._valid_entry({"phrases": [], "run": "echo hi"})            # нет фраз
    assert not custom._valid_entry({"phrases": ["x"], "run": "  "})              # пустая команда
    assert not custom._valid_entry({"run": "echo hi"})                          # нет phrases
    assert not custom._valid_entry("not a dict")


def test_phrases_normalizes_str_and_filters_blanks():
    assert custom._phrases({"phrases": "заблокируй"}) == ["заблокируй"]
    assert custom._phrases({"phrases": ["a", "  ", "", "b"]}) == ["a", "b"]
    assert custom._phrases({"phrases": 42}) == []


def test_build_skill_spawn_kind_invokes_ctx_spawn_and_speaks_say():
    sk = custom._build_skill(
        {"phrases": ["открой видеоплеер"], "run": "dolphin ~", "spawn": True,
         "say": "Открываю, сэр."},
        set(),
    )
    assert sk is not None
    assert sk.id == "custom_otkroi_videopleer"
    assert sk.speaks_result is True
    assert sk.destructive is False
    assert sk.aliases == ("открой видеоплеер",)
    ctx = _FakeCtx()
    out = asyncio.run(sk.handler(ctx, {}))
    assert ctx.spawned == ["dolphin ~"] and ctx.ran == []
    assert out == "Открываю, сэр."


def test_build_skill_run_kind_waits_and_defaults_say():
    sk = custom._build_skill({"phrases": ["проверь связь"], "run": "ping -c1 1.1.1.1"}, set())
    ctx = _FakeCtx()
    out = asyncio.run(sk.handler(ctx, {}))
    assert ctx.ran == ["ping -c1 1.1.1.1"] and ctx.spawned == []
    assert out == custom._DEFAULT_SAY  # say по умолчанию


def test_build_skill_destructive_flag_and_category():
    sk = custom._build_skill(
        {"phrases": ["очисти корзину"], "run": "rm -rf ~/.local/share/Trash/*",
         "destructive": True, "category": "SYSTEM_OPS"},
        set(),
    )
    assert sk.destructive is True
    assert sk.category == IntentCategory.SYSTEM_OPS


def test_build_skill_bad_category_falls_back_to_ui_control():
    sk = custom._build_skill({"phrases": ["тест"], "run": "echo hi", "category": "NONSENSE"}, set())
    assert sk.category == IntentCategory.UI_CONTROL


def test_entry_id_explicit_and_collision_suffix():
    taken: set[str] = set()
    sk1 = custom._build_skill({"phrases": ["раз"], "run": "echo 1", "id": "downloads"}, taken)
    taken.add(sk1.id)
    assert sk1.id == "custom_downloads"
    # Та же первая фраза/без id у двух записей → второй получает числовой суффикс.
    a = custom._build_skill({"phrases": ["открой почту"], "run": "thunderbird"}, taken)
    taken.add(a.id)
    b = custom._build_skill({"phrases": ["открой почту"], "run": "evolution"}, taken)
    assert a.id != b.id
    assert b.id.startswith(a.id)


def test_read_raw_parses_commands_table(tmp_path, monkeypatch):
    path = tmp_path / "commands.toml"
    path.write_text(
        '[[command]]\nphrases = ["a", "b"]\nrun = "echo hi"\nsay = "ок"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(custom, "COMMANDS_PATH", path)
    raw = custom._read_raw()
    assert len(raw) == 1 and raw[0]["run"] == "echo hi" and raw[0]["phrases"] == ["a", "b"]


def test_read_raw_tolerates_corrupt_toml(tmp_path, monkeypatch):
    path = tmp_path / "commands.toml"
    path.write_text("[[command]\nthis is not toml", encoding="utf-8")
    monkeypatch.setattr(custom, "COMMANDS_PATH", path)
    assert custom._read_raw() == []


def test_read_raw_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(custom, "COMMANDS_PATH", tmp_path / "nope.toml")
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", tmp_path / "also-nope.toml")
    assert custom._read_raw() == []


def test_active_path_prefers_personal_over_example(tmp_path, monkeypatch):
    personal = tmp_path / "commands.toml"
    example = tmp_path / "commands.example.toml"
    example.write_text("", encoding="utf-8")
    monkeypatch.setattr(custom, "COMMANDS_PATH", personal)
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", example)
    assert custom._active_path() == example   # личного ещё нет → пример
    personal.write_text("", encoding="utf-8")
    assert custom._active_path() == personal  # появился личный → он главнее


def test_example_file_is_valid_and_loadable():
    """Поставляемый шаблон обязан быть валидным TOML и давать рабочие навыки —
    иначе «из коробки» сломается у оператора."""
    raw = custom._read_raw()  # на CI личного файла нет → читается example
    assert raw, "config/commands.example.toml пуст или не читается"
    taken: set[str] = set()
    for entry in raw:
        assert custom._valid_entry(entry), f"невалидная запись примера: {entry}"
        sk = custom._build_skill(entry, taken)
        taken.add(sk.id)
        assert sk.id.startswith(custom.CUSTOM_PREFIX)
        assert sk.aliases
