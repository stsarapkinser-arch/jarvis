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


def test_read_table_parses_commands_table(tmp_path):
    path = tmp_path / "commands.toml"
    path.write_text(
        '[[command]]\nphrases = ["a", "b"]\nrun = "echo hi"\nsay = "ок"\n',
        encoding="utf-8",
    )
    raw = custom._read_table(path)
    assert len(raw) == 1 and raw[0]["run"] == "echo hi" and raw[0]["phrases"] == ["a", "b"]


def test_read_table_tolerates_corrupt_toml(tmp_path):
    path = tmp_path / "commands.toml"
    path.write_text("[[command]\nthis is not toml", encoding="utf-8")
    assert custom._read_table(path) == []


def test_read_table_missing_file_returns_empty(tmp_path):
    assert custom._read_table(tmp_path / "nope.toml") == []


def test_merged_loads_defaults_when_no_personal_file(tmp_path, monkeypatch):
    """Личного файла нет → грузится только дефолтный каталог (всегда активен)."""
    example = tmp_path / "commands.example.toml"
    example.write_text(
        '[[command]]\nphrases = ["открой загрузки"]\nrun = "xdg-open ~/Downloads"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(custom, "COMMANDS_PATH", tmp_path / "nope.toml")
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", example)
    merged = custom._merged_entries()
    assert [e["run"] for e in merged] == ["xdg-open ~/Downloads"]


def test_merged_personal_extends_defaults(tmp_path, monkeypatch):
    """Личные команды ДОПОЛНЯЮТ дефолтные (а не заменяют их)."""
    personal = tmp_path / "commands.toml"
    example = tmp_path / "commands.example.toml"
    personal.write_text(
        '[[command]]\nphrases = ["открой проект"]\nrun = "code ~/proj"\n', encoding="utf-8"
    )
    example.write_text(
        '[[command]]\nphrases = ["выключи компьютер"]\nrun = "poweroff"\n', encoding="utf-8"
    )
    monkeypatch.setattr(custom, "COMMANDS_PATH", personal)
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", example)
    runs = [e["run"] for e in custom._merged_entries()]
    assert "code ~/proj" in runs and "poweroff" in runs


def test_merged_personal_overrides_default_on_phrase_collision(tmp_path, monkeypatch):
    """При совпадении фразы личная запись ВЫИГРЫВАЕТ, дефолтная отбрасывается."""
    personal = tmp_path / "commands.toml"
    example = tmp_path / "commands.example.toml"
    personal.write_text(
        '[[command]]\nphrases = ["открой загрузки"]\nrun = "MINE"\n', encoding="utf-8"
    )
    example.write_text(
        '[[command]]\nphrases = ["открой загрузки"]\nrun = "DEFAULT"\n', encoding="utf-8"
    )
    monkeypatch.setattr(custom, "COMMANDS_PATH", personal)
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", example)
    runs = [e["run"] for e in custom._merged_entries()]
    assert runs == ["MINE"]  # дефолт с той же фразой целиком отброшен


def test_merged_partial_phrase_overlap_keeps_default(tmp_path, monkeypatch):
    """Если у дефолта есть СОБСТВЕННАЯ фраза — он сохраняется (перекрыт не весь)."""
    personal = tmp_path / "commands.toml"
    example = tmp_path / "commands.example.toml"
    personal.write_text(
        '[[command]]\nphrases = ["общая фраза"]\nrun = "MINE"\n', encoding="utf-8"
    )
    example.write_text(
        '[[command]]\nphrases = ["общая фраза", "своя фраза"]\nrun = "DEFAULT"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(custom, "COMMANDS_PATH", personal)
    monkeypatch.setattr(custom, "COMMANDS_EXAMPLE_PATH", example)
    runs = [e["run"] for e in custom._merged_entries()]
    assert runs == ["MINE", "DEFAULT"]


def test_example_file_is_valid_and_loadable():
    """Поставляемый дефолтный каталог обязан быть валидным TOML и давать рабочие
    навыки — иначе «из коробки» сломается у оператора. Заодно стережём уникальность
    нормализованных фраз внутри каталога (дубль фразы = тихо мёртвая команда)."""
    raw = custom._read_table(custom.COMMANDS_EXAMPLE_PATH)
    assert raw, "config/commands.example.toml пуст или не читается"
    taken: set[str] = set()
    seen_phrases: set[str] = set()
    for entry in raw:
        assert custom._valid_entry(entry), f"невалидная запись каталога: {entry}"
        for p in custom._phrases(entry):
            norm = custom.normalize_phrase(p)
            assert norm not in seen_phrases, f"дубль фразы в каталоге: {p!r}"
            seen_phrases.add(norm)
        sk = custom._build_skill(entry, taken)
        taken.add(sk.id)
        assert sk.id.startswith(custom.CUSTOM_PREFIX)
        assert sk.aliases
