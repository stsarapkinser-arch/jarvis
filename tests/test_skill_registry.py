"""Тесты каталога навыков (Skill Registry) — архитектурный переворот.

Проверяем: реестр заполняется при импорте, id уникальны и валидны, категории
корректны, хендлеры — корутины, и каталог для промпта непуст для action-категорий.
Также — что выверенные навыки реально зовут заданную команду через SkillContext
(без реального исполнения — контекст замокан).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import skills
from src.inference.router import IntentCategory


class _FakeCtx:
    """SkillContext-заглушка: записывает команды, ничего не исполняет."""

    def __init__(self, run_result=(0, "", "")):
        self.spawned: list[str] = []
        self.ran: list[str] = []
        self._run_result = run_result

    async def spawn(self, cmd: str) -> int:
        self.spawned.append(cmd)
        return 4242

    async def run(self, cmd: str):
        self.ran.append(cmd)
        return self._run_result


def test_registry_populated_on_import():
    ids = skills.skill_ids()
    assert ids, "каталог навыков не должен быть пустым"
    # Примеры из ТЗ оператора обязаны присутствовать.
    assert "open_terminal" in ids
    assert "open_files" in ids
    assert "enable_night_mode" in ids


def test_skill_ids_unique():
    ids = skills.skill_ids()
    assert len(ids) == len(set(ids)), "id навыков должны быть уникальны"


def test_every_skill_well_formed():
    for sk in skills.all_skills():
        assert isinstance(sk.id, str) and sk.id
        assert isinstance(sk.category, IntentCategory)
        assert isinstance(sk.description, str) and sk.description
        assert asyncio.iscoroutinefunction(sk.handler), f"{sk.id}: хендлер не async"


def test_skill_ids_order_is_stable():
    """Порядок enum должен быть детерминирован между вызовами (KV-префикс)."""
    assert skills.skill_ids() == skills.skill_ids()


def test_skills_for_category_filters():
    ui = {s.id for s in skills.skills_for(IntentCategory.UI_CONTROL)}
    assert "open_terminal" in ui
    # телеметрия — в SYSTEM_OPS, не в UI.
    assert "report_cpu" not in ui
    sysops = {s.id for s in skills.skills_for(IntentCategory.SYSTEM_OPS)}
    assert "report_cpu" in sysops


def test_catalog_nonempty_for_action_categories():
    for cat in (IntentCategory.UI_CONTROL, IntentCategory.SYSTEM_OPS,
                IntentCategory.PENTEST_RECON):
        cat_text = skills.catalog_for(cat)
        assert cat_text and "skill_id" in cat_text


def test_get_unknown_returns_none():
    assert skills.get("does_not_exist") is None
    assert skills.get("") is None


def test_open_files_spawns_dolphin():
    ctx = _FakeCtx()
    sk = skills.get("open_files")
    result = asyncio.run(sk.handler(ctx, {}))
    assert ctx.spawned and "dolphin" in ctx.spawned[0]
    assert "pid=" in result


def test_set_volume_clamps_and_runs_wpctl():
    ctx = _FakeCtx()
    sk = skills.get("set_volume")
    asyncio.run(sk.handler(ctx, {"percent": 250}))  # вне диапазона → клампим до 100
    assert ctx.ran and "wpctl set-volume" in ctx.ran[0]
    assert "100%" in ctx.ran[0]


def test_report_cpu_speaks_result():
    sk = skills.get("report_cpu")
    assert sk.speaks_result is True
    ctx = _FakeCtx()
    out = asyncio.run(sk.handler(ctx, {}))
    assert "процент" in out.lower()  # человекочитаемая фраза, не сырые цифры


def test_nmap_quick_rejects_bad_target():
    ctx = _FakeCtx()
    sk = skills.get("nmap_quick")
    out = asyncio.run(sk.handler(ctx, {"target": "; rm -rf /"}))
    assert not ctx.ran, "грязная цель не должна запускать команду"
    assert "цель" in out.lower()


def test_nmap_quick_accepts_clean_target():
    ctx = _FakeCtx(run_result=(0, "Nmap done: 1 host up\n", ""))
    sk = skills.get("nmap_quick")
    asyncio.run(sk.handler(ctx, {"target": "10.0.0.1"}))
    assert ctx.ran and "nmap" in ctx.ran[0] and "10.0.0.1" in ctx.ran[0]


# ───────────────────────── alias fast-path ─────────────────────────
def test_match_alias_exact():
    """Точная фраза-алиас резолвится в свой навык."""
    sk = skills.match_alias("терминал")
    assert sk is not None and sk.id == "open_terminal"


def test_match_alias_normalizes_case_punct_and_yo():
    """Регистр, крайняя пунктуация и ё/е игнорируются при матчинге."""
    assert skills.match_alias("  Терминал!  ").id == "open_terminal"
    # ночной режим объявлен через «ё» в одном из алиасов — ищем через «е».
    assert skills.match_alias("теплый экран") is not None
    assert skills.match_alias("тёплый экран") is not None
    assert skills.match_alias("Тёплый Экран") is not None


def test_match_alias_no_partial_match():
    """Фраза с аргументом НЕ матчится точным алиасом — уходит модели."""
    assert skills.match_alias("быстрый скан 192.168.1.1") is None
    assert skills.match_alias("просто болтаем о погоде") is None
    assert skills.match_alias("") is None


def test_match_alias_points_to_real_skill():
    """Каждый проиндексированный алиас ведёт к существующему навыку."""
    from src.skills import registry
    for alias, sid in registry._ALIAS_INDEX.items():
        assert skills.get(sid) is not None, f"алиас {alias!r} → несуществующий {sid}"


def test_alias_index_has_no_skill_id_collisions_with_real_phrases():
    """Контроль здравомыслия: парность скан-фраз ведёт в pentest-навыки."""
    assert skills.match_alias("открытые порты").id == "list_listening_ports"
    assert skills.match_alias("кто в сети").id == "nmap_ping_sweep"
