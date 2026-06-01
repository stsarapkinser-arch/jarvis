"""Тесты L1a — резолвер приложений «открой X» без 3B.

Три плоскости:
  1. Резолвер (apps.resolve): точный синоним, fuzzy (шум Vosk), round-trip по
     ключу, отказ на неизвестном/коротком имени.
  2. Матчинг через patterns.match: глагол запуска + имя → open_app(app=ключ);
     отказ, когда имя не резолвится (уходит 3B).
  3. Навык open_app: команда берётся ИЗ КАТАЛОГА (анти-инъекция), неизвестное
     имя — мягкий отказ без запуска.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.skills import apps, patterns


# ───────────────────────── 1. Резолвер ─────────────────────────
@pytest.mark.parametrize("name,key", [
    ("браузер", "browser"),
    ("хром", "browser"),
    ("терминал", "terminal"),
    ("консоль", "terminal"),
    ("телеграм", "telegram"),
    ("монитор системы", "system_monitor"),
    ("параметры системы", "settings"),
    ("настройки системы", "settings"),   # порядок слов: раньше уходило на 3B
    ("панель управления", "settings"),
    ("текстовый редактор", "editor"),
    ("командную строку", "terminal"),
    ("менеджер файлов", "files"),
    ("эксель", "spreadsheet"),
])
def test_resolve_exact_synonym(name, key):
    entry = apps.resolve(name)
    assert entry is not None and entry.key == key


@pytest.mark.parametrize("name,key", [
    ("браузор", "browser"),     # типичная подмена гласной Vosk
    ("телеграмм", "telegram"),  # лишняя буква
    ("калькулятар", "calculator"),
])
def test_resolve_fuzzy_absorbs_asr_drift(name, key):
    entry = apps.resolve(name)
    assert entry is not None and entry.key == key


def test_resolve_roundtrip_on_key():
    """resolve(ключ) обязан вернуть ту же запись — open_app получает ключ и
    должен суметь его зарезолвить обратно."""
    for entry in apps.all_apps():
        got = apps.resolve(entry.key)
        assert got is not None and got.key == entry.key


@pytest.mark.parametrize("name", [
    "дверь", "ракета", "музыка", "смысл жизни", "что-нибудь интересное",
])
def test_resolve_unknown_returns_none(name):
    assert apps.resolve(name) is None


def test_resolve_short_noise_returns_none():
    # короче _FUZZY_MIN_LEN — только точный матч, иначе None (без fuzzy-шума).
    assert apps.resolve("ок") is None
    assert apps.resolve("ыы") is None


# ───────────────────────── 2. Матчинг через patterns.match ─────────────────────────
@pytest.mark.parametrize("phrase,key", [
    ("открой браузер", "browser"),
    ("запусти терминал", "terminal"),
    ("вруби телеграм", "telegram"),
    ("открой мне калькулятор", "calculator"),
    ("открой монитор системы", "system_monitor"),
    ("открой браузер пожалуйста", "browser"),
    ("открой приложение телеграм", "telegram"),
])
def test_match_open_resolves_to_open_app(phrase, key):
    m = patterns.match(phrase)
    assert m is not None, f"не поймал {phrase!r}"
    assert m.skill_id == "open_app"
    assert m.args == {"app": key}
    assert "Открываю" in m.reply


@pytest.mark.parametrize("phrase", [
    "открой",                 # без имени
    "открой что-нибудь",      # неизвестная сущность → 3B
    "включи музыку",          # не приложение
    "расскажи как дела",      # вообще не запуск
])
def test_match_open_unknown_passes_through(phrase):
    assert patterns.match(phrase) is None


def test_open_does_not_shadow_l1_numeric():
    """L1a не должен перехватывать числовые шаблоны L1 (глаголы не пересекаются)."""
    m = patterns.match("громкость 30")
    assert m is not None and m.skill_id == "set_volume"


# ───────────────────────── 3. Навык open_app ─────────────────────────
class _FakeCtx:
    def __init__(self) -> None:
        self.spawned: list[str] = []

    async def spawn(self, cmd: str) -> int:
        self.spawned.append(cmd)
        return 4242

    async def run(self, cmd: str):
        return (0, "", "")


def test_open_app_spawns_catalog_command():
    ctx = _FakeCtx()
    result = asyncio.run(apps.open_app(ctx, {"app": "browser"}))
    assert "launched pid=4242" in result
    assert ctx.spawned and ("chrome" in ctx.spawned[0] or "chromium" in ctx.spawned[0])


def test_open_app_accepts_synonym_not_just_key():
    ctx = _FakeCtx()
    result = asyncio.run(apps.open_app(ctx, {"app": "хром"}))
    assert "launched pid=4242" in result
    assert ctx.spawned


def test_open_app_unknown_does_not_spawn():
    ctx = _FakeCtx()
    result = asyncio.run(apps.open_app(ctx, {"app": "ракетный двигатель"}))
    assert "Не нашёл" in result
    assert ctx.spawned == [], "неизвестное имя не должно ничего запускать"


def test_open_app_command_never_built_from_slot():
    """Анти-инъекция: что бы оператор ни произнёс, в spawn уходит только
    захардкоженная команда каталога, а не текст слота."""
    ctx = _FakeCtx()
    asyncio.run(apps.open_app(ctx, {"app": "браузер; rm -rf /"}))
    # Слот не зарезолвился (мусор) → ничего не запущено; инъекция невозможна.
    assert ctx.spawned == []
