"""Пользовательские команды — каталог, который оператор пишет РУКАМИ.

Цель (ТЗ оператора): на голосовой запрос Джарвис должен МГНОВЕННО выполнить
заранее положенную команду, как можно реже дёргая нейросеть. Этот модуль даёт
простой, человекочитаемый файл ``config/commands.toml``, куда оператор сам
добавляет свои команды — без правки Python. Каждая запись регистрируется как
обычный навык с алиасами, поэтому ловится точным fast-path'ом (L0, микросекунды,
минуя 3B), а при дрейфе Vosk — fuzzy-ступенью (L1b).

Формат записи (TOML-таблица ``[[command]]``):
    phrases = ["открой загрузки", "папка загрузки"]   # голосовые фразы (≥1)
    run     = "xdg-open ~/Downloads"                   # что выполнить (shell)
    say     = "Открываю загрузки, сэр."                # (необяз.) ответ голосом
    spawn   = true                                     # (необяз.) GUI в фоне
    destructive = false                                # (необяз.) спросить «да?»
    id      = "downloads"                              # (необяз.) свой id
    category = "UI_CONTROL"                            # (необяз.)

Модель безопасности (как у learned-навыков): команда ФИКСИРОВАНА автором-человеком
и исполняется ровно как написана — аргументов/подстановок нет, поверхность инъекций
нулевая. Произнесённая фраза в shell НЕ попадает (она лишь ключ-алиас). Файл лежит
в ``config/`` под наблюдением FileChangeWatcher (суффикс .toml), поэтому правка и
сохранение горячо перезапускают процесс — команда становится активной сразу.
"""
from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from src.inference.router import IntentCategory
from src.skills.registry import Skill, SkillContext, get as _get, register, slugify

log = logging.getLogger("jarvis.skills.custom")

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
# Личный файл оператора (в .gitignore — правки не конфликтуют с git pull).
COMMANDS_PATH = _CONFIG_DIR / "commands.toml"
# Шаблон с примерами (в git). Если личного файла ещё нет — работаем по примеру,
# чтобы команды были «из коробки», как и просили («положены заранее»).
COMMANDS_EXAMPLE_PATH = _CONFIG_DIR / "commands.example.toml"

# Префикс id всех пользовательских команд — отделяет их от встроенного каталога
# и выученных навыков, и не даёт случайно перекрыть встроенный навык.
CUSTOM_PREFIX = "custom_"

# Потолок: не раздуваем enum-грамматику run_skill бесконечным каталогом.
MAX_CUSTOM = 128

_DEFAULT_SAY = "Готово, сэр."


def _active_path() -> Path:
    """Личный файл оператора, если есть; иначе шаблон с примерами."""
    return COMMANDS_PATH if COMMANDS_PATH.exists() else COMMANDS_EXAMPLE_PATH


def _read_raw() -> list[dict[str, Any]]:
    """Прочитать ``[[command]]``-таблицы. Любая ошибка → пустой список (не падаем)."""
    path = _active_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        log.exception("чтение %s не удалось", path)
        return []
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        log.warning("%s повреждён (невалидный TOML) — игнорирую", path)
        return []
    cmds = data.get("command")
    if not isinstance(cmds, list):
        return []
    return [c for c in cmds if isinstance(c, dict)]


def _phrases(entry: dict[str, Any]) -> list[str]:
    """Непустые строковые фразы записи (нормализацию делает сам реестр алиасов)."""
    raw = entry.get("phrases")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [p.strip() for p in raw if isinstance(p, str) and p.strip()]


def _valid_entry(entry: Any) -> bool:
    """Минимальная валидация: хотя бы одна фраза и непустая команда ``run``."""
    if not isinstance(entry, dict):
        return False
    run = entry.get("run")
    return bool(_phrases(entry)) and isinstance(run, str) and bool(run.strip())


def _make_handler(command: str, spawn: bool, say: str):
    """Хендлер над ФИКСИРОВАННОЙ командой. ``spawn`` — detached GUI (не ждём),
    иначе ждём завершения. Озвучивается ``say`` (speaks_result у навыка)."""
    async def handler(ctx: SkillContext, args: dict[str, Any]) -> str:
        if spawn:
            await ctx.spawn(command)
        else:
            await ctx.run(command)
        return say
    return handler


def _entry_id(entry: dict[str, Any], phrases: list[str], taken: set[str]) -> str:
    """Уникальный ASCII-id с префиксом ``custom_``: из поля ``id`` либо из первой
    фразы; при коллизии добавляем числовой суффикс."""
    raw = entry.get("id")
    base = slugify(str(raw)) if isinstance(raw, str) and raw.strip() else slugify(phrases[0])
    sid = f"{CUSTOM_PREFIX}{base}"
    if sid not in taken and _get(sid) is None:
        return sid
    i = 2
    while f"{sid}_{i}" in taken or _get(f"{sid}_{i}") is not None:
        i += 1
    return f"{sid}_{i}"


def _build_skill(entry: dict[str, Any], taken: set[str]) -> Skill | None:
    """Собрать Skill из валидной записи (id уникален в пределах ``taken``)."""
    phrases = _phrases(entry)
    command = str(entry["run"]).strip()
    spawn = bool(entry.get("spawn", False))
    say = str(entry.get("say") or _DEFAULT_SAY).strip() or _DEFAULT_SAY
    destructive = bool(entry.get("destructive", False))
    try:
        category = IntentCategory(str(entry.get("category", IntentCategory.UI_CONTROL)))
    except ValueError:
        category = IntentCategory.UI_CONTROL
    sid = _entry_id(entry, phrases, taken)
    # description: для destructive она и есть фраза подтверждения — берём say.
    description = str(entry.get("description") or say)
    return Skill(
        id=sid,
        category=category,
        description=description,
        handler=_make_handler(command, spawn, say),
        destructive=destructive,
        speaks_result=True,
        aliases=tuple(phrases),
    )


def load_custom() -> int:
    """Зарегистрировать все валидные пользовательские команды. Возвращает их число.

    Встроенный каталог всегда главнее: id-коллизия с уже зарегистрированным
    навыком пропускается. Невалидные записи логируются и пропускаются — один
    кривой блок в файле не должен валить весь каталог."""
    count = 0
    taken: set[str] = set()
    for entry in _read_raw()[:MAX_CUSTOM]:
        if not _valid_entry(entry):
            log.warning("пропускаю невалидную команду: %r", entry)
            continue
        sk = _build_skill(entry, taken)
        if sk is None:
            continue
        try:
            register(sk)
            taken.add(sk.id)
            count += 1
        except ValueError:
            log.warning("дубль id пользовательской команды: %s", sk.id)
    if count:
        log.info("загружено пользовательских команд: %d", count)
    return count


# Side-effect импорта: регистрируем пользовательские команды (как learned-навыки).
load_custom()


__all__ = [
    "COMMANDS_PATH",
    "COMMANDS_EXAMPLE_PATH",
    "CUSTOM_PREFIX",
    "load_custom",
]
