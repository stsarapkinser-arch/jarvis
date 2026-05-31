"""Макросы/сценарии — композиция навыков одной фразой.

Один навык = одно действие. Макрос = именованная ПОСЛЕДОВАТЕЛЬНОСТЬ навыков под
одну команду оператора: «рабочее место пентеста», «режим фокуса», «доложи
обстановку». Это следующий уровень над каталогом: не новые низкоуровневые
команды, а сценарии из уже выверенных навыков.

Исполнение детерминировано (никакого LLM): фраза точно матчит алиас макроса →
ядро гонит шаги по очереди через тот же ``_invoke_skill``. Шаги ссылаются ТОЛЬКО
на существующие skill_id; разрушительные навыки в макросы не кладём (а ядро при
исполнении их всё равно пропустит — двойная защита).

Матчинг — точный, по нормализованной фразе (как alias fast-path навыков), так
что параметризованная речь сюда не попадает и уходит модели.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.skills.registry import normalize_phrase

log = logging.getLogger("jarvis.macros")

# Один шаг макроса: (skill_id, args). Большинству шагов args не нужен.
Step = tuple[str, dict]


@dataclass(frozen=True, slots=True)
class Macro:
    """Сценарий: фраза → короткая интро-реплика + последовательность навыков."""
    id: str
    description: str
    reply: str                       # короткая интро-фраза («Готовлю рабочее место.»)
    steps: tuple[Step, ...]
    aliases: tuple[str, ...] = field(default_factory=tuple)


_MACROS: dict[str, Macro] = {}
_ALIAS_INDEX: dict[str, str] = {}


def register(macro: Macro) -> None:
    if macro.id in _MACROS:
        raise ValueError(f"duplicate macro id: {macro.id!r}")
    _MACROS[macro.id] = macro
    for alias in macro.aliases:
        key = normalize_phrase(alias)
        if not key:
            continue
        owner = _ALIAS_INDEX.get(key)
        if owner is not None and owner != macro.id:
            log.warning("macro-alias %r уже у %s — игнор дубля от %s", key, owner, macro.id)
            continue
        _ALIAS_INDEX.setdefault(key, macro.id)


def match_macro(text: str) -> Macro | None:
    """Точное нормализованное совпадение фразы с алиасом макроса, иначе None."""
    mid = _ALIAS_INDEX.get(normalize_phrase(text))
    return _MACROS.get(mid) if mid else None


def all_macros() -> tuple[Macro, ...]:
    return tuple(_MACROS.values())


def reset() -> None:
    """Только для тестов: очистить реестр макросов."""
    _MACROS.clear()
    _ALIAS_INDEX.clear()


# ─────────────────────────── Встроенные сценарии ───────────────────────────
# Каждый шаг — существующий skill_id. Телеметрию (speaks_result) ядро озвучит,
# действия (UI) просто выполнит.
_BUILTINS: tuple[Macro, ...] = (
    Macro(
        id="pentest_workspace",
        description="развернуть рабочее место для пентеста",
        reply="Готовлю рабочее место для пентеста, сэр.",
        steps=(
            ("open_terminal", {}),
            ("report_ip_address", {}),
            ("list_listening_ports", {}),
        ),
        aliases=("рабочее место пентеста", "рабочее место для пентеста",
                 "режим пентеста", "боевой режим"),
    ),
    Macro(
        id="focus_mode",
        description="режим фокуса: тёплый экран и тишина",
        reply="Включаю режим фокуса.",
        steps=(
            ("enable_night_mode", {}),
            ("volume_mute", {}),
        ),
        aliases=("режим фокуса", "сосредоточиться", "не отвлекай", "фокус режим"),
    ),
    Macro(
        id="system_briefing",
        description="устный отчёт о состоянии системы",
        reply="Снимаю показатели, сэр.",
        steps=(
            ("report_cpu", {}),
            ("report_memory", {}),
            ("report_disk", {}),
        ),
        aliases=("доложи обстановку", "статус системы", "как самочувствие",
                 "полный отчёт", "как дела у системы"),
    ),
    Macro(
        id="start_day",
        description="утренний сценарий: дневной экран, время, заряд, браузер",
        reply="Доброе утро, сэр. Разворачиваю рабочий день.",
        steps=(
            ("disable_night_mode", {}),
            ("report_datetime", {}),
            ("report_battery", {}),
            ("open_browser", {}),
        ),
        aliases=("доброе утро", "начни день", "рабочий день", "утренний режим"),
    ),
)

for _m in _BUILTINS:
    register(_m)


__all__ = [
    "Macro",
    "Step",
    "register",
    "match_macro",
    "all_macros",
    "reset",
]
