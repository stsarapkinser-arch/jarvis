"""Каталог навыков Jarvis.

Импорт пакета РЕГИСТРИРУЕТ все навыки (декораторы ``@skill`` в подмодулях
исполняются при импорте). Порядок импорта фиксирует порядок ``skill_ids()`` →
enum инструмента ``run_skill`` стабилен между запросами (KV-префикс сервера не
пересчитывается).

Публичный API проксируется из ``registry`` для краткости вызовов:
    from src.skills import get, skill_ids, catalog_for
"""
from __future__ import annotations

from src.skills.registry import (
    Skill,
    SkillContext,
    SkillHandler,
    all_skills,
    catalog_for,
    fuzzy_match_alias,
    get,
    match_alias,
    register,
    reset,
    skill,
    skill_ids,
    skills_for,
)

# Регистрация навыков (side-effect импорта). Порядок = порядок enum.
from src.skills import ui_control as _ui_control  # noqa: E402,F401
from src.skills import system_ops as _system_ops  # noqa: E402,F401
from src.skills import pentest as _pentest        # noqa: E402,F401
from src.skills import screen as _screen          # noqa: E402,F401
# Резолвер приложений (L1a): регистрирует обобщённый навык open_app («открой X»).
from src.skills import apps as _apps              # noqa: E402,F401
# Выученные навыки (self-authoring): подхватываются из config/learned_skills.json.
# Импорт последним — id выученных навыков идут в хвост enum, не сдвигая
# стабильный префикс встроенного каталога (KV-кэш run_skill не плывёт).
from src.skills import learned as _learned        # noqa: E402,F401
# Макросы/сценарии (композиция навыков). Импорт последним — ссылаются на
# skill_id, уже зарегистрированные выше.
from src.skills import macros as _macros           # noqa: E402,F401
# Шаблоны со слотами (L1): параметрические команды → навык + слот, минуя 3B.
# Ссылаются на skill_id выше (валидируется тестом), сами навыков не регистрируют.
from src.skills import patterns as _patterns        # noqa: E402,F401

__all__ = [
    "Skill",
    "SkillContext",
    "SkillHandler",
    "all_skills",
    "catalog_for",
    "fuzzy_match_alias",
    "get",
    "match_alias",
    "register",
    "reset",
    "skill",
    "skill_ids",
    "skills_for",
]
