"""Skill Registry — смерть «нейросеть пишет bash на лету».

Архитектурный переворот (ТЗ оператора): LLM больше НЕ придумывает команды.
Она лишь переводит человеческую речь в строгий ``skill_id`` из заранее
написанного, протестированного, 100% рабочего каталога навыков. Python ловит
``skill_id`` и запускает заведомо рабочую функцию (выверенный qdbus6/wpctl/…).

Почему это надёжнее текущего ``execute_bash``:
  * нет dry-run в песочнице на каждую команду (ShadowExec) — навык уже проверен;
  * нет heal-цикла (модель не ошибается в синтаксисе — синтаксис захардкожен);
  * нет эвристики DESTRUCTIVE_RE — автор навыка САМ помечает ``destructive``;
  * один раунд диалога: ``run_skill(skill_id, reply)`` и говорит, и действует.

``execute_bash`` НЕ удалён — он понижен в ранге до gated-fallback для длинного
хвоста (см. ``tools.tools_for_category`` и orchestrator). Известный интент →
детерминированный навык; неизвестный → песочница + подтверждение.

Реестр — чистые данные (каталог) + протокол контекста. Никакого знания о
конкретных подсистемах Jarvis здесь нет: хендлер получает ``SkillContext``
(замыкание оркестратора над spawn/run), что держит слой навыков тестируемым.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.inference.router import IntentCategory

log = logging.getLogger("jarvis.skills")


@runtime_checkable
class SkillContext(Protocol):
    """Минимальный контракт, который оркестратор даёт хендлеру навыка.

    ``spawn`` — detached-запуск GUI-приложения (konsole/chrome/dolphin): не
    ждём завершения, возвращаем PID. ``run`` — короткая команда с ожиданием
    результата (wpctl/brightnessctl/qdbus6): возвращает (rc, stdout, stderr)."""

    async def spawn(self, cmd: str) -> int: ...
    async def run(self, cmd: str) -> tuple[int | None, str, str]: ...


# Хендлер навыка: (контекст, аргументы) -> короткая строка-результат.
# Для большинства навыков args пуст; строку результата мы пишем в память и
# (если speaks_result) озвучиваем.
SkillHandler = Callable[[SkillContext, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Skill:
    """Один навык каталога.

    ``destructive`` — навык требует голосового подтверждения перед запуском
    (как разрушительный bash). ``speaks_result`` — озвучиваем СТРОКУ-результат
    хендлера, а не ``reply`` модели (для телеметрии: живые цифры знает только
    хендлер, модель их выдумала бы)."""
    id: str
    category: IntentCategory
    description: str
    handler: SkillHandler
    params: dict[str, Any] = field(default_factory=dict)   # JSON-schema свойств args
    destructive: bool = False
    speaks_result: bool = False
    aliases: tuple[str, ...] = ()


_REGISTRY: dict[str, Skill] = {}
# Обратный индекс: нормализованный алиас → skill_id. Заполняется при
# register(); основа alias fast-path в оркестраторе (точная фраза минует 3B).
_ALIAS_INDEX: dict[str, str] = {}

# Нормализация фразы для сопоставления с алиасом: регистр, ё→е, схлопывание
# пробелов и снятие крайней пунктуации. Vosk пунктуацию не выдаёт, но текст из
# тестов/тулзов может — приводим обе стороны к одному виду.
_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT = " .,!?;:…\"'«»()[]-—–"


def normalize_phrase(text: str) -> str:
    """Нормализация фразы для матчинга алиасов/макросов: регистр, ё→е, пробелы,
    крайняя пунктуация. Публична — переиспользуется реестром макросов."""
    s = (text or "").strip().lower().replace("ё", "е")
    s = _WS_RE.sub(" ", s)
    return s.strip(_EDGE_PUNCT)


# Транслитерация кириллицы → латиница для ASCII-безопасных skill_id (id уходят
# в enum-грамматику инструмента run_skill — нелатинские символы там нежелательны).
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 32, fallback: str = "cmd") -> str:
    """Фраза → ASCII-слаг для skill_id: транслит кириллицы, не-буквы → ``_``."""
    base = (text or "").strip().lower().translate(_TRANSLIT)
    slug = _SLUG_RE.sub("_", base).strip("_")
    return slug[:max_len] or fallback


# Внутренний алиас для краткости в этом модуле.
_normalize_phrase = normalize_phrase


def register(skill: Skill) -> None:
    """Зарегистрировать навык. Дубль id — ошибка конфигурации (падаем рано)."""
    if skill.id in _REGISTRY:
        raise ValueError(f"duplicate skill id: {skill.id!r}")
    _REGISTRY[skill.id] = skill
    # Индексируем алиасы для fast-path. Коллизия алиаса между навыками — это
    # смысловая неоднозначность фразы: первый зарегистрировавший выигрывает,
    # остальные логируем (не падаем — алиас не критичен для работы навыка).
    for alias in skill.aliases:
        key = _normalize_phrase(alias)
        if not key:
            continue
        owner = _ALIAS_INDEX.get(key)
        if owner is not None and owner != skill.id:
            log.warning(
                "alias %r уже привязан к %s — игнорирую дубль от %s",
                key, owner, skill.id,
            )
            continue
        _ALIAS_INDEX.setdefault(key, skill.id)


def match_alias(text: str) -> Skill | None:
    """Точное нормализованное совпадение фразы с алиасом → навык, иначе None.

    Только ТОЧНОЕ совпадение (по нормализованной форме). Параметризованные
    фразы вроде «быстрый скан 192.168.1.1» сюда НЕ попадают и уходят модели,
    которая извлечёт аргумент. Это и есть граница fast-path: высокая точность,
    нулевой риск ложного срабатывания на curated-алиасах оператора."""
    sid = _ALIAS_INDEX.get(_normalize_phrase(text))
    if sid is None:
        return None
    return _REGISTRY.get(sid)


# ─────────────────────────── Fuzzy-матч (L1b) ───────────────────────────
# Спасательная ступень ПЕРЕД 3B: гасит дрейф распознавания Vosk («терминэл» →
# «терминал»), когда точного совпадения нет. Строгие гарды против ложных
# срабатываний: минимальная длина (короткие команды надёжны и так), высокий
# порог близости и ЗАПАС над вторым кандидатом (неоднозначное — лучше 3B).
FUZZY_THRESHOLD = 0.86   # difflib ratio: ниже — не уверены, отдаём 3B
FUZZY_MARGIN = 0.06      # лучший должен опережать второй (иначе неоднозначно)
FUZZY_MIN_LEN = 6        # и фраза, и алиас короче — только точный матч


def fuzzy_best(
    text: str,
    items: Iterable[tuple[str, str]],
    *,
    threshold: float = FUZZY_THRESHOLD,
    margin: float = FUZZY_MARGIN,
    min_len: int = FUZZY_MIN_LEN,
) -> str | None:
    """Лучший group_id для фразы среди ``items`` (alias_key → group_id), иначе None.

    Группировка по ``group_id`` (а не по алиасу) важна: синонимы ОДНОЙ цели не
    создают ложной неоднозначности — сравниваем лучший балл цели с лучшим баллом
    ДРУГОЙ цели. None при: короткой фразе, балле ниже порога, или зазоре с
    ближайшей другой целью меньше ``margin``."""
    key = _normalize_phrase(text)
    if len(key) < min_len:
        return None
    per_group: dict[str, float] = {}
    for alias, gid in items:
        if len(alias) < min_len:
            continue
        ratio = SequenceMatcher(None, key, alias).ratio()
        if ratio > per_group.get(gid, 0.0):
            per_group[gid] = ratio
    if not per_group:
        return None
    best_gid = max(per_group, key=lambda g: per_group[g])
    best = per_group[best_gid]
    if best < threshold:
        return None
    others = [r for g, r in per_group.items() if g != best_gid]
    if others and best - max(others) < margin:
        return None  # неоднозначно — отдаём 3B
    return best_gid


def fuzzy_match_alias(text: str) -> "Skill | None":
    """Fuzzy-совпадение фразы с алиасом → навык, иначе None.

    Разрушительные навыки в fuzzy-индекс НЕ входят: близкое-по-звучанию никогда
    не должно тянуть за собой `destructive` (его подтверждают, но даже вопрос
    «снести всё?» на ослышке недопустим). Их по-прежнему запускает только точный
    алиас + голосовое подтверждение."""
    items = (
        (alias, sid)
        for alias, sid in _ALIAS_INDEX.items()
        if (sk := _REGISTRY.get(sid)) is not None and not sk.destructive
    )
    gid = fuzzy_best(text, items)
    return _REGISTRY.get(gid) if gid else None


def skill(
    *,
    id: str,
    category: IntentCategory,
    description: str,
    params: dict[str, Any] | None = None,
    destructive: bool = False,
    speaks_result: bool = False,
    aliases: Sequence[str] = (),
) -> Callable[[SkillHandler], SkillHandler]:
    """Декоратор регистрации навыка прямо над его async-хендлером."""
    def deco(fn: SkillHandler) -> SkillHandler:
        register(Skill(
            id=id, category=category, description=description, handler=fn,
            params=dict(params or {}), destructive=destructive,
            speaks_result=speaks_result, aliases=tuple(aliases),
        ))
        return fn
    return deco


def get(skill_id: str) -> Skill | None:
    return _REGISTRY.get((skill_id or "").strip())


def all_skills() -> tuple[Skill, ...]:
    return tuple(_REGISTRY.values())


def skill_ids() -> list[str]:
    """Все id в порядке регистрации (стабильно → KV-префикс run_skill не плывёт).

    Порядок детерминирован порядком импорта модулей навыков в ``skills/__init__``
    и порядком декораторов внутри них."""
    return list(_REGISTRY.keys())


def skills_for(category: IntentCategory | str) -> tuple[Skill, ...]:
    try:
        cat = IntentCategory(str(category))
    except ValueError:
        return ()
    return tuple(s for s in _REGISTRY.values() if s.category == cat)


def catalog_for(category: IntentCategory | str) -> str:
    """Человекочитаемый список навыков категории для МИКРО-промпта.

    Грамматика run_skill (enum по ВСЕМ навыкам) гарантирует валидный id; этот
    каталог даёт модели семантику — какой id под какую фразу. Кладётся в
    меняющийся хвост системного промпта (не в кэшируемый префикс)."""
    items = skills_for(category)
    if not items:
        return ""
    lines = [f"  • {s.id}: {s.description}" for s in items]
    return "Доступные навыки (skill_id):\n" + "\n".join(lines)


def reset() -> None:
    """Только для тестов: очистить реестр."""
    _REGISTRY.clear()
    _ALIAS_INDEX.clear()


__all__ = [
    "SkillContext",
    "SkillHandler",
    "Skill",
    "register",
    "skill",
    "get",
    "match_alias",
    "fuzzy_match_alias",
    "fuzzy_best",
    "normalize_phrase",
    "slugify",
    "all_skills",
    "skill_ids",
    "skills_for",
    "catalog_for",
    "reset",
]
