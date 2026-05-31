"""Reflector — память, которая взрослеет.

Раз в сутки (в простое, под низкой нагрузкой) Джарвис перечитывает события
последних суток из тёплого/lite-слоёв ``ChronoMemory`` и просит модель выделить
УСТОЙЧИВЫЕ факты об операторе и его предпочтениях — то, что стоит помнить
постоянно. Извлечённое уходит в core-слой (``remember_core``), который не
истекает по TTL. Через недели Джарвис начинает по-настоящему знать оператора, а
не отвечать с чистого листа.

Слой намеренно тонкий и тестируемый: построение промпта и разбор ответа модели —
чистые функции; ввод-вывод (recall/LLM/запись) делегируется вызывающему.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger("jarvis.memory.reflection")

# Метка строки ответа → core-«kind» (см. engine.CORE_KINDS).
_KIND_MAP = {
    "PREFERENCE": "preference",
    "ПРЕДПОЧТЕНИЕ": "preference",
    "HABIT": "preference",
    "ПРИВЫЧКА": "preference",
    "FACT": "identity",
    "ФАКТ": "identity",
    "PROJECT": "project",
    "ПРОЕКТ": "project",
}
_LINE_RE = re.compile(r"^\s*([A-ZА-Я]+)\s*[:：]\s*(.+?)\s*$")

# Сколько недавних записей максимум скармливаем модели (бюджет контекста 3B).
MAX_REFLECT_DOCS = 24
# Не плодим core-факты лавиной: максимум извлечений за один проход.
MAX_FACTS_PER_RUN = 5
# Минимальная длина факта — отсекаем «ок», «да» и прочий шум.
_MIN_FACT_LEN = 8


@dataclass(frozen=True, slots=True)
class Fact:
    kind: str
    text: str


def build_prompt(docs: list[str]) -> str:
    """Собрать промпт-консолидатор из недавних записей памяти."""
    body = "\n".join(f"- {d.strip()}" for d in docs[:MAX_REFLECT_DOCS] if d.strip())
    return (
        "Ниже — журнал недавних взаимодействий оператора с системой.\n"
        f"{body}\n\n"
        "Выдели УСТОЙЧИВЫЕ факты об операторе: предпочтения, привычки, имена, "
        "проекты, рабочие пути — то, что полезно помнить ПОСТОЯННО. Игнорируй "
        "разовые команды и сиюминутный шум.\n"
        "Ответь строго строками вида «PREFERENCE: ...», «FACT: ...» или "
        "«PROJECT: ...», по одному факту в строке, без markdown и пояснений. "
        "Если ничего достойного постоянной памяти нет — ответь одним словом NONE."
    )


def parse_facts(text: str) -> list[Fact]:
    """Разобрать ответ модели в список фактов. Мусор/NONE → пустой список."""
    out: list[Fact] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        if line.strip().upper() == "NONE":
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        label, body = m.group(1).upper(), m.group(2).strip()
        kind = _KIND_MAP.get(label)
        if kind is None or len(body) < _MIN_FACT_LEN:
            continue
        key = body.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(Fact(kind=kind, text=body))
        if len(out) >= MAX_FACTS_PER_RUN:
            break
    return out


def novel_facts(facts: list[Fact], existing_core_docs: list[str]) -> list[Fact]:
    """Отсеять факты, уже присутствующие в core-памяти (грубый substring-дедуп)."""
    existing = [d.lower() for d in existing_core_docs]
    fresh: list[Fact] = []
    for f in facts:
        low = f.text.lower()
        if any(low in e or e in low for e in existing if e):
            continue
        fresh.append(f)
    return fresh


async def consolidate(
    docs: list[str],
    existing_core_docs: list[str],
    llm_complete: Callable[[str], Awaitable[str]],
    remember_core: Callable[[str, str], object],
) -> list[Fact]:
    """Полный цикл рефлексии: prompt → LLM → разбор → дедуп → запись в core.

    ``llm_complete(prompt) -> text`` и ``remember_core(fact, kind)`` инъектятся
    вызывающим (оркестратор передаёт свои реализации), что держит модуль чистым
    и тестируемым. Возвращает фактически записанные факты."""
    if not docs:
        return []
    try:
        raw = await llm_complete(build_prompt(docs))
    except Exception:
        log.exception("reflection LLM call failed")
        return []
    facts = novel_facts(parse_facts(raw), existing_core_docs)
    written: list[Fact] = []
    for f in facts:
        try:
            remember_core(f.text, f.kind)
            written.append(f)
        except Exception:
            log.exception("remember_core failed for %r", f.text[:60])
    if written:
        log.info("reflection wrote %d core facts", len(written))
    return written


__all__ = ["Fact", "build_prompt", "parse_facts", "novel_facts", "consolidate"]
