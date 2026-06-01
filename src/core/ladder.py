"""Лестница распознавания (L0→L2) — mixin для оркестратора.

Вынесено из orchestrator.py без изменения поведения: это связная группа
детерминированных ступеней «речь → навык, минуя 3B» (alias L0, macro L0,
pattern L1, fuzzy L1b, semantic L2) плюс их общий исполнитель навыка/сценария.
Mixin держит только методы; всё состояние и сервисы (память, шина, say/_state,
_invoke_skill, _semantic, _pending_skill) приходят через ``self`` из Jarvis.

Полное описание лестницы и её стоимости — в src/README.md.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from src.common.event_bus import Event, EventType
from src.core.confirm import CONFIRM_HINT
from src.memory.snapshot import snapshot
from src import skills
from src.skills import macros
from src.skills import patterns

log = logging.getLogger("jarvis.core")


class RecognitionLadderMixin:
    """Детерминированные ступени распознавания + исполнители fast-path навыка
    и макроса. Подмешивается в :class:`~src.core.orchestrator.Jarvis`."""

    async def _try_macro(self, text: str) -> bool:
        """Точная фраза-сценарий → последовательность навыков, минуя 3B.

        Композиция уже выверенных навыков под одну команду («рабочее место
        пентеста», «режим фокуса»). Детерминирована; разрушительные шаги
        пропускаются (двойная защита — макросы их и так не содержат)."""
        macro = macros.match_macro(text)
        if macro is None:
            return False
        log.info("macro: %r → %s (%d шагов)", text[:60], macro.id, len(macro.steps))
        await self._run_macro(macro, text)
        return True

    async def _run_macro(self, macro: macros.Macro, text: str) -> None:
        """Исполнить сценарий: интро-реплика + шаги-навыки + сводный брифинг.

        Общий исполнитель для точного (``_try_macro``) и fuzzy
        (``_try_fuzzy_fastpath``) путей — поведение идентично, отличается лишь
        способ, которым макрос найден."""
        snap = await asyncio.to_thread(snapshot)
        await self._state("THINKING", macro.id)
        if macro.reply.strip():
            self.say(macro.reply.strip())

        spoken_results: list[str] = []
        for skill_id, args in macro.steps:
            sk = skills.get(skill_id)
            if sk is None:
                log.warning("macro %s: неизвестный навык %s — пропуск", macro.id, skill_id)
                continue
            if sk.destructive:
                log.warning("macro %s: разрушительный навык %s — пропуск", macro.id, skill_id)
                continue
            result = await self._invoke_skill(sk, dict(args))
            if sk.speaks_result and result.strip():
                spoken_results.append(result.strip())

        # Телеметрию шагов (speaks_result) собираем в один краткий брифинг.
        summary = " ".join(spoken_results)
        if summary:
            self.say(summary)
            await self.bus.publish(Event(EventType.TOKEN_STREAM, summary))
        await asyncio.to_thread(
            self.memory.remember, text, f"macro:{macro.id}", summary or "ok", "macro", snap,
        )
        self._record_turn(text, summary or macro.reply or self.ack("ok"))
        await self._state("IDLE", "")

    async def _run_fastpath_skill(
        self,
        sk: skills.Skill,
        text: str,
        args: dict[str, Any],
        *,
        source: str,
        reply: str | None = None,
    ) -> bool:
        """Исполнить одиночный навык вне агентного цикла — общий хвост всех
        fast-path'ов (alias L0, pattern L1, fuzzy L1b).

        ``source`` тегирует память (``{source}_skill``). ``reply`` — фраза для
        озвучки у не-телеметрийных навыков (шаблонная у L1); None → короткий ack
        (контекстную фразу без модели не сочинить). Разрушительный навык уходит
        на голосовое подтверждение с переданными args. Всегда возвращает True."""
        snap = await asyncio.to_thread(snapshot)

        # Разрушительный навык — голосовое подтверждение (как из агентного цикла);
        # переиспользуем очередь _pending_skill / _try_resolve_skill.
        if sk.destructive:
            self._pending_skill = {
                "skill": sk, "args": dict(args), "intent": text,
                "snap": snap, "ts": time.time(),
            }
            await self._state("ALERT", f"CONFIRM skill: {sk.id}")
            self.say(f"{sk.description}. {CONFIRM_HINT}", tone="alert")
            return True

        await self._state("THINKING", sk.id)
        result = await self._invoke_skill(sk, dict(args))
        # speaks_result → живой результат хендлера (телеметрия); иначе — переданная
        # reply (шаблонная у L1) либо короткий ack.
        spoken = result if sk.speaks_result else (reply if reply is not None else self.ack("ok"))
        if spoken.strip():
            self.say(spoken.strip())
            await self.bus.publish(Event(EventType.TOKEN_STREAM, spoken.strip()))
        await asyncio.to_thread(
            self.memory.remember, text, f"skill:{sk.id}", result, f"{source}_skill", snap,
        )
        self._record_turn(text, spoken)
        await self._state("IDLE", "")
        return True

    async def _try_alias_fastpath(self, text: str) -> bool:
        """Точная фраза-алиас → прямой детерминированный вызов навыка, минуя
        маршрутизатор и 3B. Главный рычаг латентности на N100: бытовые команды
        («терминал», «тише», «что на экране») отвечают мгновенно, не занимая
        одно-слотовый llama-server. Возвращает True, если интент поглощён.

        Только ТОЧНОЕ совпадение (см. skills.match_alias): фразы с аргументами
        («быстрый скан 10.0.0.1») не матчатся и уходят модели — она извлечёт
        цель. Навыки без аргумента, требующие его (nmap), сами вежливо попросят
        уточнить — поведение идентично агентному пути."""
        sk = skills.match_alias(text)
        if sk is None:
            return False
        log.info("alias fast-path: %r → skill=%s", text[:60], sk.id)
        return await self._run_fastpath_skill(sk, text, {}, source="alias")

    async def _try_pattern_fastpath(self, text: str) -> bool:
        """Параметрическая фраза-ШАБЛОН → навык с извлечённым слотом, минуя 3B (L1).

        Ступень между alias fast-path (точная фраза) и агентным циклом: ловит
        КЛАСС команд с аргументом — «громкость 30», «яркость на 70», «быстрый
        скан 10.0.0.1». Раньше всё это уходило на 3B (медленно, и модель путалась
        в извлечении слота); теперь regex с именованными группами достаёт слот
        детерминированно и зовёт выверенный навык. Возвращает True, если поглощено."""
        pm = patterns.match(text)
        if pm is None:
            return False
        sk = skills.get(pm.skill_id)
        if sk is None:
            # Дрейф каталога: шаблон ссылается на снятый навык. Не падаем —
            # отдаём интент 3B (тест-инвариант ловит это в CI, а не в проде).
            log.warning("pattern fast-path: навык %r не зарегистрирован — отдаю агенту", pm.skill_id)
            return False
        log.info("pattern fast-path: %r → skill=%s args=%s", text[:60], sk.id, pm.args)
        return await self._run_fastpath_skill(sk, text, dict(pm.args), source="pattern", reply=pm.reply)

    async def _try_fuzzy_fastpath(self, text: str) -> bool:
        """L1b: fuzzy-совпадение фразы с алиасом макроса/навыка → исполнить, минуя
        3B. Спасательная ступень ПОСЛЕ точных L0/L1/L1a и ПЕРЕД агентом: гасит
        дрейф Vosk («терминэл» → «терминал»), когда точного совпадения нет.

        Сценарий выше одиночного навыка (как и при точном матче). Строгие гарды
        (длина/порог/зазор, исключение разрушительных) живут в registry.fuzzy_*
        — здесь только диспетчеризация на общие исполнители."""
        macro = macros.fuzzy_match_macro(text)
        if macro is not None:
            log.info("fuzzy macro: %r → %s", text[:60], macro.id)
            await self._run_macro(macro, text)
            return True
        sk = skills.fuzzy_match_alias(text)
        if sk is not None:
            log.info("fuzzy alias: %r → skill=%s", text[:60], sk.id)
            return await self._run_fastpath_skill(sk, text, {}, source="fuzzy")
        return False

    @staticmethod
    def _init_semantic_matcher():
        """Создать SemanticSkillMatcher, если включён JARVIS_SEMANTIC_MATCH=1.

        Возвращает None при выключенном флаге или сбое (нет httpx и т.п.) —
        матчер строго опционален. EmbeddingClient ленив (сети при создании нет),
        так что конструктор безопасен даже при лежащем embed-сервере."""
        if os.getenv("JARVIS_SEMANTIC_MATCH", "0") != "1":
            return None
        try:
            from src.inference.embeddings import EmbeddingClient
            from src.inference.semantic import SemanticSkillMatcher
            matcher = SemanticSkillMatcher(EmbeddingClient().embed)
            log.info("L2 semantic matcher включён (порог %.2f)", matcher.threshold)
            return matcher
        except Exception:
            log.exception("L2 semantic matcher: инициализация не удалась — выключен")
            return None

    async def _try_semantic_fastpath(self, text: str) -> bool:
        """L2: семантический матч навыка через эмбеддинги, минуя 3B. Последняя
        ступень перед агентом: ловит парафраз без строкового сходства («запусти
        браузер» → open_browser). Дороже строковых слоёв (HTTP+CPU на embed), но
        на порядки дешевле 3B — поэтому только когда L0/L1/L1a/L1b промахнулись.

        Выключен (matcher=None) → no-op. Эмбеддинг синхронный → в тред, чтобы не
        морозить event-loop. Гарды (порог/зазор, исключение разрушительных и
        параметрических) живут в матчере; здесь только диспетчеризация."""
        if self._semantic is None:
            return False
        sid = await asyncio.to_thread(self._semantic.match, text)
        if not sid:
            return False
        sk = skills.get(sid)
        if sk is None or sk.destructive or sk.params:
            # Матчер не должен такое отдавать (в индексе их нет), но страхуемся:
            # семантическая догадка не запускает разрушительное/слот-зависимое.
            return False
        log.info("semantic fast-path: %r → skill=%s", text[:60], sk.id)
        return await self._run_fastpath_skill(sk, text, {}, source="semantic")
