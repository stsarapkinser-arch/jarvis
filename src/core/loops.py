"""Фоновые циклы оркестратора — mixin.

Вынесено из orchestrator.py без изменения поведения: три долгоживущие
asyncio-задачи и их хелперы — проактивная речь, проактивный взгляд (OCR) и
ночная рефлексия памяти. Все три гейтятся одним ``ProactiveGate`` (тихие часы,
анти-спам, простой) и берегут N100. Mixin держит только методы/константы; всё
состояние и сервисы приходят через ``self`` из Jarvis.

Описание воли/взгляда/рефлексии — в src/README.md.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from src.common.event_bus import SystemLoad, SystemState
from src.inference.router import IntentCategory
from src.memory import reflection
from src.skills import screen as screen_skills

log = logging.getLogger("jarvis.core")


class BackgroundLoopsMixin:
    """Проактивная речь, проактивный взгляд и ночная рефлексия. Подмешивается в
    :class:`~src.core.orchestrator.Jarvis`."""

    async def _proactive_loop(self) -> None:
        """Checks every 60s if Jarvis has been silent for PROACTIVE_INTERVAL.

        Generates a proactive phrase via LLM based on live context — no
        hardcoded templates. The LLM sees system state and is asked to
        come up with a natural observation as Jarvis."""
        await asyncio.sleep(60)  # let boot complete before first check
        _last_proactive_ts: float = 0.0
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                state = SystemState()
                # Такт инициативы — единый гейт: тихие часы, анти-спам, простой.
                if not self._proactive_gate.should_speak(
                    now=now,
                    last_say_ts=self._last_say_ts,
                    last_proactive_ts=_last_proactive_ts,
                    load_busy=state.load in (SystemLoad.HIGH, SystemLoad.CRITICAL),
                    hour=time.localtime(now).tm_hour,
                ):
                    continue

                snap = state.snapshot

                # Контекст для LLM — живые метрики, недавние события
                context_parts: list[str] = []
                if snap is not None:
                    context_parts.append(
                        f"[System: load={state.load.value} "
                        f"cpu={snap.cpu_pct:.0f}% ram={snap.ram_pct:.0f}% "
                        f"thermal={snap.thermal:.0f}C "
                        f"time={time.strftime('%H:%M')}]"
                    )
                recent = list(self._recent_os)
                warns = [e for e in recent if e.get("level") in ("warn", "critical")]
                if warns:
                    sensors = ", ".join(dict.fromkeys(e["sensor"] for e in warns[-3:]))
                    context_parts.append(f"[RecentAlerts: {sensors}]")

                proactive_prompt = (
                    "\n".join(context_parts) + "\n"
                    "[User Intent: (проактивная инициатива Джарвиса — оператор давно молчит)]\n\n"
                    "Придумай одну короткую реплику от лица Джарвиса. "
                    "Она должна органично вытекать из контекста выше: "
                    "заметь что-то конкретное в системе или поведении оператора. "
                    "Никаких шаблонов «Сэр, краткий статус» — только живое наблюдение. "
                    "Ответь одной фразой живой речью, без markdown и без пояснений."
                )
                try:
                    # Лёгкий completion без tools — проактивной реплике не нужен
                    # function-calling, только одна фраза. CONVERSATION-промпт даёт
                    # персону, не нагружая модель системными правилами.
                    phrase = await self._llama_complete(
                        proactive_prompt,
                        system=self.router.system_prompt_for(IntentCategory.CONVERSATION),
                        max_tokens=120,
                        temperature=0.7,
                    )
                    phrase = phrase.strip()[:200]
                except Exception:
                    log.exception("proactive phrase generation failed")
                    phrase = ""

                # Не повторяем недавно сказанную тему (гейт ведёт окно дедупа).
                if phrase and not self._proactive_gate.is_repeat(phrase):
                    _last_proactive_ts = time.time()
                    self._proactive_gate.record(phrase)
                    self.say(phrase, tone="idle")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("proactive loop error")

    def start_proactive_loop(self) -> asyncio.Task:
        return asyncio.create_task(self._proactive_loop(), name="jarvis-proactive")

    # ──────────────────── proactive screen perception ────────────────────────
    # Каденция дорогого OCR. Запускаем РЕДКО и только когда оператор в простое
    # (тот же ProactiveGate) — N100 не должен пыхтеть OCR'ом в фоне.
    SCREEN_WATCH_INTERVAL = 90  # сек между проверками (не чаще, и лишь при простое)

    @staticmethod
    def _screen_watch_enabled() -> bool:
        # Приватность: постоянное чтение экрана — сильная штука. По умолчанию
        # включено (оператор сам просил «глаза»), но отключаемо одним env.
        return os.getenv("JARVIS_SCREEN_WATCH", "1").strip().lower() not in ("0", "false", "no", "off")

    async def _screen_watch_loop(self) -> None:
        """Проактивный взгляд: заметив на экране НОВЫЙ стек-трейс/ошибку в
        dev-контексте, Джарвис сам предлагает помощь — связка «глаз» (OCR) с
        волей (ProactiveGate). Дёшево по такту: пред-фильтр по заголовку окна,
        дорогой OCR — только в терминале/IDE и только в простое."""
        if not self._screen_watch_enabled():
            log.info("screen-watch disabled (JARVIS_SCREEN_WATCH=0)")
            return
        await asyncio.sleep(90)  # дать загрузке устаканиться
        last_offer_ts: float = 0.0
        while True:
            try:
                await asyncio.sleep(self.SCREEN_WATCH_INTERVAL)
                now = time.time()
                state = SystemState()
                # Тот же такт, что у проактивной речи: тихие часы, простой,
                # анти-спам. Если оператор активен/система занята — даже не OCR'им.
                if not self._proactive_gate.should_speak(
                    now=now,
                    last_say_ts=self._last_say_ts,
                    last_proactive_ts=last_offer_ts,
                    load_busy=state.load in (SystemLoad.HIGH, SystemLoad.CRITICAL),
                    hour=time.localtime(now).tm_hour,
                ):
                    continue

                # Дёшево: заголовок активного окна. OCR только если это похоже на
                # терминал/редактор/IDE (и приватность, и экономия CPU).
                _, title_out, _ = await self._run(
                    "kdotool getactivewindow getwindowname 2>/dev/null "
                    "|| xdotool getactivewindow getwindowname 2>/dev/null"
                )
                title = (title_out.strip().splitlines() or [""])[0].strip()
                if not screen_skills.looks_like_dev_context(title):
                    continue

                # Дорого: OCR экрана. Сюда доходим редко — гейт + dev-контекст.
                _, ocr_out, _ = await self._run(screen_skills.ocr_command())
                frags = screen_skills.extract_error_fragments(ocr_out)
                if not frags:
                    continue

                offer = screen_skills.build_error_offer(frags)
                # Не предлагать одно и то же (окно дедупа гейта).
                if self._proactive_gate.is_repeat(offer):
                    continue
                last_offer_ts = time.time()
                self._proactive_gate.record(offer)
                self.say(offer, tone="idle")
                # В память — факт инициативы (без полного текста экрана: приватность).
                await asyncio.to_thread(
                    self.memory.remember,
                    "[SCREEN_WATCH]", f"window={title[:60]}",
                    "proactive error offer", "screen_watch", None,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("screen-watch loop error")

    def start_screen_watch_loop(self) -> asyncio.Task:
        return asyncio.create_task(self._screen_watch_loop(), name="jarvis-screen-watch")

    # ──────────────────── nightly memory reflection ──────────────────────────

    REFLECTION_CHECK_SEC = 3600          # проверяем раз в час
    REFLECTION_INTERVAL_SEC = 20 * 3600  # консолидируем не чаще раза в ~сутки

    def _recent_memory_docs(self, hours: int = 24, limit: int = 24) -> list[str]:
        """Собрать документы памяти за последние ``hours`` из warm+lite слоёв.

        Синхронно (Chroma .get блокирующий) — вызывать через ``to_thread``."""
        cutoff = time.time() - hours * 3600
        docs: list[str] = []
        for col in (self.memory.warm, self.memory.lite):
            try:
                data = col.get(where={"ts": {"$gte": cutoff}})
                docs.extend(data.get("documents") or [])
            except Exception:
                log.debug("reflection: tier read failed", exc_info=True)
        return docs[-limit:]

    async def _reflection_loop(self) -> None:
        """Раз в сутки в простое: перечитать недавнюю память, извлечь устойчивые
        факты об операторе и записать их в core-слой (постоянная память)."""
        await asyncio.sleep(120)  # дать системе подняться
        last_run = 0.0
        while True:
            try:
                await asyncio.sleep(self.REFLECTION_CHECK_SEC)
                now = time.time()
                if now - last_run < self.REFLECTION_INTERVAL_SEC:
                    continue
                if SystemState().load in (SystemLoad.HIGH, SystemLoad.CRITICAL):
                    continue  # не грузим мозг рефлексией под нагрузкой
                if not await self._llm.health():
                    continue
                last_run = now
                docs = await asyncio.to_thread(self._recent_memory_docs)
                if not docs:
                    continue
                existing = await asyncio.to_thread(
                    lambda: [d["document"] for d in self.memory.list_core()]
                )

                async def _reflect_llm(prompt: str) -> str:
                    return await self._llama_complete(
                        prompt,
                        system=self.router.system_prompt_for(IntentCategory.CONVERSATION),
                        max_tokens=200,
                        temperature=0.3,
                    )

                written = await reflection.consolidate(
                    docs=docs,
                    existing_core_docs=existing,
                    llm_complete=_reflect_llm,
                    remember_core=self.memory.remember_core,
                )
                if written:
                    log.info("reflection consolidated %d core facts", len(written))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("reflection loop error")

    def start_reflection_loop(self) -> asyncio.Task:
        return asyncio.create_task(self._reflection_loop(), name="jarvis-reflection")
