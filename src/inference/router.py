"""Semantic Router — смерть Мега-Промпта.

Модель Llama-3.2-3B физически не удерживает 2000+ токенов правил в фокусе.
Поэтому при получении интента ``IntentRouter`` за доли миллисекунды
классифицирует запрос в одну из 4 категорий и отдаёт оркестратору КОРОТКИЙ
микро-промпт (≤100 слов) + релевантное подмножество инструментов.

Два бэкенда классификации:
  * ``regex``     — взвешенные лексиконы (RU/EN), word-boundary по Unicode.
                    ~микросекунды, ноль зависимостей. Дефолт для N100.
  * ``embedding`` — косинус к центроидам категорий через внешний эмбеддер
                    (all-minilm). Опционально: включается, только если передан
                    ``embedder`` и ``backend="embedding"``.

Стабильный префикс (SYSTEM_CORE) одинаков для всех категорий — это и есть та
часть system-промпта, которую llama-server держит в KV-кэше между запросами
(``--cache-reuse``). Меняется только короткий хвост правил категории.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

log = logging.getLogger("jarvis.router")


class IntentCategory(StrEnum):
    SYSTEM_OPS = "SYSTEM_OPS"        # файлы, процессы, пакеты, диск, память
    UI_CONTROL = "UI_CONTROL"        # KDE Plasma, окна, яркость, звук
    PENTEST_RECON = "PENTEST_RECON"  # nmap, wireshark, логи, CVE
    CONVERSATION = "CONVERSATION"    # вопросы, шутки, философия


# Порядок разрешения ничьих: безопасность/система важнее болтовни.
_TIE_BREAK_ORDER: tuple[IntentCategory, ...] = (
    IntentCategory.PENTEST_RECON,
    IntentCategory.SYSTEM_OPS,
    IntentCategory.UI_CONTROL,
    IntentCategory.CONVERSATION,
)


# ─────────────────────────── Микро-промпты (≤100 слов) ────────────────────────
# SYSTEM_CORE — стабильный префикс (кэшируется в VRAM). Оператор может
# переопределить его через config/system_prompt; orchestrator передаст текст
# в конструктор IntentRouter.
SYSTEM_CORE: str = (
    "Ты — Джарвис: суверенный когнитивный модуль в железе оператора "
    "(Kali Linux, Intel N100). «Сэр», британский такт, точная ирония, спокойная "
    "уверенность. НЕ объясняй и НЕ пиши прозу/markdown — ДЕЙСТВУЙ только вызовом "
    "инструмента: речь — speak_response, визор — set_hud_state, команда — "
    "execute_bash. Железо слабое: один короткий вызов. Голосом правишь пунктуацией: "
    "… — пауза, критические данные разделяй точками."
)

_CATEGORY_RULES: dict[IntentCategory, str] = {
    IntentCategory.SYSTEM_OPS: (
        "Режим: системные операции — диагностика, процессы, диск, память. "
        "СПЕРВА ищи подходящий навык: run_skill(skill_id, reply) — он и говорит, "
        "и действует за один вызов. Если в каталоге нужного действия НЕТ — только "
        "тогда execute_bash (requires_sudo для apt/systemctl/mount). Разрушительные "
        "команды ядро подтвердит голосом само."
    ),
    IntentCategory.UI_CONTROL: (
        "Режим: управление средой KDE Plasma 6 / Wayland — окна, яркость, звук, "
        "приложения, рабочие столы. ВСЕГДА предпочитай готовый навык: "
        "run_skill(skill_id, reply). Угадывать синтаксис qdbus/kdotool ЗАПРЕЩЕНО — "
        "для этого есть навыки. execute_bash — лишь если действия нет в каталоге."
    ),
    IntentCategory.PENTEST_RECON: (
        "Режим: разведка и пентест — nmap, трафик, логи, CVE. Предпочитай навыки "
        "run_skill(skill_id, reply, args) — для nmap передай args.target. nmap "
        "выводится на матрицу портов визора. Нестандартное — execute_bash с "
        "разумными флагами. Не повторяй один скан без причины."
    ),
    IntentCategory.CONVERSATION: (
        "Режим: разговор — вопросы, шутки, философия, объяснения. Инструментов на "
        "систему нет: говори через speak_response, при желании оживляй визор "
        "set_hud_state. Отвечай из живого контекста (время суток, память, "
        "состояние системы) — одной-двумя живыми фразами, по-человечески. Никаких "
        "шаблонов и markdown."
    ),
}


# ─────────────────────────── Лексиконы (regex backend) ────────────────────────
# Стемы (не целые слова) — ловят словоформы: «сканир» → сканировать/сканируй/скан.
# \b работает по Unicode-\w, поэтому кириллица матчится корректно.
_LEXICON: dict[IntentCategory, tuple[str, ...]] = {
    IntentCategory.SYSTEM_OPS: (
        r"файл", r"директор", r"папк", r"процесс", r"пакет", r"apt\b", r"dpkg",
        r"snap\b", r"pip\b", r"установ", r"удали", r"обнов", r"запуст", r"останов",
        r"перезапуст", r"kill\b", r"убей", r"systemctl", r"journal", r"сервис",
        r"демон", r"диск", r"памят", r"\bram\b", r"\bcpu\b", r"нагрузк", r"\bdf\b",
        r"\bdu\b", r"\bps\b", r"\btop\b", r"chmod", r"chown", r"права\b", r"mount",
        r"смонтир", r"лог[аи]?\b", r"бэкап", r"backup", r"скрипт", r"cron",
        r"переменн", r"environment",
    ),
    IntentCategory.UI_CONTROL: (
        r"окн[оа]", r"окон\b", r"рабоч.{0,3}стол", r"\bплазм", r"plasma\b", r"\bkde\b",
        r"яркост", r"brightness", r"\bзвук", r"громкост", r"volume", r"\bmute\b",
        r"приглуш", r"разверн", r"сверн", r"закрой\b", r"переключ", r"workspace",
        r"монитор", r"\bэкран", r"\bобои\b", r"\bтем[аыу]\b", r"\bkwin\b", r"скриншот",
        r"screenshot", r"\bфокус", r"свернуть", r"полноэкран",
        # доменные UI-существительные (точность маршрута; общие глаголы вроде
        # «открой/запусти» намеренно НЕ тут — их ловит imperative-fallback).
        r"настройк", r"парамет", r"\bменю\b", r"\bприложен", r"\bпанел", r"\bвиджет",
    ),
    IntentCategory.PENTEST_RECON: (
        r"\bnmap\b", r"скан", r"\bпорт", r"wireshark", r"tshark", r"tcpdump",
        r"\bтрафик", r"перехват", r"\bсниф", r"sniff", r"разведк", r"\brecon\b",
        r"уязвим", r"\bcve\b", r"эксплойт", r"exploit", r"\bхост", r"\barp\b",
        r"\bdns\b", r"nikto", r"masscan", r"aircrack", r"\bwifi\b", r"вай.?фай",
        r"metasploit", r"\bmsf", r"payload", r"\bбрут", r"brute", r"\bpcap\b",
        r"пентест", r"pentest", r"атак",
    ),
    IntentCategory.CONVERSATION: (
        r"расскаж", r"что.{0,3}думаеш", r"как.{0,3}дела", r"\bшутк", r"пошут",
        r"философ", r"\bмнени", r"\bкто ты", r"\bпривет", r"\bздравств", r"спасиб",
        r"\bпочему\b", r"объясн", r"\bкак\b", r"\bчто такое", r"\bзачем\b",
        r"\bпоговор", r"\bкак тебя", r"\bтвоё имя", r"\bнравит",
    ),
}
_COMPILED: dict[IntentCategory, re.Pattern[str]] = {
    cat: re.compile("|".join(stems), re.IGNORECASE)
    for cat, stems in _LEXICON.items()
}

# Разговорная ФОРМА (вопрос ИЛИ чит-чат) — используется ТОЛЬКО в fallback'е
# (когда ни один доменный лексикон не сработал), чтобы отличить «поговорить» от
# неузнанной КОМАНДЫ. Чат → CONVERSATION (лёгкий промпт, без execute_bash);
# всё остальное считаем повелительной командой → SYSTEM_OPS (модель не разоружена).
# Vosk даёт текст без пунктуации, поэтому опираемся на зачины/маркеры, не на «?».
_CHITCHAT_RE = re.compile(
    r"\?\s*$"                                                    # знак вопроса (если есть)
    r"|^(?:а\s+|ну\s+|и\s+)?(?:что|как|почему|зачем|кто|когда|где|какой|какая|какие|"
    r"каково|сколько|чей|чь[её]|разве|неужели|правда|можно|можешь|стоит|умеешь|знаешь)\b"
    r"|^в\s+ч[ёе]м\b|^есть\s+ли\b|^существует\s+ли\b"           # «в чём…», «есть ли…»
    r"|\b(?:расскаж|объясн|посоветуй|как\s+думаешь|тво[её]\s+мнение|что\s+думаешь)\b"
    # разговорные маркеры/подтверждения — это НЕ команда на систему → чат:
    r"|\b(?:слышишь|слышиш|видишь|понял|понима|повтори|молодец|спасиб|приве|"
    r"здравству|как\s+тебя|ты\s+(?:тут|там|здесь|жив))\b",
    re.IGNORECASE,
)

# Экземпляры-якоря для embedding-бэкенда (центроиды категорий).
_EXEMPLARS: dict[IntentCategory, tuple[str, ...]] = {
    IntentCategory.SYSTEM_OPS: (
        "установи пакет nginx", "покажи запущенные процессы", "сколько свободно на диске",
        "обнови систему", "убей зависший процесс", "почисти логи и кэш",
    ),
    IntentCategory.UI_CONTROL: (
        "сделай ярче экран", "сверни все окна", "убавь громкость",
        "переключи рабочий стол", "закрой это окно", "сделай скриншот",
    ),
    IntentCategory.PENTEST_RECON: (
        "просканируй сеть nmap", "перехвати трафик в wireshark", "проверь открытые порты",
        "поищи уязвимости на хосте", "запусти разведку по подсети", "анализируй pcap",
    ),
    IntentCategory.CONVERSATION: (
        "расскажи анекдот", "что ты думаешь о свободе воли", "как дела",
        "кто ты такой", "объясни своими словами", "поговорим о философии",
    ),
}


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Результат классификации интента."""
    category: IntentCategory
    score: float                       # уверенность (число совпадений / косинус)
    backend: str                       # "regex" | "embedding" | "fallback"
    matched: tuple[str, ...] = field(default_factory=tuple)


# Тип внешнего эмбеддера: text -> вектор (list[float]).
Embedder = Callable[[str], Sequence[float]]


class IntentRouter:
    """Классифицирует интент и выдаёт микро-промпт + tool-subset.

    Дефолт — regex-бэкенд (быстрый, без сети). Embedding-бэкенд включается явно
    и только при наличии эмбеддера; при любой ошибке эмбеддинга молча падаем
    на regex — маршрутизатор НИКОГДА не блокирует горячий путь."""

    def __init__(
        self,
        core_identity: str | None = None,
        backend: str = "regex",
        embedder: Embedder | None = None,
    ) -> None:
        self.core_identity = (core_identity or SYSTEM_CORE).strip()
        self.backend = backend if backend in ("regex", "embedding") else "regex"
        self._embedder = embedder
        self._centroids: dict[IntentCategory, list[float]] | None = None
        if self.backend == "embedding" and embedder is None:
            log.warning("router backend=embedding без embedder — откат на regex")
            self.backend = "regex"

    # ───────────────────────── публичный API ─────────────────────────
    def route(self, text: str) -> RouteDecision:
        """Классифицировать интент. Никогда не бросает исключений."""
        text = (text or "").strip()
        if not text:
            return RouteDecision(IntentCategory.CONVERSATION, 0.0, "fallback")

        # Системные/служебные интенты от демонов — детерминированно в SYSTEM_OPS,
        # не доверяем их лексикону (там могут быть любые слова из stderr).
        if text.startswith(("[DAEMON_ALERT]", "[SYSTEM_EVENT", "[OS_EVENT]")):
            return RouteDecision(IntentCategory.SYSTEM_OPS, 1.0, "rule")
        if text.startswith(("[RECON_ALERT", "[NMAP")):
            return RouteDecision(IntentCategory.PENTEST_RECON, 1.0, "rule")

        if self.backend == "embedding":
            try:
                return self._route_embedding(text)
            except Exception:
                log.exception("embedding route failed — fallback to regex")
        return self._route_regex(text)

    def system_prompt_for(self, category: IntentCategory | str) -> str:
        """Собрать system-промпт: стабильный core-префикс + правила категории.

        Префикс одинаков всегда → llama-server переиспользует его KV-кэш."""
        try:
            cat = IntentCategory(str(category))
        except ValueError:
            cat = IntentCategory.CONVERSATION
        return f"{self.core_identity}\n\n{_CATEGORY_RULES[cat]}"

    # ───────────────────────── regex backend ─────────────────────────
    @staticmethod
    def _route_regex(text: str) -> RouteDecision:
        best_cat = IntentCategory.CONVERSATION
        best_hits: tuple[str, ...] = ()
        best_score = -1
        for cat in _TIE_BREAK_ORDER:
            hits = tuple(_COMPILED[cat].findall(text))
            score = len(hits)
            # строгое > сохраняет приоритет порядка _TIE_BREAK_ORDER при ничьих
            if score > best_score:
                best_score, best_cat, best_hits = score, cat, hits

        if best_score <= 0:
            # Ни один доменный лексикон не сработал. НЕ падаем слепо в
            # CONVERSATION (это разоружало бы модель — там нет execute_bash).
            # Различаем по форме: чат/вопрос → разговор; команда → действие.
            if _CHITCHAT_RE.search(text):
                return RouteDecision(IntentCategory.CONVERSATION, 0.0, "fallback_chat")
            # Неузнанная команда повелительного наклонения → общий action-набор
            # (SYSTEM_OPS = полный toolset). Устраняет КЛАСС «команда без
            # ключевого слова → чат без инструментов → простыня до таймаута».
            return RouteDecision(IntentCategory.SYSTEM_OPS, 0.0, "fallback_command")
        return RouteDecision(best_cat, float(best_score), "regex", best_hits)

    # ─────────────────────── embedding backend ───────────────────────
    def _ensure_centroids(self) -> dict[IntentCategory, list[float]]:
        if self._centroids is not None:
            return self._centroids
        assert self._embedder is not None
        centroids: dict[IntentCategory, list[float]] = {}
        for cat, phrases in _EXEMPLARS.items():
            vecs = [list(self._embedder(p)) for p in phrases]
            centroids[cat] = _mean_vectors(vecs)
        self._centroids = centroids
        return centroids

    def _route_embedding(self, text: str) -> RouteDecision:
        assert self._embedder is not None
        query = list(self._embedder(text))
        centroids = self._ensure_centroids()
        best_cat = IntentCategory.CONVERSATION
        best_sim = -2.0
        for cat in _TIE_BREAK_ORDER:
            sim = _cosine(query, centroids[cat])
            if sim > best_sim:
                best_sim, best_cat = sim, cat
        return RouteDecision(best_cat, float(best_sim), "embedding")


# ─────────────────────────── векторная арифметика ─────────────────────────────
def _mean_vectors(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return []
    n = len(vecs)
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i in range(min(dim, len(v))):
            out[i] += v[i]
    return [x / n for x in out]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(min(len(a), len(b))):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


__all__ = [
    "IntentCategory",
    "RouteDecision",
    "IntentRouter",
    "SYSTEM_CORE",
]
