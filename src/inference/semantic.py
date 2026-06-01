"""Семантический матч навыка (L2) — намерение → навык через эмбеддинги, без 3B.

Ступень НАД строковыми слоями (L0 точный алиас, L1 шаблоны, L1a резолвер
приложений, L1b fuzzy): ловит ПАРАФРАЗ, у которого нет строкового сходства с
алиасом, но есть смысловое. «запусти браузер», «хочу выйти в интернет» →
``open_browser``, хотя строкой это далеко от алиаса «браузер».

Где в лестнице: ПОСЛЕ дешёвых строковых слоёв и ПЕРЕД 3B. Эмбеддинг фразы — это
HTTP-вызов к embed-серверу (:8090) + CPU-счёт (~десятки мс на N100): дороже
regex/fuzzy (микросекунды), но НА ПОРЯДКИ дешевле декода 3B (секунды). Поэтому
консультируемся семантикой только когда строковые слои промахнулись.

Безопасность и сдержанность:
  * В индекс берём ТОЛЬКО безаргументные, неразрушительные навыки с алиасами:
    семантика даёт НАМЕРЕНИЕ, а не СЛОТ — для параметрических навыков (nmap,
    set_volume) пустой вызов бесполезен/опасен, их ловит L1.
  * Порог косинуса + ЗАЗОР над вторым кандидатом + группировка по навыку —
    как в L1b, против ложных срабатываний. Неуверенность → None → уходит 3B.
  * Никогда не бросает: нет embedder'а / сервер лёг / любая ошибка → None
    (горячий путь не блокируется). Сборка экземпляров — с backoff при сбое.

ВАЖНО (калибровка): порог семантически зависит от embedding-МОДЕЛИ. Дефолт ниже
— заведомо консервативный плейсхолдер; перед включением (``JARVIS_SEMANTIC_MATCH``)
откалибруйте его на железе: ``python scripts/calibrate_semantic.py`` гоняет
golden-набор и печатает порог, дающий максимум recall при НУЛЕ ложных приёмов.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

log = logging.getLogger("jarvis.semantic")

# batch-эмбеддер: список текстов → список векторов (тот же порядок).
EmbedFn = Callable[[Sequence[str]], "list[list[float]]"]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


# Консервативные плейсхолдеры — КАЛИБРУЙТЕ на целевой embedding-модели.
DEFAULT_THRESHOLD = _env_float("JARVIS_SEMANTIC_THRESHOLD", 0.62)
DEFAULT_MARGIN = _env_float("JARVIS_SEMANTIC_MARGIN", 0.05)
# Пауза перед повторной сборкой экземпляров, если embed-сервер был недоступен
# (на старте поднимается дольше мозга). На горячий путь не влияет — сборка ленивая.
_BUILD_BACKOFF_SEC = 60.0


def _e5_prefixes() -> tuple[str, str]:
    """(query-префикс, passage-префикс) для асимметричного матча.

    Семейство **e5** (наш дефолт multilingual-e5-small) ОБУЧЕНО на префиксах
    ``query: `` / ``passage: ``: без них близость коротких фраз заметно хуже —
    парафраз не дотягивает до уверенного матча (в калибровке это видно как плоско
    низкий recall на всех порогах). Префиксуем экземпляры как passage, запрос —
    как query. Автодетект по имени модели; ``JARVIS_SEMANTIC_E5_PREFIX=0/1`` —
    принудительно. Не-e5 модель → пустые префиксы (поведение прежнее)."""
    env = os.getenv("JARVIS_SEMANTIC_E5_PREFIX", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return ("", "")
    if env in ("1", "true", "yes", "on"):
        return ("query: ", "passage: ")
    model = os.getenv("JARVIS_EMBED_MODEL", "multilingual-e5-small").lower()
    return ("query: ", "passage: ") if "e5" in model else ("", "")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for i in range(min(len(a), len(b))):
        x, y = a[i], b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _apply(rank: "list[tuple[str, float]]", threshold: float, margin: float) -> str | None:
    """Решение по ранжированному списку (skill_id, cosine) с порогом и зазором.

    Вынесено отдельно, чтобы и рантайм (``match``), и калибровка применяли
    ИДЕНТИЧНОЕ правило к одним и тем же score'ам."""
    if not rank:
        return None
    best_id, best = rank[0]
    if best < threshold:
        return None
    if len(rank) > 1 and best - rank[1][1] < margin:
        return None  # неоднозначно — отдаём 3B
    return best_id


def collect_exemplars() -> dict[str, list[str]]:
    """Per-skill экземпляры из реестра: алиасы + описание.

    Берём только навыки, пригодные для семантического АВТО-вызова: без слотов
    (params пуст), неразрушительные, с алиасами. Их id и есть индекс L2."""
    from src import skills
    out: dict[str, list[str]] = {}
    for sk in skills.all_skills():
        if sk.destructive or sk.params or not sk.aliases:
            continue
        phrases = list(sk.aliases)
        if sk.description:
            phrases.append(sk.description)
        out[sk.id] = phrases
    return out


def indexable_skill_ids() -> set[str]:
    """Множество навыков, которые L2 МОЖЕТ авто-вызвать (для тестов/калибровки)."""
    return set(collect_exemplars().keys())


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Строка отчёта калибровки для одного порога."""
    threshold: float
    correct: int        # позитив → верный навык
    missed: int         # позитив → None (не дотянул)
    wrong: int          # позитив → ЧУЖОЙ навык (опасно)
    false_accept: int   # негатив → какой-то навык (должен был уйти 3B)
    n_pos: int
    n_neg: int

    @property
    def recall(self) -> float:
        return self.correct / self.n_pos if self.n_pos else 0.0

    @property
    def clean(self) -> bool:
        """Идеально безопасный порог: ноль чужих матчей и ноль ложных приёмов."""
        return self.wrong == 0 and self.false_accept == 0


class SemanticSkillMatcher:
    """Сопоставляет фразу с навыком по близости эмбеддингов. Никогда не бросает.

    ``embed_fn`` инъектируется (EmbeddingClient.embed в проде, фейк в тестах) —
    модуль развязан с транспортом и калибруется на синтетике."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        *,
        exemplars: Mapping[str, Sequence[str]] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._exemplars = exemplars
        self.threshold = threshold
        self.margin = margin
        # e5-префиксы (query/passage). Берём из env, если не заданы явно — так и
        # рантайм, и калибровка применяют ОДНО правило к одной модели.
        _qp, _pp = _e5_prefixes()
        self._qpref = query_prefix if query_prefix is not None else _qp
        self._ppref = passage_prefix if passage_prefix is not None else _pp
        self._index: list[tuple[str, list[float]]] = []
        self._available = True
        self._next_build_at = 0.0

    # ───────────────────────── сборка экземпляров ─────────────────────────
    def _ensure_index(self) -> None:
        if self._index or not self._available:
            return
        if time.monotonic() < self._next_build_at:
            return
        exemplars = self._exemplars if self._exemplars is not None else collect_exemplars()
        ids: list[str] = []
        texts: list[str] = []
        for sid, phrases in exemplars.items():
            for phrase in phrases:
                ids.append(sid)
                texts.append(self._ppref + phrase)   # экземпляр = passage
        if not texts:
            self._available = False
            return
        try:
            vecs = self._embed_fn(texts)
        except Exception:
            log.warning("semantic: сборка экземпляров не удалась — повтор через %.0fс",
                        _BUILD_BACKOFF_SEC, exc_info=True)
            self._next_build_at = time.monotonic() + _BUILD_BACKOFF_SEC
            return
        self._index = list(zip(ids, vecs))
        log.info("semantic: индекс собран — %d экземпляров, %d навыков",
                 len(self._index), len(set(ids)))

    # ───────────────────────── матчинг ─────────────────────────
    def rank(self, text: str) -> list[tuple[str, float]]:
        """(skill_id, cosine) по убыванию — лучший косинус НА НАВЫК. [] при сбое."""
        if not self._available:
            return []
        self._ensure_index()
        if not self._index:
            return []
        try:
            qv = self._embed_fn([self._qpref + text])[0]   # запрос = query
        except Exception:
            log.debug("semantic: эмбеддинг запроса не удался", exc_info=True)
            return []
        best: dict[str, float] = {}
        for sid, vec in self._index:
            c = _cosine(qv, vec)
            if c > best.get(sid, -2.0):
                best[sid] = c
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def match(self, text: str) -> str | None:
        """skill_id, если уверенно и однозначно, иначе None (→ 3B)."""
        return _apply(self.rank(text), self.threshold, self.margin)


def evaluate(
    embed_fn: EmbedFn,
    positives: Sequence[tuple[str, str]],
    negatives: Sequence[str],
    *,
    thresholds: Sequence[float],
    margin: float = DEFAULT_MARGIN,
    exemplars: Mapping[str, Sequence[str]] | None = None,
) -> list[ThresholdResult]:
    """Свип порогов на golden-наборе. ``positives`` = (фраза, ожидаемый skill_id);
    ``negatives`` = фразы, которые ДОЛЖНЫ уйти 3B. Ранги считаем один раз, порог
    применяем post-hoc — дешёвый честный свип. Используется и скриптом
    калибровки, и тестом (на фейк-эмбеддере)."""
    matcher = SemanticSkillMatcher(embed_fn, exemplars=exemplars, margin=margin, threshold=0.0)
    pos_ranks = [(expected, matcher.rank(phrase)) for phrase, expected in positives]
    neg_ranks = [matcher.rank(phrase) for phrase in negatives]
    out: list[ThresholdResult] = []
    for th in thresholds:
        correct = missed = wrong = 0
        for expected, rank in pos_ranks:
            pred = _apply(rank, th, margin)
            if pred == expected:
                correct += 1
            elif pred is None:
                missed += 1
            else:
                wrong += 1
        false_accept = sum(1 for rank in neg_ranks if _apply(rank, th, margin) is not None)
        out.append(ThresholdResult(
            threshold=th, correct=correct, missed=missed, wrong=wrong,
            false_accept=false_accept, n_pos=len(positives), n_neg=len(negatives),
        ))
    return out


__all__ = [
    "SemanticSkillMatcher",
    "ThresholdResult",
    "EmbedFn",
    "collect_exemplars",
    "indexable_skill_ids",
    "evaluate",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MARGIN",
]
