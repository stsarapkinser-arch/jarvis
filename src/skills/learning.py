"""SkillLearner — наблюдатель, превращающий привычку в навык.

Когда gated-fallback ``execute_bash`` успешно (rc=0, не разрушительно) исполняет
одну и ту же команду ``PROMOTE_THRESHOLD`` раз, учитель выдаёт ``Candidate`` —
предложение закрепить её как постоянный навык. Оркестратор спрашивает оператора
голосом; на «да» вызывается ``promote`` → запись в ``config/learned_skills.json``
→ горячая перезагрузка регистрирует навык (см. ``src/skills/learned.py``).

Счётчики использования живут в ``jarvis_memory/`` (НЕ под наблюдением
FileChangeWatcher), поэтому переживают os.execv-перезагрузки и не дёргают reload
сами по себе. Чистый, тестируемый слой: вся логика порога/слага/записи —
синхронная и без побочных эффектов сверх одного JSON-файла.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.inference.router import IntentCategory
from src.skills import learned
from src.skills.registry import get as _get

log = logging.getLogger("jarvis.skills.learning")

# Сколько успешных повторов одной команды до предложения закрепить её навыком.
PROMOTE_THRESHOLD = 3
# Не учим тривиальное (echo/ls в одно слово) и сверхдлинное (вряд ли повторяемое).
_MIN_CMD_LEN = 3
_MAX_CMD_LEN = 400

_USAGE_PATH_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent / "jarvis_memory" / "skill_usage.json"
)

_SLUG_RE = re.compile(r"[^a-zа-я0-9]+", re.IGNORECASE)
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def _canonical(command: str) -> str:
    """Канонизировать команду как ключ счётчика: схлопнуть пробелы, обрезать."""
    return re.sub(r"\s+", " ", (command or "").strip())


def _slug(intent: str) -> str:
    """Человеко-понятный id-фрагмент из интента (транслит → латиница, _)."""
    base = (intent or "").strip().lower().translate(_TRANSLIT)
    slug = _SLUG_RE.sub("_", base).strip("_")
    return slug[:32] or "skill"


@dataclass(frozen=True, slots=True)
class Candidate:
    """Предложение закрепить команду навыком."""
    skill_id: str
    command: str
    kind: str            # "run" | "spawn"
    intent: str
    count: int
    description: str

    def to_entry(self) -> dict[str, object]:
        """Запись для config/learned_skills.json (формат learned.py)."""
        return {
            "id": self.skill_id,
            "category": IntentCategory.SYSTEM_OPS.value,
            "description": self.description,
            "command": self.command,
            "kind": self.kind,
            "aliases": [],
        }


class SkillLearner:
    """Счётчик успешных fallback-команд + генератор предложений."""

    def __init__(self, usage_path: Path | None = None, threshold: int = PROMOTE_THRESHOLD) -> None:
        self._path = usage_path or _USAGE_PATH_DEFAULT
        self._threshold = threshold
        self._usage: dict[str, dict] = self._load()

    # ───────────────────────── персистентность ─────────────────────────
    def _load(self) -> dict[str, dict]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, TypeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._usage, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            log.exception("сохранение счётчиков %s не удалось", self._path)

    # ───────────────────────── наблюдение ─────────────────────────
    @staticmethod
    def _learnable(command: str, destructive: bool) -> bool:
        cmd = _canonical(command)
        if destructive:
            return False
        if not (_MIN_CMD_LEN <= len(cmd) <= _MAX_CMD_LEN):
            return False
        # sudo/секреты не кэшируем как «готовый навык» — пусть проходят гейты.
        if cmd.startswith(("sudo ", "sudo-n ")) or "sudo -n" in cmd:
            return False
        return True

    def observe(
        self,
        intent: str,
        command: str,
        *,
        destructive: bool,
        kind: str = "run",
    ) -> Candidate | None:
        """Учесть успешный fallback. Вернуть Candidate, когда команда дозрела.

        Возвращает предложение РОВНО один раз — при пересечении порога; далее
        запись помечается ``proposed`` и больше не предлагается (до промоушена
        или ручной чистки счётчиков)."""
        if not self._learnable(command, destructive):
            return None
        key = _canonical(command)
        rec = self._usage.setdefault(
            key, {"count": 0, "intent": intent, "proposed": False, "learned": False}
        )
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["last_ts"] = time.time()
        if intent and not rec.get("intent"):
            rec["intent"] = intent
        self._save()

        if rec["count"] < self._threshold or rec.get("proposed") or rec.get("learned"):
            return None
        rec["proposed"] = True
        self._save()
        return self._candidate(key, rec, kind)

    def _candidate(self, command: str, rec: dict, kind: str) -> Candidate:
        intent = str(rec.get("intent") or command)
        sid = self._unique_id(_slug(intent))
        desc = intent.strip()[:80] or command[:80]
        return Candidate(
            skill_id=sid, command=command, kind=kind,
            intent=intent, count=int(rec["count"]), description=desc,
        )

    @staticmethod
    def _unique_id(slug: str) -> str:
        base = f"{learned.LEARNED_PREFIX}{slug}"
        if _get(base) is None:
            return base
        for i in range(2, 100):
            cand = f"{base}_{i}"
            if _get(cand) is None:
                return cand
        return f"{base}_{int(time.time())}"

    # ───────────────────────── промоушен ─────────────────────────
    def promote(self, candidate: Candidate) -> bool:
        """Записать кандидата в каталог выученных навыков (после голосового «да»).

        Помечает счётчик как ``learned``, чтобы не предлагать снова. Реальную
        регистрацию выполнит ``learned.load_learned`` на горячей перезагрузке,
        которую триггерит запись в config/."""
        ok = learned.append_learned(candidate.to_entry())
        key = _canonical(candidate.command)
        if key in self._usage:
            self._usage[key]["learned"] = ok
            self._save()
        return ok
