"""Самолечение команд навыков — автоматизация ручного «# VERIFY on target».

Боль оператора: встроенный навык систематически падает, потому что DBus-имя
бинаря/интерфейса различается между версиями Plasma (qdbus6 vs qdbus vs
qdbus-qt6). Раньше это правилось руками («поправь ОДНУ строку»). Теперь — сам:

  1. ``_SkillCtx.run`` гонит команду навыка;
  2. если она упала (rc≠0), оркестратор просит у ``SkillHealer`` детерминированных
     кандидатов-замен (варианты бинаря qdbus) и пробует их по очереди;
  3. первый, давший rc=0 — это и есть проверка «на целевой машине» (VERIFY),
     только автоматическая. Рабочую замену запоминаем (override-стор), и впредь
     навык сразу идёт по ней.

Стор персистится в ``jarvis_memory/skill_overrides.json`` — per-host runtime
state, НЕ под наблюдением FileChangeWatcher (как и счётчики learner'а), чтобы
запись не дёрнула горячую перезагрузку. Кандидаты ДЕТЕРМИНИРОВАНЫ (никакого LLM
на горячем пути): мы лишь подменяем имя бинаря из известного семейства qdbus —
безопасно, быстро, предсказуемо. Само исполнение кандидата (rc=0) и есть гарантия
корректности, поэтому healer ничего не «выдумывает».
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("jarvis.heal")

_STORE_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent / "jarvis_memory" / "skill_overrides.json"
)

# Семейство DBus-CLI бинарей KDE. Между Plasma 5/6 и сборками Qt имя плавает:
# qdbus6 (Plasma 6), qdbus (generic), qdbus-qt6/qdbus-qt5 (раздельные пакеты).
_QDBUS_BINARIES: tuple[str, ...] = ("qdbus6", "qdbus", "qdbus-qt6", "qdbus-qt5")
_QDBUS_RE = re.compile(r"\bqdbus(?:6|-qt6|-qt5)?\b")

_WS_RE = re.compile(r"\s+")


def _normalize(cmd: str) -> str:
    """Ключ стора: схлопнутые пробелы, без крайних — устойчив к форматированию."""
    return _WS_RE.sub(" ", (cmd or "").strip())


def first_binary(cmd: str) -> str:
    """Первый «значимый» токен команды (имя бинаря) — для понятной озвучки."""
    for tok in _normalize(cmd).split():
        if tok and not tok.startswith("-"):
            return tok.split("/")[-1]
    return ""


class SkillHealer:
    """Стор выученных замен + генератор детерминированных кандидатов-починок."""

    def __init__(self, store_path: Path | None = None) -> None:
        self._path = store_path or _STORE_DEFAULT
        self._overrides: dict[str, str] = {}
        self._load()

    # ───────────────────────── persistence ─────────────────────────
    def _load(self) -> None:
        try:
            if self._path.is_file():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._overrides = {str(k): str(v) for k, v in data.items()}
        except Exception:
            log.exception("не удалось прочитать override-стор %s", self._path)
            self._overrides = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._overrides, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            log.exception("не удалось сохранить override-стор %s", self._path)

    # ───────────────────────── public API ─────────────────────────
    def override_for(self, cmd: str) -> str | None:
        """Выученная рабочая замена для команды, либо None."""
        return self._overrides.get(_normalize(cmd))

    def remember(self, cmd: str, healed: str) -> None:
        """Запомнить, что ``cmd`` следует исполнять как ``healed`` (verified rc=0)."""
        key = _normalize(cmd)
        val = _normalize(healed)
        if not key or not val or key == val:
            return
        self._overrides[key] = val
        self._save()

    def forget(self, cmd: str) -> None:
        """Забыть замену (например, она внезапно перестала работать)."""
        if self._overrides.pop(_normalize(cmd), None) is not None:
            self._save()

    def candidates(self, cmd: str, stderr: str = "") -> list[str]:
        """Детерминированные кандидаты-починки для упавшей команды.

        Сейчас — подмена имени бинаря из семейства qdbus (каждый вариант на ВСЕ
        вхождения). Это покрывает основной класс «# VERIFY on target»: команда
        верна по сути, но бинарь называется иначе в этой версии Plasma. Само
        исполнение отфильтрует отсутствующие бинари (command not found → rc≠0)."""
        out: list[str] = []
        if _QDBUS_RE.search(cmd):
            for binary in _QDBUS_BINARIES:
                cand = _QDBUS_RE.sub(binary, cmd)
                if cand != _normalize(cmd) and cand != cmd and cand not in out:
                    out.append(cand)
        return out


__all__ = ["SkillHealer", "first_binary"]
