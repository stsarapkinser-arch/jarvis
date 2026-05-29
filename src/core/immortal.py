"""Бессмертие Jarvis, слой 1 из 2 — горячая перезагрузка кода в самом процессе.

Архитектура бессмертия двухслойная:

  • Слой 1 (этот файл, ``FileChangeWatcher``) — живёт ВНУТРИ процесса
    bootstrap. Следит за ``src/`` и ``config/``; как только вы отредактировали
    файл или сделали ``git pull``, процесс делает ``os.execv`` — перезапускает
    сам себя с тем же PID и свежим кодом. Все изменения подхватываются
    автоматически, вручную перезапускать ничего не нужно.

  • Слой 2 (``src/core/supervisor.py``) — ВНЕШНИЙ супервизор-демон. Он
    переживает падение/убийство/Ctrl+C процесса bootstrap и поднимает его
    заново. Именно он делает так, что «остановка python -m src.core.bootstrap
    не убивает Джарвиса».

Почему ``os.execv``, а не ``importlib.reload``? Потому что Jarvis — это живое
приложение Qt + asyncio + потоки. ``importlib.reload`` оставляет половину
объектов привязанной к старым классам (Qt-слоты, замыкания, синглтоны) —
получается полусломанное состояние. ``os.execv`` же даёт честный чистый
старт нового кода, ничего не теряя: тот же PID, те же дескрипторы, мгновенно.
Сервер инференса llama-cpp живёт в ОТДЕЛЬНОМ процессе, поэтому при re-exec
он не перезагружается — модель остаётся «тёплой» в RAM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("jarvis.immortal")

# Код выхода, по которому супервизор понимает «это плановая перезагрузка кода,
# поднимай немедленно без backoff». Любой другой код — это краш.
RELOAD_EXIT_CODE = 42

# Сколько ждать после первого замеченного изменения, прежде чем перезапуститься
# (редактор/`git pull` могут писать несколько файлов подряд — собираем пачку).
FILE_WATCH_DEBOUNCE_SEC = 1.2
# Период опроса mtime файлов.
FILE_WATCH_POLL_SEC = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[2]


def reexec(reason: str = "code change") -> None:
    """Перезапустить текущий процесс на месте свежим кодом.

    Под супервизором/systemd выходим с ``RELOAD_EXIT_CODE`` — там поднимут
    мгновенно. Без супервизора делаем ``os.execv`` — сами заменяем образ
    процесса (тот же PID), что тоже подхватывает новый код без участия
    оператора."""
    log.warning("Перезапуск Jarvis (%s) — подхватываю новый код", reason)
    sys.stdout.flush()
    sys.stderr.flush()

    if os.environ.get("JARVIS_SUPERVISED") == "1":
        # Супервизор ждёт нас и поднимет немедленно с новым кодом.
        os._exit(RELOAD_EXIT_CODE)

    # Автономный режим: заменяем образ процесса на свежий интерпретатор.
    # cwd уже корень репозитория, ``-m src.core.bootstrap`` запустит заново.
    try:
        os.execv(sys.executable, [sys.executable, "-m", "src.core.bootstrap"])
    except Exception:
        log.exception("os.execv не удался — выходим с RELOAD_EXIT_CODE")
        os._exit(RELOAD_EXIT_CODE)


class FileChangeWatcher:
    """Следит за ``src/`` и ``config/``; на изменение .py/конфигов перезапускает
    процесс свежим кодом (см. :func:`reexec`).

    Опрос по mtime — без внешних зависимостей (watchdog/inotify не нужны),
    надёжно работает и под Wayland, и в headless-CI."""

    WATCH_SUFFIXES = (".py", ".toml", ".service", "system_prompt")

    def __init__(self, on_reload: Callable[[str], None] | None = None) -> None:
        self.watch_dirs = [_REPO_ROOT / "src", _REPO_ROOT / "config"]
        self._on_reload = on_reload or reexec
        self._mtimes: dict[Path, float] = {}
        self._pending_since: float | None = None
        self._armed = False  # первый проход только снимает baseline, не триггерит

    def _iter_files(self):
        for d in self.watch_dirs:
            if not d.is_dir():
                continue
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                if p.name == "system_prompt" or p.suffix in self.WATCH_SUFFIXES:
                    yield p

    def _scan(self) -> set[Path]:
        """Вернуть множество изменившихся с прошлого скана файлов."""
        changed: set[Path] = set()
        seen: set[Path] = set()
        for p in self._iter_files():
            seen.add(p)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            prev = self._mtimes.get(p)
            if prev is None:
                self._mtimes[p] = mtime
                if self._armed:
                    changed.add(p)  # новый файл появился
            elif mtime > prev:
                self._mtimes[p] = mtime
                changed.add(p)
        # Удалённые файлы тоже считаем изменением.
        removed = set(self._mtimes) - seen
        for p in removed:
            self._mtimes.pop(p, None)
            if self._armed:
                changed.add(p)
        return changed

    async def start(self) -> None:
        """Запустить цикл наблюдения (как asyncio-таск)."""
        log.info("FileChangeWatcher: слежу за %s",
                 ", ".join(str(d) for d in self.watch_dirs))
        # Первый проход — снимаем baseline mtime, не реагируем.
        self._scan()
        self._armed = True
        while True:
            try:
                changed = self._scan()
                if changed:
                    if self._pending_since is None:
                        self._pending_since = time.monotonic()
                        log.info("Замечены изменения: %s",
                                 ", ".join(sorted(p.name for p in changed)))
                    else:
                        # пришли ещё файлы — продлеваем debounce
                        self._pending_since = time.monotonic()
                elif self._pending_since is not None:
                    if time.monotonic() - self._pending_since >= FILE_WATCH_DEBOUNCE_SEC:
                        self._pending_since = None
                        self._on_reload("изменены файлы проекта")
            except Exception:
                log.exception("FileChangeWatcher: ошибка скана")
            await asyncio.sleep(FILE_WATCH_POLL_SEC)
