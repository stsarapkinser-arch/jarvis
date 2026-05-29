"""Jarvis immortality watchdog: auto-restart on crash, hot reload on code changes.

This module wraps the main bootstrap in a watchdog loop. If the event loop crashes
or any subsystem fails catastrophically, the watchdog:

1. Detects the crash (exception or event loop exit)
2. Waits for a backoff period (exponential: 2s, 4s, 8s, 16s, then 30s)
3. Attempts to git pull (reload code from repo)
4. Restarts the entire Jarvis event loop

Additionally, a file watcher detects changes in src/ and config/ and triggers
hot reloads of modified modules (importlib.reload).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import importlib
from pathlib import Path
from typing import Callable, Any

log = logging.getLogger("jarvis.immortal")

RESTART_BACKOFF_SEQUENCE = [2, 4, 8, 16, 30]  # exponential backoff in seconds
FILE_WATCH_DEBOUNCE_SEC = 1.0


class ImmortalWatchdog:
    """Wraps amain() in a crash-resistant watchdog loop."""

    def __init__(self, amain_coro: Callable[[], Any]) -> None:
        self.amain_coro = amain_coro
        self.restart_count = 0
        self.last_crash_time = 0.0
        self._file_watcher_task: asyncio.Task | None = None

    async def run(self) -> None:
        """Run the main event loop with auto-restart on crash."""
        backoff_idx = 0
        while True:
            try:
                log.info(f"Jarvis startup (restart #{self.restart_count})")
                await self.amain_coro()
                # If amain() returns normally, that's unexpected
                log.warning("amain() returned normally — restarting")
            except KeyboardInterrupt:
                log.info("Keyboard interrupt — shutting down")
                break
            except Exception:
                log.exception("Jarvis crashed")
                self.restart_count += 1

            # Exponential backoff before restart
            backoff = RESTART_BACKOFF_SEQUENCE[min(backoff_idx, len(RESTART_BACKOFF_SEQUENCE) - 1)]
            log.warning(f"Restarting in {backoff}s (backoff level {backoff_idx})")
            await asyncio.sleep(backoff)
            backoff_idx = min(backoff_idx + 1, len(RESTART_BACKOFF_SEQUENCE) - 1)

            # Attempt git pull before restart
            try:
                await self._attempt_git_pull()
            except Exception:
                log.exception("git pull failed — continuing with current code")

            self.last_crash_time = time.time()

    async def _attempt_git_pull(self) -> None:
        """Try to pull latest code from git. Silent failure — don't block restart."""
        repo_root = Path(__file__).resolve().parents[2]  # /home/user/jarvis
        if not (repo_root / ".git").is_dir():
            log.debug("Not a git repo — skipping pull")
            return

        try:
            # Run git pull in subprocess
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "pull", "--rebase"],
                cwd=str(repo_root),
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                log.info(f"git pull successful: {result.stdout.decode().strip()}")
            else:
                log.warning(f"git pull failed: {result.stderr.decode().strip()}")
        except subprocess.TimeoutExpired:
            log.warning("git pull timed out")
        except Exception as e:
            log.debug(f"git pull exception: {e}")


class FileChangeWatcher:
    """Watches src/ and config/ for file changes, triggers hot reload."""

    def __init__(self) -> None:
        self.watch_dirs = [
            Path(__file__).resolve().parents[2] / "src",
            Path(__file__).resolve().parents[2] / "config",
        ]
        self.last_seen_mtime: dict[Path, float] = {}
        self._debounce_handle: asyncio.TimerHandle | None = None

    async def start(self) -> None:
        """Start the file watcher task."""
        while True:
            try:
                await self._check_for_changes()
            except Exception:
                log.exception("File watcher check failed")
            await asyncio.sleep(0.5)

    async def _check_for_changes(self) -> None:
        """Scan watched directories for modified files."""
        changed_files: set[Path] = set()

        for watch_dir in self.watch_dirs:
            if not watch_dir.is_dir():
                continue
            for py_file in watch_dir.rglob("*.py"):
                if py_file.is_file():
                    try:
                        mtime = py_file.stat().st_mtime
                        last = self.last_seen_mtime.get(py_file, 0.0)
                        if mtime > last:
                            changed_files.add(py_file)
                            self.last_seen_mtime[py_file] = mtime
                    except OSError:
                        pass

        if changed_files:
            log.info(f"Detected {len(changed_files)} changed files: {[f.name for f in changed_files]}")
            # Debounce: wait a bit for the editor to finish writing
            if self._debounce_handle:
                self._debounce_handle.cancel()
            self._debounce_handle = asyncio.get_event_loop().call_later(
                FILE_WATCH_DEBOUNCE_SEC,
                lambda: asyncio.create_task(self._hot_reload(changed_files))
            )

    async def _hot_reload(self, changed_files: set[Path]) -> None:
        """Attempt to reload changed modules."""
        for py_file in changed_files:
            try:
                rel_path = py_file.relative_to(Path(__file__).resolve().parents[2])
                # Convert path to module name: src/ui/hud.py -> src.ui.hud
                module_name = str(rel_path).replace("/", ".").replace(".py", "")
                if module_name in sys.modules:
                    log.info(f"Hot reloading {module_name}")
                    importlib.reload(sys.modules[module_name])
                else:
                    log.debug(f"Module {module_name} not loaded yet — skipping")
            except Exception:
                log.exception(f"Failed to hot reload {py_file}")
