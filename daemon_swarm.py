from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.swarm")

ALERT_RE = re.compile(
    r"\b(thermal|overheat|temperature|docker|fail|failed|attack|denied|invalid user|panic|oom|segfault)\b",
    re.IGNORECASE,
)


class DaemonSwarm(metaclass=Singleton):
    """Background observers that publish events to the bus."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self._qdbus: str | None = shutil.which("qdbus6") or shutil.which("qdbus")
        self._last_clipboard: dict[str, str] = {}

    async def watch_journal(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "journalctl", "-f", "-p", "3", "--no-pager", "-n", "0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("journalctl not found; journal watcher disabled")
            return

        assert proc.stdout is not None
        log.info("journal watcher started")
        try:
            while True:
                try:
                    raw = await proc.stdout.readline()
                except Exception:
                    log.exception("journal readline failed")
                    await asyncio.sleep(0.5)
                    continue
                if not raw:
                    await asyncio.sleep(0.2)
                    continue
                line = raw.decode(errors="replace").strip()
                if not line or not ALERT_RE.search(line):
                    continue
                await self.bus.publish(
                    Event(EventType.DAEMON_ALERT, line, urgency="high")
                )
        except asyncio.CancelledError:
            log.info("journal watcher cancelled")
            raise
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    async def _kdeconnect_devices(self) -> list[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "kdeconnect-cli", "-l", "--id-only",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            return [ln for ln in stdout.decode().strip().splitlines() if ln]
        except (FileNotFoundError, asyncio.TimeoutError):
            return []
        except Exception:
            log.exception("kdeconnect-cli failed")
            return []

    async def _qdbus_clipboard(self, dev_id: str) -> str | None:
        if not self._qdbus:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self._qdbus, "org.kde.kdeconnect",
                f"/modules/kdeconnect/devices/{dev_id}/clipboard", "getClipboard",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            if proc.returncode != 0:
                return None
            return stdout.decode().strip()
        except (FileNotFoundError, asyncio.TimeoutError):
            return None
        except Exception:
            log.exception("qdbus clipboard probe failed")
            return None

    async def watch_pixel(self) -> None:
        log.info("pixel watcher started")
        try:
            while True:
                for dev_id in await self._kdeconnect_devices():
                    cb = await self._qdbus_clipboard(dev_id)
                    if cb is None:
                        continue
                    prev = self._last_clipboard.get(dev_id)
                    if prev is not None and cb and cb != prev:
                        await self.bus.publish(
                            Event(
                                EventType.PIXEL_EVENT,
                                {"kind": "clipboard", "device": dev_id, "payload": cb},
                            )
                        )
                    self._last_clipboard[dev_id] = cb
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            log.info("pixel watcher cancelled")
            raise

    async def start_all(self) -> list[asyncio.Task]:
        return [
            asyncio.create_task(self.watch_journal(), name="journal-watcher"),
            asyncio.create_task(self.watch_pixel(), name="pixel-watcher"),
        ]
