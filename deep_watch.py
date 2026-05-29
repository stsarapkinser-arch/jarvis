from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections import deque
from typing import Final

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.deepwatch")

# Inline bpftrace program. Two probes:
#   1. sys_enter_execve   — every new process the kernel actually creates.
#   2. tcp_v4_connect     — every outbound IPv4 TCP connect from user space.
# We filter ourselves (PID==$1) to avoid feedback loops when our own bash spawns.
# Output is single-line, tab-delimited, easy to parse.
BPFTRACE_PROGRAM: Final = r"""
tracepoint:syscalls:sys_enter_execve
/ pid != $1 /
{
    printf("EXEC\t%d\t%s\t%s\n", pid, comm, str(args->filename));
}

kprobe:tcp_v4_connect
/ pid != $1 /
{
    printf("TCP\t%d\t%s\n", pid, comm);
}
"""

# Soft rate-limit per kind to keep the bus quiet under a fork-bomb / SYN-flood.
RATE_WINDOW_SEC: Final = 1.0
RATE_MAX_PER_WINDOW: Final = 80

# Dedup per (kind, comm) so the same noisy daemon doesn't flood.
DEDUP_WINDOW_SEC: Final = 4.0


class DeepWatch(metaclass=Singleton):
    """Ring-0 sensor. Read-only.

    Runs `sudo -n bpftrace` as a child process and parses every line it prints.
    Each line becomes a DEEP_WATCH event on the bus.

    Self-disables gracefully if:
      - bpftrace is not installed,
      - sudo -n requires a password (no NOPASSWD),
      - the kernel verifier rejects the probes (rare on Plasma 6 Kali).
    """

    def __init__(self) -> None:
        self.bus = EventBus()
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._bpftrace = shutil.which("bpftrace")
        self._sudo = shutil.which("sudo")
        self._window_start: float = 0.0
        self._window_count: int = 0
        self._dedup: dict[tuple[str, str], float] = {}
        self._recent: deque[dict] = deque(maxlen=64)

    @property
    def available(self) -> bool:
        return bool(self._bpftrace and self._sudo)

    def recent(self) -> list[dict]:
        return list(self._recent)

    async def _check_sudo_passwordless(self) -> bool:
        if not self._sudo:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                self._sudo, "-n", "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await asyncio.wait_for(proc.wait(), timeout=1.5)
            return rc == 0
        except (TimeoutError, FileNotFoundError):
            return False
        except Exception:
            log.exception("sudo -n probe failed")
            return False

    def _under_rate_limit(self) -> bool:
        now = time.monotonic()
        if now - self._window_start >= RATE_WINDOW_SEC:
            self._window_start = now
            self._window_count = 0
        self._window_count += 1
        return self._window_count <= RATE_MAX_PER_WINDOW

    def _deduped(self, kind: str, comm: str) -> bool:
        key = (kind, comm)
        now = time.monotonic()
        last = self._dedup.get(key, 0.0)
        if now - last < DEDUP_WINDOW_SEC:
            return True
        self._dedup[key] = now
        if len(self._dedup) > 256:
            cutoff = now - DEDUP_WINDOW_SEC * 2
            self._dedup = {k: v for k, v in self._dedup.items() if v >= cutoff}
        return False

    def _parse_line(self, raw: bytes) -> dict | None:
        line = raw.decode(errors="replace").strip()
        if not line or line.startswith("Attaching") or line.startswith("@"):
            return None
        parts = line.split("\t")
        if len(parts) < 3:
            return None
        kind = parts[0]
        if kind not in ("EXEC", "TCP"):
            return None
        try:
            pid = int(parts[1])
        except ValueError:
            return None
        comm = parts[2][:32]
        payload: dict = {"kind": kind.lower(), "pid": pid, "comm": comm, "ts": time.time()}
        if kind == "EXEC" and len(parts) >= 4:
            payload["filename"] = parts[3][:256]
        return payload

    async def _spawn(self) -> asyncio.subprocess.Process | None:
        if not self.available:
            log.info("deep_watch disabled: bpftrace=%s sudo=%s", self._bpftrace, self._sudo)
            return None
        if not await self._check_sudo_passwordless():
            log.info("deep_watch disabled: sudo -n requires password")
            return None

        own_pid = os.getpid()
        program = BPFTRACE_PROGRAM
        cmd = (
            self._sudo, "-n", self._bpftrace,
            "-q",
            "-e", program,
            str(own_pid),
        )
        try:
            return await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None
        except Exception:
            log.exception("bpftrace spawn failed")
            return None

    async def run(self) -> None:
        self._proc = await self._spawn()
        if self._proc is None or self._proc.stdout is None:
            return
        log.info("deep_watch attached (bpftrace pid=%s)", self._proc.pid)
        try:
            while True:
                try:
                    raw = await self._proc.stdout.readline()
                except Exception:
                    log.exception("bpftrace readline failed")
                    await asyncio.sleep(0.3)
                    continue
                if not raw:
                    if self._proc.returncode is not None:
                        log.warning("bpftrace exited rc=%s", self._proc.returncode)
                        break
                    await asyncio.sleep(0.15)
                    continue
                payload = self._parse_line(raw)
                if payload is None:
                    continue
                if not self._under_rate_limit():
                    continue
                if self._deduped(payload["kind"], payload["comm"]):
                    continue
                self._recent.append(payload)
                urgency = "high" if payload["kind"] == "tcp" else "low"
                await self.bus.publish(
                    Event(EventType.DEEP_WATCH, payload, urgency=urgency)
                )
        except asyncio.CancelledError:
            log.info("deep_watch cancelled")
            raise
        finally:
            await self._terminate()

    async def _terminate(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
        # Drain stderr for diagnostics if anything was emitted.
        if self._proc.stderr is not None:
            try:
                tail = await asyncio.wait_for(self._proc.stderr.read(2048), timeout=0.5)
                if tail:
                    log.info("bpftrace stderr: %s", tail.decode(errors="replace")[:512])
            except TimeoutError:
                pass

    async def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="deep-watch")
        return self._task
