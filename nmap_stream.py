"""Streaming wrapper around ``nmap`` for the HUD port matrix.

When the core dispatches a command that starts with ``nmap`` we hand it to
``stream_nmap`` instead of the standard ``run_bash``. The wrapper:

  * spawns nmap with line-buffered output (``--stats-every 2s``);
  * parses each line into ``PortHit`` / ``HostHit`` records;
  * pushes them as ``NMAP_SCAN`` events so the HUD can light up its port
    matrix in real time;
  * still returns the full ``(rc, stdout, stderr)`` triple at the end so
    ``Jarvis._execute_with_healing`` can heal/retry as usual.

Output goes through nice so a long ``-A`` sweep does not steal cycles from
Ollama or KWin compositing.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass

from event_bus import Event, EventBus, EventType

log = logging.getLogger("jarvis.nmap")

NMAP_NICE_LEVEL = 10  # gentler than Wraith — nmap is user-driven.

PORT_LINE_RE = re.compile(
    r"^(?P<port>\d{1,5})/(?P<proto>tcp|udp)\s+"
    r"(?P<state>open|closed|filtered|unfiltered|open\|filtered|closed\|filtered)\s+"
    r"(?P<service>\S+)",
    re.IGNORECASE,
)
HOST_LINE_RE = re.compile(
    r"^Nmap scan report for\s+(?P<host>\S+?)(?:\s+\((?P<ip>[0-9a-fA-F.:]+)\))?\s*$"
)
PROGRESS_RE = re.compile(r"About\s+(?P<pct>\d+(?:\.\d+)?)%\s+done")


@dataclass(frozen=True)
class PortHit:
    host: str
    port: int
    proto: str
    state: str
    service: str
    ts: float

    def as_dict(self) -> dict:
        return {
            "host": self.host, "port": self.port, "proto": self.proto,
            "state": self.state, "service": self.service, "ts": self.ts,
        }


@dataclass(frozen=True)
class HostHit:
    host: str
    ip: str | None
    ts: float

    def as_dict(self) -> dict:
        return {"host": self.host, "ip": self.ip, "ts": self.ts}


def parse_nmap_line(line: str, current_host: str) -> tuple[str | None, dict | None]:
    """Inspect a single nmap output line.

    Returns ``(event_kind, payload)``. ``event_kind`` is one of
    ``"host"``, ``"port"``, ``"progress"`` or None.
    """
    if not line:
        return None, None
    line = line.rstrip()
    now = time.time()

    host_m = HOST_LINE_RE.match(line)
    if host_m:
        host = host_m.group("host")
        return "host", HostHit(host=host, ip=host_m.group("ip"), ts=now).as_dict()

    port_m = PORT_LINE_RE.match(line)
    if port_m:
        port = int(port_m.group("port"))
        hit = PortHit(
            host=current_host or "unknown",
            port=port,
            proto=port_m.group("proto").lower(),
            state=port_m.group("state").lower(),
            service=port_m.group("service"),
            ts=now,
        )
        return "port", hit.as_dict()

    prog_m = PROGRESS_RE.search(line)
    if prog_m:
        return "progress", {"percent": float(prog_m.group("pct")), "ts": now}

    return None, None


def is_nmap_command(cmd: str) -> bool:
    """Recognise commands that should be tee'd into the HUD overlay."""
    if not cmd:
        return False
    stripped = cmd.lstrip()
    if stripped.startswith("sudo "):
        stripped = stripped[5:].lstrip()
        if stripped.startswith("-n "):
            stripped = stripped[3:].lstrip()
    head = stripped.split(None, 1)[0] if stripped else ""
    return head == "nmap"


def _wrap_nice(cmd_list: Iterable[str], nice_level: int) -> list[str]:
    """Prefix with `nice -n N` when available; otherwise return cmd untouched."""
    nice_bin = shutil.which("nice")
    if not nice_bin:
        return list(cmd_list)
    return [nice_bin, "-n", str(nice_level), *cmd_list]


async def stream_nmap(
    cmd: str,
    bus: EventBus | None = None,
    nice_level: int = NMAP_NICE_LEVEL,
    timeout: float | None = None,
) -> tuple[int | None, str, str]:
    """Run ``cmd`` (assumed to be an nmap invocation), tee every line into
    the bus as ``NMAP_SCAN`` events, and return the final exit tuple."""
    bus = bus or EventBus()
    # Force stats-every for steady progress; harmless if user already set it.
    if "--stats-every" not in cmd:
        cmd = cmd.rstrip() + " --stats-every 2s"

    proc = await asyncio.create_subprocess_shell(
        " ".join(_wrap_nice(["bash", "-lc", cmd], nice_level)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        return await proc.wait(), "", ""

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    current_host = ""

    async def _drain_stdout() -> None:
        nonlocal current_host
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return
            line = raw.decode(errors="replace")
            stdout_chunks.append(line)
            kind, payload = parse_nmap_line(line, current_host)
            if kind == "host" and isinstance(payload, dict):
                current_host = str(payload.get("host", ""))
            if kind and payload is not None:
                payload["event"] = kind
                try:
                    await bus.publish(Event(EventType.NMAP_SCAN, payload))
                except Exception:
                    log.debug("NMAP_SCAN publish failed", exc_info=True)

    async def _drain_stderr() -> None:
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                return
            stderr_chunks.append(raw.decode(errors="replace"))

    try:
        if timeout is None:
            await asyncio.gather(_drain_stdout(), _drain_stderr())
            rc = await proc.wait()
        else:
            await asyncio.wait_for(
                asyncio.gather(_drain_stdout(), _drain_stderr()),
                timeout=timeout,
            )
            rc = await proc.wait()
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        rc = None

    return rc, "".join(stdout_chunks), "".join(stderr_chunks)
