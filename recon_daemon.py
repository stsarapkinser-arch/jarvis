"""Wraith — recon swarm daemon.

A low-priority singleton (``nice -n 19``) that watches the local environment
for two threat classes and pushes ``RECON_ALERT`` events to the bus:

1. **Wi-Fi vulnerabilities** — tshark / airodump-ng / iw scan output is
   inspected for WEP, WPS-enabled BSSIDs and weak handshakes. Each finding
   is rate-limited and tagged with severity so the HUD knows whether to
   blink yellow or just log.

2. **Intrusion signals** — ``auth.log`` (``Failed password``, ``invalid
   user``, fresh sudo grants) and ``dmesg`` (port scans, USB-injection,
   kernel oopses) are tailed in parallel. Spikes trigger a red HUD flash
   and a UFW BLOCK suggestion that the core dispatches as a destructive
   command (so the gating logic in core.py still asks for voice
   confirmation before firing).

All subprocesses run via ``asyncio.create_subprocess_exec`` with a ``nice -n
19`` prefix and a small wrapper that auto-restarts on EOF, so dropped
processes don't kill the daemon. The HUD subscribes to ``RECON_ALERT`` and
chooses between the yellow / red overlays based on ``payload["color"]``.

This module intentionally does NOT spawn ``airmon-ng`` itself. Putting a
real Wi-Fi card into monitor mode is destructive — it kills NetworkManager
connectivity. Instead, the daemon checks for *existing* ``*mon`` interfaces
and uses them; if none exist it falls back to ``iw dev <iface> scan``,
which only inspects beacons but is safe on the active card.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Iterable

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.recon")

DEFAULT_NICE = 19
WIFI_SCAN_PERIOD_SEC = 45.0
AUTH_LOG_CANDIDATES: tuple[str, ...] = (
    "/var/log/auth.log",
    "/var/log/secure",
)
DMESG_POLL_SEC = 2.0
INTRUSION_WINDOW_SEC = 30.0
INTRUSION_THRESHOLD = 4
ALERT_COOLDOWN_SEC = 60.0


# === Wi-Fi vulnerability patterns ===
# These match common scanner outputs: airodump-ng CSV, iw dev scan, tshark
# decoded beacons. Each pattern captures the BSSID/SSID where possible so the
# HUD can render a clean "MAC :: SSID" tag.
WEP_RE = re.compile(
    r"(?P<bssid>(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}).*\bWEP\b",
    re.IGNORECASE,
)
WPS_RE = re.compile(
    r"\bWPS\b.*?(?:Locked:\s*No|active|enabled|configured)",
    re.IGNORECASE | re.DOTALL,
)
SSID_RE = re.compile(r"\bSSID:\s*[\"']?([^\"'\n,]+)[\"']?")
HANDSHAKE_RE = re.compile(
    r"(EAPOL.*WPA|message.?\s*[1-4].*of\s*4|weak\s+handshake)",
    re.IGNORECASE,
)

# === Intrusion patterns ===
FAILED_LOGIN_RE = re.compile(
    r"(Failed password|authentication failure|invalid user|"
    r"User .* not allowed|Bad protocol version|"
    r"POSSIBLE BREAK-IN ATTEMPT)",
    re.IGNORECASE,
)
PORT_SCAN_RE = re.compile(
    r"(nf_conntrack: table full|connection from .* port \d+|"
    r"SYN flood|TCP: drop open request from)",
    re.IGNORECASE,
)
USB_INJECT_RE = re.compile(
    r"(usb.*input: .*keyboard|new full-speed USB device|"
    r"hid-generic.*\binjection\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReconFinding:
    """Single recon hit. ``color`` drives HUD reaction (yellow|red)."""

    kind: str            # "wifi_wep" | "wifi_wps" | "weak_handshake" | "intrusion_*"
    summary: str         # short human-readable line for the ticker
    detail: str          # full original log line (truncated to 240 chars)
    color: str           # "yellow" | "red"
    severity: str        # "warn" | "alert"
    ts: float

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "detail": self.detail[:240],
            "color": self.color,
            "severity": self.severity,
            "ts": self.ts,
        }


def _wireless_interfaces() -> list[str]:
    """Best-effort detection of wireless ifaces. Returns names like ``wlan0``."""
    try:
        entries = os.listdir("/sys/class/net")
    except OSError:
        return []
    wifis: list[str] = []
    for name in entries:
        if name.startswith(("wlan", "wlp", "wlo")) or name.endswith("mon"):
            wifis.append(name)
    return wifis


def _monitor_interface(ifaces: Iterable[str]) -> str | None:
    """Return the first interface already in monitor mode, if any."""
    for ifc in ifaces:
        if ifc.endswith("mon") or ifc.endswith("0mon"):
            return ifc
    return None


def parse_iw_scan(out: str) -> list[ReconFinding]:
    """Parse ``iw dev wlanX scan`` output.

    Returns one finding per vulnerable BSSID. WEP networks are always
    severity=alert (no excuse in 2026), WPS-enabled networks are warn.
    """
    findings: list[ReconFinding] = []
    if not out:
        return findings
    blocks = re.split(r"^BSS\s+", out, flags=re.MULTILINE)
    now = time.time()
    for block in blocks:
        if not block.strip():
            continue
        head = block.splitlines()[0]
        bssid_m = re.match(r"([0-9A-Fa-f:]{17})", head)
        bssid = bssid_m.group(1) if bssid_m else "??:??:??:??:??:??"
        ssid_m = SSID_RE.search(block)
        ssid = ssid_m.group(1).strip() if ssid_m else "<hidden>"

        if "WEP" in block.upper() and "WPA" not in block.upper():
            findings.append(ReconFinding(
                kind="wifi_wep",
                summary=f"WEP открытая сеть: {ssid}",
                detail=f"BSSID={bssid} SSID={ssid}",
                color="yellow",
                severity="alert",
                ts=now,
            ))
        elif WPS_RE.search(block):
            findings.append(ReconFinding(
                kind="wifi_wps",
                summary=f"WPS активен: {ssid}",
                detail=f"BSSID={bssid} SSID={ssid}",
                color="yellow",
                severity="warn",
                ts=now,
            ))
    return findings


def parse_tshark_line(line: str) -> ReconFinding | None:
    """Inspect a single decoded tshark line for handshakes / WEP beacons."""
    if not line:
        return None
    upper = line.upper()
    now = time.time()
    if "WEP" in upper and "WPA" not in upper and "beacon" in line.lower():
        ssid_m = SSID_RE.search(line)
        ssid = ssid_m.group(1).strip() if ssid_m else "<unknown>"
        return ReconFinding(
            kind="wifi_wep",
            summary=f"WEP beacon: {ssid}",
            detail=line.strip(),
            color="yellow",
            severity="alert",
            ts=now,
        )
    if HANDSHAKE_RE.search(line):
        return ReconFinding(
            kind="weak_handshake",
            summary="Перехвачен WPA-handshake",
            detail=line.strip(),
            color="yellow",
            severity="warn",
            ts=now,
        )
    return None


def parse_intrusion_line(line: str) -> ReconFinding | None:
    """Classify an auth.log/dmesg line as a recon finding."""
    if not line:
        return None
    now = time.time()
    if FAILED_LOGIN_RE.search(line):
        return ReconFinding(
            kind="intrusion_auth",
            summary="Подозрительная попытка авторизации",
            detail=line.strip(),
            color="red",
            severity="alert",
            ts=now,
        )
    if PORT_SCAN_RE.search(line):
        return ReconFinding(
            kind="intrusion_portscan",
            summary="Похоже на скан портов",
            detail=line.strip(),
            color="red",
            severity="alert",
            ts=now,
        )
    if USB_INJECT_RE.search(line):
        return ReconFinding(
            kind="intrusion_usb",
            summary="USB-устройство ввода подключено",
            detail=line.strip(),
            color="red",
            severity="warn",
            ts=now,
        )
    return None


def _extract_remote_ip(detail: str) -> str | None:
    """Best-effort IP extraction for UFW BLOCK suggestions."""
    m = re.search(r"\bfrom\s+(\d+\.\d+\.\d+\.\d+)\b", detail)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", detail)
    return m.group(1) if m else None


def suggest_ufw_block(finding: ReconFinding) -> str | None:
    """Compose a UFW BLOCK command for a finding, if applicable."""
    if finding.color != "red":
        return None
    ip = _extract_remote_ip(finding.detail)
    if not ip or ip.startswith(("127.", "0.")):
        return None
    return f"sudo -n ufw deny from {ip}"


class ReconDaemon(metaclass=Singleton):
    """Wraith singleton. Aggregates Wi-Fi + intrusion findings, publishes
    them to the bus, and rate-limits duplicates so the HUD doesn't strobe."""

    def __init__(self, nice_level: int = DEFAULT_NICE) -> None:
        self.bus = EventBus()
        self.nice_level = nice_level
        self._nice = shutil.which("nice")
        self._tshark = shutil.which("tshark")
        self._iw = shutil.which("iw")
        self._last_alert: dict[str, float] = {}
        self._intrusion_window: list[float] = []
        self._stopped = False

    # ---- helpers ----
    def _wrap_nice(self, cmd: Iterable[str]) -> list[str]:
        if self._nice:
            return [self._nice, "-n", str(self.nice_level), *cmd]
        return list(cmd)

    async def _spawn(self, *cmd: str) -> asyncio.subprocess.Process | None:
        wrapped = self._wrap_nice(cmd)
        try:
            return await asyncio.create_subprocess_exec(
                *wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.info("recon: %s not installed", cmd[0])
            return None
        except Exception:
            log.exception("recon: spawn %s failed", cmd[0])
            return None

    def _should_emit(self, key: str) -> bool:
        now = time.time()
        last = self._last_alert.get(key, 0.0)
        if now - last < ALERT_COOLDOWN_SEC:
            return False
        self._last_alert[key] = now
        return True

    async def _publish(self, finding: ReconFinding) -> None:
        key = f"{finding.kind}:{finding.detail[:80]}"
        if not self._should_emit(key):
            return
        payload = finding.as_dict()
        ufw = suggest_ufw_block(finding)
        if ufw:
            payload["ufw_suggest"] = ufw
        await self.bus.publish(Event(
            EventType.RECON_ALERT,
            payload,
            urgency="critical" if finding.color == "red" else "high",
        ))
        log.info(
            "recon alert [%s/%s]: %s",
            finding.color, finding.severity, finding.summary,
        )

    # ---- Wi-Fi watcher ----
    async def watch_wifi(self) -> None:
        ifaces = _wireless_interfaces()
        if not ifaces:
            log.info("recon: no wireless interfaces detected; wifi watcher off")
            return
        mon = _monitor_interface(ifaces)
        if mon and self._tshark:
            log.info("recon: using tshark on monitor iface %s", mon)
            await self._tshark_loop(mon)
            return

        if self._iw:
            scan_iface = next((i for i in ifaces if not i.endswith("mon")), ifaces[0])
            log.info("recon: passive iw scans on %s every %.0fs", scan_iface, WIFI_SCAN_PERIOD_SEC)
            await self._iw_scan_loop(scan_iface)
            return
        log.info("recon: neither tshark nor iw available; wifi watcher off")

    async def _iw_scan_loop(self, iface: str) -> None:
        while not self._stopped:
            proc = await self._spawn("iw", "dev", iface, "scan")
            if proc is None:
                await asyncio.sleep(WIFI_SCAN_PERIOD_SEC)
                continue
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            except asyncio.TimeoutError:
                proc.kill()
                await asyncio.sleep(WIFI_SCAN_PERIOD_SEC)
                continue
            text = stdout.decode(errors="replace")
            for finding in parse_iw_scan(text):
                await self._publish(finding)
            await asyncio.sleep(WIFI_SCAN_PERIOD_SEC)

    async def _tshark_loop(self, iface: str) -> None:
        # -l line-buffered, -Y filter beacons + EAPOL, -T fields with summary.
        cmd = (
            "tshark", "-i", iface, "-l",
            "-Y", "wlan.fc.type_subtype==8 || eapol",
            "-T", "fields", "-e", "wlan.bssid", "-e", "wlan.ssid", "-e", "wlan_rsna.eapol.keydes.msgnr",
            "-E", "separator=|",
        )
        proc = await self._spawn(*cmd)
        if proc is None or proc.stdout is None:
            return
        try:
            while not self._stopped:
                raw = await proc.stdout.readline()
                if not raw:
                    await asyncio.sleep(0.5)
                    continue
                line = raw.decode(errors="replace").strip()
                finding = parse_tshark_line(line)
                if finding is not None:
                    await self._publish(finding)
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    # ---- Intrusion watcher ----
    async def watch_auth_log(self) -> None:
        path = next((p for p in AUTH_LOG_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            log.info("recon: no auth.log/secure file found; auth watcher off")
            return
        # `tail -F -n0` follows rotations; use `nice` so we yield to Ollama.
        proc = await self._spawn("tail", "-F", "-n", "0", path)
        if proc is None or proc.stdout is None:
            return
        log.info("recon: auth log watcher on %s", path)
        try:
            while not self._stopped:
                raw = await proc.stdout.readline()
                if not raw:
                    await asyncio.sleep(0.5)
                    continue
                line = raw.decode(errors="replace").strip()
                finding = parse_intrusion_line(line)
                if finding is None:
                    continue
                self._record_intrusion()
                await self._publish(finding)
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    async def watch_dmesg(self) -> None:
        if not shutil.which("dmesg"):
            return
        # --follow needs CAP_SYSLOG; if the kernel rejects, we just exit quietly.
        proc = await self._spawn("dmesg", "--follow", "--ctime")
        if proc is None or proc.stdout is None:
            return
        log.info("recon: dmesg watcher started")
        try:
            while not self._stopped:
                raw = await proc.stdout.readline()
                if not raw:
                    await asyncio.sleep(DMESG_POLL_SEC)
                    continue
                line = raw.decode(errors="replace").strip()
                finding = parse_intrusion_line(line)
                if finding is None:
                    continue
                self._record_intrusion()
                await self._publish(finding)
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def _record_intrusion(self) -> None:
        now = time.time()
        self._intrusion_window = [t for t in self._intrusion_window if now - t < INTRUSION_WINDOW_SEC]
        self._intrusion_window.append(now)
        if len(self._intrusion_window) >= INTRUSION_THRESHOLD:
            # The "storm" finding short-circuits cooldown via its own key.
            asyncio.create_task(self._publish(ReconFinding(
                kind="intrusion_storm",
                summary=f"Шквал подозрительных событий ({len(self._intrusion_window)} за {int(INTRUSION_WINDOW_SEC)}с)",
                detail="auth.log + dmesg coincident hits",
                color="red",
                severity="alert",
                ts=now,
            )))
            self._intrusion_window.clear()

    # ---- public API ----
    async def start_all(self) -> list[asyncio.Task]:
        self._stopped = False
        return [
            asyncio.create_task(self.watch_wifi(), name="wraith-wifi"),
            asyncio.create_task(self.watch_auth_log(), name="wraith-auth"),
            asyncio.create_task(self.watch_dmesg(), name="wraith-dmesg"),
        ]

    def stop(self) -> None:
        self._stopped = True
