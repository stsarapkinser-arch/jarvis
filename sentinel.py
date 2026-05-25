from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.sentinel")

THERMAL_WARN_C: Final = 80.0
THERMAL_CRITICAL_C: Final = 90.0
LOAD_RATIO_WARN: Final = 1.5
COOLDOWN_SEC: Final = 60.0
THERMAL_PERIOD_SEC: Final = 2.0
LOAD_PERIOD_SEC: Final = 5.0
PROCESS_PERIOD_SEC: Final = 10.0

PACKAGE_CHECK_PERIOD_SEC: Final = 3600.0
PACKAGE_NIGHT_START: Final = 2
PACKAGE_NIGHT_END: Final = 5
PACKAGE_NIGHT_LOAD_RATIO: Final = 0.3

HEAVY_PROCS: Final[frozenset[str]] = frozenset({
    "firefox", "firefox-esr", "chromium", "chrome", "brave",
    "code", "codium", "code-insiders", "pycharm", "rider", "idea",
    "slack", "discord", "telegram", "zoom", "obs",
    "blender", "kdenlive", "gimp", "inkscape",
    "docker", "dockerd", "containerd",
})

OLLAMA_PROC_NAMES: Final[tuple[str, ...]] = ("ollama",)

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


class Sentinel(metaclass=Singleton):
    """Proactive daemon. Two roles:
    (1) Read thermal/loadavg, publish OS_EVENT with severity.
    (2) Predictively reprioritize Ollama (nice/ionice) based on heavy GUI processes
        and current load, so KDE Plasma stays responsive on N100."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.cpu_count = os.cpu_count() or 1
        self._last_emit: dict[str, float] = {}
        self._throttled: bool = False
        self._renice = shutil.which("renice")
        self._ionice = shutil.which("ionice")
        self._last_pkg_count: int = -1
        self._last_pkg_night_apply_day: str = ""

    def _maybe_emit(self, key: str, event: Event) -> None:
        now = time.time()
        if now - self._last_emit.get(key, 0.0) < COOLDOWN_SEC:
            return
        self._last_emit[key] = now
        asyncio.create_task(self.bus.publish(event))

    @staticmethod
    def _thermal_zones() -> list[Path]:
        return sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))

    @staticmethod
    def _read_temp_c(path: Path) -> float | None:
        try:
            return int(path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    async def watch_thermal(self) -> None:
        zones = self._thermal_zones()
        if not zones:
            log.info("no thermal zones found; thermal watcher disabled")
            return
        log.info("thermal watcher started (%d zones)", len(zones))
        try:
            while True:
                hottest, hot_zone = 0.0, ""
                for z in zones:
                    t = self._read_temp_c(z)
                    if t is None:
                        continue
                    if t > hottest:
                        hottest, hot_zone = t, z.parent.name
                if hottest >= THERMAL_CRITICAL_C:
                    self._maybe_emit("thermal_critical", Event(
                        EventType.OS_EVENT,
                        {"sensor": "thermal", "zone": hot_zone, "value": hottest,
                         "unit": "celsius", "level": "critical",
                         "suggest": "sudo cpupower frequency-set -g powersave"},
                        urgency="critical",
                    ))
                    await self._throttle_ollama(low=True, reason="thermal_critical")
                elif hottest >= THERMAL_WARN_C:
                    self._maybe_emit("thermal_warn", Event(
                        EventType.OS_EVENT,
                        {"sensor": "thermal", "zone": hot_zone, "value": hottest,
                         "unit": "celsius", "level": "warn",
                         "suggest": "sudo cpupower frequency-set -g powersave"},
                        urgency="high",
                    ))
                await asyncio.sleep(THERMAL_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("thermal watcher cancelled")
            raise
        except Exception:
            log.exception("thermal watcher crashed")

    async def watch_loadavg(self) -> None:
        log.info("loadavg watcher started (cpu_count=%d)", self.cpu_count)
        try:
            while True:
                try:
                    la1, _, _ = os.getloadavg()
                except (OSError, AttributeError):
                    return
                threshold = self.cpu_count * LOAD_RATIO_WARN
                if la1 > threshold:
                    self._maybe_emit("load_high", Event(
                        EventType.OS_EVENT,
                        {"sensor": "loadavg", "value": la1, "threshold": threshold,
                         "level": "warn", "suggest": "ps aux --sort=-%cpu | head -5"},
                        urgency="high",
                    ))
                    await self._throttle_ollama(low=True, reason="load_high")
                await asyncio.sleep(LOAD_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("loadavg watcher cancelled")
            raise
        except Exception:
            log.exception("loadavg watcher crashed")

    @staticmethod
    def _heavy_running() -> bool:
        if _HAS_PSUTIL:
            try:
                for p in psutil.process_iter(["name"]):
                    name = (p.info.get("name") or "").lower()
                    if name in HEAVY_PROCS:
                        return True
                return False
            except Exception:
                pass
        try:
            out = subprocess.check_output(
                ["pgrep", "-x", "-l", "-f", "|".join(HEAVY_PROCS)],
                text=True, timeout=1, stderr=subprocess.DEVNULL,
            )
            return bool(out.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _ollama_pids() -> list[int]:
        if _HAS_PSUTIL:
            try:
                return [
                    p.pid for p in psutil.process_iter(["name"])
                    if (p.info.get("name") or "").lower() in OLLAMA_PROC_NAMES
                ]
            except Exception:
                pass
        pids: list[int] = []
        for name in OLLAMA_PROC_NAMES:
            try:
                out = subprocess.check_output(
                    ["pgrep", "-x", name], text=True, timeout=1, stderr=subprocess.DEVNULL,
                )
                pids.extend(int(s) for s in out.split())
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return pids

    async def _run_quiet(self, *cmd: str) -> int:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            return await proc.wait()
        except FileNotFoundError:
            return 127
        except Exception:
            log.exception("subprocess %s failed", cmd[0])
            return 1

    async def _throttle_ollama(self, *, low: bool, reason: str) -> None:
        if low and self._throttled:
            return
        if not low and not self._throttled:
            return
        pids = self._ollama_pids()
        if not pids:
            return
        nice_val = "10" if low else "0"
        ionice_class = "3" if low else "2"

        for pid in pids:
            if self._renice:
                await self._run_quiet(self._renice, "-n", nice_val, "-p", str(pid))
            if self._ionice:
                await self._run_quiet(self._ionice, "-c", ionice_class, "-p", str(pid))
        self._throttled = low
        log.info(
            "ollama %s (reason=%s, pids=%s, nice=%s, ionice=%s)",
            "throttled" if low else "restored", reason, pids, nice_val, ionice_class,
        )

    async def watch_processes(self) -> None:
        log.info("process watcher started")
        try:
            while True:
                try:
                    if self._heavy_running():
                        await self._throttle_ollama(low=True, reason="heavy_apps")
                    else:
                        try:
                            la1, _, _ = os.getloadavg()
                        except (OSError, AttributeError):
                            la1 = 0.0
                        if la1 < self.cpu_count * 0.9:
                            await self._throttle_ollama(low=False, reason="idle")
                except Exception:
                    log.exception("process watcher iteration failed")
                await asyncio.sleep(PROCESS_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("process watcher cancelled")
            raise

    async def start_all(self) -> list[asyncio.Task]:
        return [
            asyncio.create_task(self.watch_thermal(), name="sentinel-thermal"),
            asyncio.create_task(self.watch_loadavg(), name="sentinel-load"),
            asyncio.create_task(self.watch_processes(), name="sentinel-procs"),
            asyncio.create_task(self.watch_packages(), name="sentinel-packages"),
        ]

    async def _count_upgradable(self) -> int:
        if not shutil.which("apt"):
            return 0
        try:
            proc = await asyncio.create_subprocess_exec(
                "apt", "list", "--upgradable",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ, "LANG": "C", "LC_ALL": "C"},
            )
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
        except (asyncio.TimeoutError, FileNotFoundError):
            return 0
        except Exception:
            log.exception("apt list --upgradable failed")
            return 0
        lines = [ln for ln in stdout_b.decode(errors="replace").splitlines() if "/" in ln]
        return len(lines)

    async def _count_security_upgradable(self) -> int:
        if not shutil.which("apt"):
            return 0
        try:
            proc = await asyncio.create_subprocess_shell(
                "apt list --upgradable 2>/dev/null | grep -ciE 'security|kali-security'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            return int(stdout_b.decode().strip() or "0")
        except (asyncio.TimeoutError, ValueError, FileNotFoundError):
            return 0
        except Exception:
            log.exception("security count failed")
            return 0

    async def _apt_upgrade_quiet(self) -> tuple[int, str]:
        cmd = ("sudo", "-n", "DEBIAN_FRONTEND=noninteractive",
               "apt-get", "-y", "-o", "Dpkg::Options::=--force-confold",
               "-o", "Dpkg::Options::=--force-confdef", "upgrade")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=1800.0)
            return proc.returncode or 0, (err.decode(errors="replace") or out.decode(errors="replace"))[:400]
        except asyncio.TimeoutError:
            return 124, "apt-get upgrade timed out"
        except Exception as exc:
            return 1, str(exc)

    @staticmethod
    def _is_night() -> bool:
        h = time.localtime().tm_hour
        return PACKAGE_NIGHT_START <= h < PACKAGE_NIGHT_END

    @staticmethod
    def _today_key() -> str:
        t = time.localtime()
        return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"

    async def watch_packages(self) -> None:
        log.info("package watcher started")
        await asyncio.sleep(15)
        try:
            while True:
                try:
                    count = await self._count_upgradable()
                    if count == 0:
                        self._last_pkg_count = 0
                    elif count != self._last_pkg_count:
                        self._last_pkg_count = count
                        sec = await self._count_security_upgradable()
                        suggest = "sudo apt-get upgrade -y"
                        if sec > 0:
                            self._maybe_emit("pkg_security", Event(
                                EventType.OS_EVENT,
                                {"sensor": "packages", "value": sec, "level": "warn",
                                 "suggest": suggest, "kind": "security"},
                                urgency="high",
                            ))
                        else:
                            self._maybe_emit("pkg_avail", Event(
                                EventType.OS_EVENT,
                                {"sensor": "packages", "value": count, "level": "info",
                                 "suggest": suggest, "kind": "regular"},
                                urgency="low",
                            ))

                    if (
                        count > 0
                        and self._is_night()
                        and self._last_pkg_night_apply_day != self._today_key()
                    ):
                        try:
                            la1, _, _ = os.getloadavg()
                        except (OSError, AttributeError):
                            la1 = 99.0
                        if la1 < self.cpu_count * PACKAGE_NIGHT_LOAD_RATIO:
                            rc, msg = await self._apt_upgrade_quiet()
                            self._last_pkg_night_apply_day = self._today_key()
                            if rc == 0:
                                log.info("nightly upgrade applied: %d packages", count)
                                await self.bus.publish(Event(
                                    EventType.OS_EVENT,
                                    {"sensor": "packages", "value": count, "level": "info",
                                     "suggest": "applied overnight", "kind": "applied"},
                                    urgency="low",
                                ))
                                self._last_pkg_count = 0
                            else:
                                log.warning("nightly upgrade failed rc=%s: %s", rc, msg)
                except Exception:
                    log.exception("package watcher iteration failed")
                await asyncio.sleep(PACKAGE_CHECK_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("package watcher cancelled")
            raise
