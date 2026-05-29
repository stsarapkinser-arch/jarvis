from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

from src.common.event_bus import Event, EventBus, EventType, SystemLoad, SystemState
from src.common.singleton import Singleton

log = logging.getLogger("jarvis.sentinel")

THERMAL_WARN_C: Final = 80.0
THERMAL_CRITICAL_C: Final = 90.0
LOAD_RATIO_WARN: Final = 1.5
COOLDOWN_SEC: Final = 60.0
THERMAL_PERIOD_SEC: Final = 2.0
LOAD_PERIOD_SEC: Final = 5.0
PROCESS_PERIOD_SEC: Final = 10.0
TRAFFIC_PERIOD_SEC: Final = 1.0
# iGPU busy poll — 1 Hz хватает: HUD neural-pulse реагирует на пороге
# 80%, без нужды в сабсекундном разрешении. /sys read стоит ~5 мкс.
IGPU_PERIOD_SEC: Final = 1.0
# Кандидаты на DRM sysfs-узел iGPU. На N100 i915 даёт /sys/class/drm/card0,
# но на свежих Mesa/i915 нумерация card0/card1 может плясать (внешний DP,
# DG2). Проверяем оба перед сдачей.
IGPU_BUSY_PATHS: Final[tuple[Path, ...]] = (
    Path("/sys/class/drm/card0/device/gpu_busy_percent"),
    Path("/sys/class/drm/card1/device/gpu_busy_percent"),
)
TRAFFIC_REFERENCE_BPS: Final = 5_242_880.0  # 5 MiB/s -> normalised to 1.0

# Disk / RAM watcher: pragmatic 30-second psutil tick. We deliberately do NOT
# poll faster — disk and memory move slowly, and we want the N100 to sleep
# between samples. The 90 % threshold matches the user's spec for "Sir, disk
# is filling up". Cleared/re-armed by the existing _maybe_emit cooldown so a
# stuck-full disk doesn't nag every 30 s — only every COOLDOWN_SEC.
DISK_RAM_PERIOD_SEC: Final = 30.0
DISK_THRESHOLD_PCT: Final = 90.0
RAM_THRESHOLD_PCT: Final = 90.0
DISK_PATH: Final = "/"

PACKAGE_CHECK_PERIOD_SEC: Final = 3600.0
PACKAGE_NIGHT_START: Final = 2
PACKAGE_NIGHT_END: Final = 5
PACKAGE_NIGHT_LOAD_RATIO: Final = 0.3

# Laptop battery watcher — fires LOW once when capacity falls below LOW_PCT
# AND status != Charging. Re-arms when capacity climbs back above REARM_PCT.
BATTERY_PERIOD_SEC: Final = 30.0
BATTERY_LOW_PCT: Final = 15
BATTERY_CRITICAL_PCT: Final = 5
BATTERY_REARM_PCT: Final = 30

# Internet stability watcher — pings a stable anycast target (Cloudflare,
# Google DNS) at INET_PERIOD_SEC. Tracks rolling RTT/loss; if last
# INET_BAD_RATIO of the last INET_WINDOW samples were lost OR RTT > the
# rolling-mean × INET_RTT_MULT, we emit a warn.
INET_PERIOD_SEC: Final = 10.0
INET_TARGETS: Final[tuple[str, ...]] = ("1.1.1.1", "8.8.8.8")
INET_WINDOW: Final = 12          # 2 minutes worth at PERIOD=10s
INET_BAD_RATIO: Final = 0.5      # >50% loss in window → "lost"
INET_RTT_MULT: Final = 2.5       # current RTT > 2.5× rolling-mean → "jittery"
INET_RTT_FLOOR_MS: Final = 50.0  # don't fire on tiny mean jitter near zero

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
        self._state = SystemState()

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
                elif hottest >= THERMAL_WARN_C:
                    self._maybe_emit("thermal_warn", Event(
                        EventType.OS_EVENT,
                        {"sensor": "thermal", "zone": hot_zone, "value": hottest,
                         "unit": "celsius", "level": "warn",
                         "suggest": "sudo cpupower frequency-set -g powersave"},
                        urgency="high",
                    ))
                # Always feed the symbiote — even a cool reading matters
                # for falling back out of HIGH after a thermal spike.
                await self._ingest_state(thermal=hottest, reason=f"thermal:{hot_zone or '?'}")
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
                # Convert loadavg to a 0..100 CPU-saturation proxy. We
                # prefer psutil.cpu_percent when present (instantaneous);
                # loadavg is a 1-minute EMA and lags behind reality.
                cpu_pct: float | None = None
                if _HAS_PSUTIL:
                    try:
                        cpu_pct = float(psutil.cpu_percent(interval=None))
                    except Exception:
                        cpu_pct = None
                if cpu_pct is None:
                    cpu_pct = min(100.0, (la1 / max(1, self.cpu_count)) * 100.0)
                await self._ingest_state(cpu=cpu_pct, reason=f"loadavg:{la1:.2f}")
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

    async def _ingest_state(
        self,
        *,
        cpu: float | None = None,
        ram: float | None = None,
        thermal: float | None = None,
        gpu: float | None = None,
        heavy: bool | None = None,
        reason: str,
    ) -> None:
        """Update the symbiote SystemState; publish + react on tier change.

        Every watcher in this class funnels its newest reading through here
        instead of independently triggering renice / HUD. The single fan-in
        point keeps cascade reactions consistent: one SYSTEM_STATE event
        per actual transition, never per-sample.

        Поле ``gpu`` отдельно — не меняет SystemLoad tier, но триггерит
        SYSTEM_STATE при пересечении 80% (см. SystemState.update). HUD
        читает gpu из snapshot и переключает neural-pulse сферы."""
        changed, snap = self._state.update(
            cpu=cpu, ram=ram, thermal=thermal, gpu=gpu, heavy=heavy, reason=reason,
        )
        if not changed:
            return
        try:
            await self.bus.publish(Event(
                EventType.SYSTEM_STATE, snap.as_dict(),
                urgency="high" if snap.load in (SystemLoad.HIGH, SystemLoad.CRITICAL) else "normal",
            ))
        except Exception:
            log.exception("SYSTEM_STATE publish failed")
        # Cascade: when the organism crosses into HIGH/CRITICAL, renice
        # Ollama so a heavy compile doesn't get its CPU stolen by an LLM
        # decode. When we fall back to NORMAL/IDLE, restore.
        if snap.load in (SystemLoad.HIGH, SystemLoad.CRITICAL):
            await self._throttle_ollama(low=True, reason=f"symbiote_{snap.load}")
        else:
            await self._throttle_ollama(low=False, reason=f"symbiote_{snap.load}")

    async def watch_processes(self) -> None:
        log.info("process watcher started")
        try:
            while True:
                try:
                    heavy = self._heavy_running()
                    await self._ingest_state(
                        heavy=heavy,
                        reason="heavy_apps" if heavy else "no_heavy_apps",
                    )
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
            asyncio.create_task(self.watch_traffic(), name="sentinel-traffic"),
            asyncio.create_task(self.watch_disk_ram(), name="sentinel-diskram"),
            asyncio.create_task(self.watch_igpu(), name="sentinel-igpu"),
            asyncio.create_task(self.watch_suspend_resume(), name="sentinel-suspend"),
            asyncio.create_task(self.watch_meeting(), name="sentinel-meeting"),
            asyncio.create_task(self.watch_battery(), name="sentinel-battery"),
            asyncio.create_task(self.watch_internet(), name="sentinel-internet"),
        ]

    # ───────────── laptop battery watcher ─────────────
    async def watch_battery(self) -> None:
        """Polls /sys/class/power_supply/BAT* — emits low/critical events.

        Edge-triggered: LOW fires once when capacity dips below threshold
        and status is Discharging. Re-arms when capacity rises above
        BATTERY_REARM_PCT, regardless of charging state."""
        bat_root = self._find_battery()
        if bat_root is None:
            log.info("no laptop battery detected — battery watcher disabled")
            return
        log.info(
            "battery watcher started (path=%s, low<=%d%%, critical<=%d%%, period=%.0fs)",
            bat_root, BATTERY_LOW_PCT, BATTERY_CRITICAL_PCT, BATTERY_PERIOD_SEC,
        )
        state = "ok"  # "ok" | "low" | "critical"
        try:
            while True:
                pct, status = self._read_battery(bat_root)
                if pct is None:
                    await asyncio.sleep(BATTERY_PERIOD_SEC)
                    continue
                discharging = status not in ("Charging", "Full")
                # Re-arm when significantly recharged.
                if pct >= BATTERY_REARM_PCT and state != "ok":
                    log.info("battery re-armed at %d%%", pct)
                    state = "ok"
                # Critical takes precedence over low.
                if discharging and pct <= BATTERY_CRITICAL_PCT and state != "critical":
                    state = "critical"
                    self._maybe_emit("battery_crit", Event(
                        EventType.OS_EVENT,
                        {
                            "sensor": "battery",
                            "value": pct,
                            "unit": "%",
                            "level": "critical",
                            "status": status,
                            "suggest": "save work; plug in charger now",
                            "ask": False,
                        },
                        urgency="high",
                    ))
                elif discharging and pct <= BATTERY_LOW_PCT and state == "ok":
                    state = "low"
                    self._maybe_emit("battery_low", Event(
                        EventType.OS_EVENT,
                        {
                            "sensor": "battery",
                            "value": pct,
                            "unit": "%",
                            "level": "warn",
                            "status": status,
                            "suggest": "plug in within the next ~15 minutes",
                            "ask": False,
                        },
                        urgency="high",
                    ))
                await asyncio.sleep(BATTERY_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("battery watcher cancelled")
            raise
        except Exception:
            log.exception("battery watcher crashed")

    @staticmethod
    def _find_battery() -> Path | None:
        root = Path("/sys/class/power_supply")
        if not root.is_dir():
            return None
        for entry in sorted(root.iterdir()):
            try:
                tname = (entry / "type").read_text().strip()
            except OSError:
                continue
            if tname == "Battery" and (entry / "capacity").is_file():
                return entry
        return None

    @staticmethod
    def _read_battery(bat: Path) -> tuple[int | None, str]:
        try:
            cap = int((bat / "capacity").read_text().strip())
        except (OSError, ValueError):
            return None, ""
        try:
            status = (bat / "status").read_text().strip()
        except OSError:
            status = ""
        return cap, status

    # ───────────── internet stability watcher ─────────────
    async def watch_internet(self) -> None:
        """Pings stable anycast targets; emits warn when loss/jitter spike.

        We sample one of INET_TARGETS every INET_PERIOD_SEC seconds, keep
        a rolling window of (rtt_ms, lost) tuples, and emit a warn event
        when either:
          * lost-ratio in the window exceeds INET_BAD_RATIO, OR
          * latest RTT > rolling-mean RTT × INET_RTT_MULT (above the floor).

        Cooldown handled by _maybe_emit('inet_unstable'); no spam."""
        if not shutil.which("ping"):
            log.info("ping not found — internet stability watcher disabled")
            return
        log.info(
            "internet stability watcher started (period=%.0fs window=%d)",
            INET_PERIOD_SEC, INET_WINDOW,
        )
        window: list[tuple[float, bool]] = []  # (rtt_ms, lost)
        target_idx = 0
        try:
            while True:
                target = INET_TARGETS[target_idx % len(INET_TARGETS)]
                target_idx += 1
                rtt = await self._ping_rtt(target)
                lost = rtt is None
                window.append((0.0 if lost else float(rtt), lost))
                if len(window) > INET_WINDOW:
                    window.pop(0)

                if len(window) >= max(4, INET_WINDOW // 2):
                    losses = sum(1 for _, was_lost in window if was_lost)
                    loss_ratio = losses / len(window)
                    good = [r for r, was_lost in window if not was_lost]
                    mean_rtt = (sum(good) / len(good)) if good else 0.0
                    last_rtt = window[-1][0]
                    jitter_spike = (
                        not lost
                        and mean_rtt > INET_RTT_FLOOR_MS
                        and last_rtt > mean_rtt * INET_RTT_MULT
                    )

                    if loss_ratio >= INET_BAD_RATIO or jitter_spike:
                        reason = "loss" if loss_ratio >= INET_BAD_RATIO else "jitter"
                        self._maybe_emit("inet_unstable", Event(
                            EventType.OS_EVENT,
                            {
                                "sensor": "internet",
                                "value": round(loss_ratio * 100, 1),
                                "unit": "%loss",
                                "level": "warn",
                                "reason": reason,
                                "last_rtt_ms": round(last_rtt, 1) if not lost else None,
                                "mean_rtt_ms": round(mean_rtt, 1) if mean_rtt else None,
                                "target": target,
                                "suggest": "nmcli device wifi list; ip -brief addr",
                                "ask": False,
                            },
                            urgency="normal",
                        ))
                await asyncio.sleep(INET_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("internet watcher cancelled")
            raise
        except Exception:
            log.exception("internet watcher crashed")

    async def _ping_rtt(self, target: str) -> float | None:
        """Single ping with 2-second deadline; returns RTT in ms or None on loss."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ping", "-n", "-c", "1", "-W", "2", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        except (TimeoutError, FileNotFoundError):
            return None
        except Exception:
            log.debug("ping %s failed", target, exc_info=True)
            return None
        if proc.returncode != 0:
            return None
        # Parse "time=12.3 ms"
        import re
        m = re.search(r"time[=<]([\d.]+)\s*ms", stdout.decode(errors="replace"))
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # ───────────── suspend/resume watcher (login1) ─────────────
    async def watch_suspend_resume(self) -> None:
        """Слушает org.freedesktop.login1.Manager.PrepareForSleep.

        Сигнал шлётся дважды:
          • перед сном — bool=True
          • после пробуждения — bool=False
        Нас интересует только переход True→False (фактический resume), на
        нём публикуем SYSTEM_WAKE. HUD на ~2 сек подкрашивает ауру в яркий
        cyan и затухает в дефолт — оператор видит, что машина «проснулась».

        Если dbus-next не установлен или system-bus недоступен (контейнер
        без /var/run/dbus, headless CI) — тихо логируемся и уходим, как
        делает watch_igpu при отсутствии sysfs."""
        try:
            from dbus_next.aio import MessageBus
            from dbus_next.constants import BusType
        except ImportError:
            log.info("dbus-next not installed — suspend/resume watcher disabled")
            return
        try:
            bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        except Exception:
            log.info("system bus unavailable — suspend/resume watcher disabled")
            return
        try:
            introspect = await bus.introspect("org.freedesktop.login1", "/org/freedesktop/login1")
            proxy = bus.get_proxy_object(
                "org.freedesktop.login1", "/org/freedesktop/login1", introspect
            )
            iface = proxy.get_interface("org.freedesktop.login1.Manager")
        except Exception:
            log.info("login1 manager not reachable — suspend/resume watcher disabled")
            try:
                bus.disconnect()
            except Exception:
                pass
            return

        state = {"sleeping": False}

        def _on_prepare_for_sleep(going_to_sleep: bool) -> None:
            # Транзишн True→False == resume.
            if state["sleeping"] and not going_to_sleep:
                log.info("login1: resume detected — publishing SYSTEM_WAKE")
                self.bus.publish_threadsafe(Event(
                    EventType.SYSTEM_WAKE,
                    {"ts": time.time()},
                    urgency="normal",
                ))
            state["sleeping"] = bool(going_to_sleep)

        try:
            iface.on_prepare_for_sleep(_on_prepare_for_sleep)
            log.info("suspend/resume watcher armed (login1 PrepareForSleep)")
        except Exception:
            log.exception("failed to subscribe to PrepareForSleep")
            try:
                bus.disconnect()
            except Exception:
                pass
            return

        # Висим живыми пока loop работает; dbus-next держит коллбек самостоятельно.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            log.info("suspend/resume watcher cancelled")
            try:
                bus.disconnect()
            except Exception:
                pass
            raise

    # ───────────── meeting auto-dim watcher ─────────────
    # Доп-фича из плана: пока запущен видеоконф процесс — HUD ауры приглушаются
    # до 0.35, оператор показывает экран и не должен светить голубым свечением.
    MEETING_PROC_RE: Final = (
        "zoom", "teams", "google-meet", "google_meet", "obs",
        "obs-studio", "obsstudio",
    )
    MEETING_PERIOD_SEC: Final = 5.0

    async def watch_meeting(self) -> None:
        """Опрос процесс-листа на predefined meeting-приложения.

        При первой детекции публикуем HUD_OVERLAY(kind="aura_intensity", value=0.35),
        при пропадании — value=1.0. Тонкая, но приятная UX-мелочь."""
        prev_meeting = False
        try:
            while True:
                meeting = await asyncio.to_thread(self._meeting_active_sync)
                if meeting != prev_meeting:
                    await self.bus.publish(Event(
                        EventType.HUD_OVERLAY,
                        {"kind": "aura_intensity", "value": 0.35 if meeting else 1.0},
                    ))
                    log.info("meeting %s — aura intensity %s",
                             "ON" if meeting else "OFF", 0.35 if meeting else 1.0)
                    prev_meeting = meeting
                await asyncio.sleep(self.MEETING_PERIOD_SEC)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("meeting watcher crashed")

    def _meeting_active_sync(self) -> bool:
        try:
            out = subprocess.check_output(
                ["ps", "-A", "-o", "comm="], text=True, timeout=2,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        lower = out.lower()
        return any(p in lower for p in self.MEETING_PROC_RE)

    # ───────────── iGPU watcher (Intel N100 chemistry) ─────────────
    # Читаем busy% напрямую из drm/i915 sysfs — без root, без deps.
    # На N100 + ядре >=5.10 узел есть всегда; на нестандартных стэках
    # (внешний DG2 как card0, отсутствие i915) — watcher тихо уходит.
    @staticmethod
    def _read_gpu_busy() -> float | None:
        """Возвращает iGPU busy % в [0, 100] или None если sysfs недоступен."""
        for path in IGPU_BUSY_PATHS:
            try:
                val = float(path.read_text().strip())
            except (OSError, ValueError):
                continue
            return max(0.0, min(100.0, val))
        return None

    async def watch_igpu(self) -> None:
        first = self._read_gpu_busy()
        if first is None:
            log.info("no iGPU sysfs (gpu_busy_percent) — igpu watcher disabled")
            return
        log.info("igpu watcher started (period=%.1fs, initial=%.0f%%)",
                 IGPU_PERIOD_SEC, first)
        try:
            while True:
                busy = self._read_gpu_busy()
                if busy is not None:
                    await self._ingest_state(gpu=busy, reason=f"igpu:{busy:.0f}%")
                await asyncio.sleep(IGPU_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("igpu watcher cancelled")
            raise
        except Exception:
            log.exception("igpu watcher crashed")

    # ───────────── disk + ram watcher ─────────────
    # Two cheap psutil reads every 30 s. Each crossing of the 90 % line emits
    # exactly one OS_EVENT (gated by _maybe_emit's 60 s cooldown), so the
    # operator hears "disk is filling up" once, not every half-minute.
    async def watch_disk_ram(self) -> None:
        if not _HAS_PSUTIL:
            log.info("psutil missing — disk/ram watcher disabled")
            return
        log.info(
            "disk/ram watcher started (path=%s, disk>=%.0f%%, ram>=%.0f%%, period=%.0fs)",
            DISK_PATH, DISK_THRESHOLD_PCT, RAM_THRESHOLD_PCT, DISK_RAM_PERIOD_SEC,
        )
        try:
            while True:
                try:
                    du = psutil.disk_usage(DISK_PATH)
                    vm = psutil.virtual_memory()
                except Exception:
                    log.exception("disk/ram sample failed")
                    await asyncio.sleep(DISK_RAM_PERIOD_SEC)
                    continue

                if du.percent >= DISK_THRESHOLD_PCT:
                    free_gb = du.free / (1024 ** 3)
                    self._maybe_emit("disk_full", Event(
                        EventType.OS_EVENT,
                        {
                            "sensor": "disk",
                            "value": round(du.percent, 1),
                            "unit": "%",
                            "level": "warn",
                            "free_gb": round(free_gb, 2),
                            "path": DISK_PATH,
                            "suggest": (
                                "sudo apt-get clean && "
                                "sudo journalctl --vacuum-time=7d"
                            ),
                            "ask": True,  # core.on_os_event should pose a question
                        },
                        urgency="high",
                    ))

                if vm.percent >= RAM_THRESHOLD_PCT:
                    self._maybe_emit("ram_full", Event(
                        EventType.OS_EVENT,
                        {
                            "sensor": "memory",
                            "value": round(vm.percent, 1),
                            "unit": "%",
                            "level": "warn",
                            "available_mb": int(vm.available / (1024 ** 2)),
                            "suggest": "ps aux --sort=-%mem | head -10",
                            "ask": True,
                        },
                        urgency="high",
                    ))

                # Feed RAM into the symbiote — used by SystemState._classify
                # to push the organism to HIGH at 80 % and CRITICAL at 92 %.
                await self._ingest_state(ram=float(vm.percent), reason=f"ram:{vm.percent:.0f}%")

                await asyncio.sleep(DISK_RAM_PERIOD_SEC)
        except asyncio.CancelledError:
            log.info("disk/ram watcher cancelled")
            raise
        except Exception:
            log.exception("disk/ram watcher crashed")

    # ───────────── traffic poller ─────────────
    # /proc/net/dev sample → bytes/s summed across non-lo interfaces →
    # TRAFFIC event with `norm` ∈ [0,1] (relative to TRAFFIC_REFERENCE_BPS).
    # Drives the HUD shader's domain-warp distortion.

    @staticmethod
    def _read_net_dev() -> dict[str, tuple[int, int]]:
        """Returns {iface: (rx_bytes, tx_bytes)}. Skips lo and down ifaces."""
        result: dict[str, tuple[int, int]] = {}
        try:
            with open("/proc/net/dev", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return result
        for line in lines[2:]:
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            if name == "lo" or not name:
                continue
            cols = rest.split()
            if len(cols) < 9:
                continue
            try:
                rx = int(cols[0])
                tx = int(cols[8])
            except ValueError:
                continue
            result[name] = (rx, tx)
        return result

    async def watch_traffic(self) -> None:
        log.info("traffic poller started (ref=%.1f MiB/s)", TRAFFIC_REFERENCE_BPS / (1024 * 1024))
        prev = self._read_net_dev()
        prev_ts = time.monotonic()
        try:
            while True:
                await asyncio.sleep(TRAFFIC_PERIOD_SEC)
                now_ts = time.monotonic()
                cur = self._read_net_dev()
                dt = max(0.05, now_ts - prev_ts)
                total_bps = 0.0
                per_iface: dict[str, float] = {}
                for iface, (rx, tx) in cur.items():
                    prv = prev.get(iface)
                    if prv is None:
                        continue
                    drx = max(0, rx - prv[0])
                    dtx = max(0, tx - prv[1])
                    bps = (drx + dtx) / dt
                    per_iface[iface] = bps
                    total_bps += bps
                prev = cur
                prev_ts = now_ts
                norm = min(1.0, total_bps / TRAFFIC_REFERENCE_BPS)
                try:
                    await self.bus.publish(Event(
                        EventType.TRAFFIC,
                        {
                            "bps": total_bps,
                            "norm": norm,
                            "per_iface": per_iface,
                        },
                        urgency="low",
                    ))
                except Exception:
                    log.exception("traffic publish failed")
        except asyncio.CancelledError:
            log.info("traffic poller cancelled")
            raise
        except Exception:
            log.exception("traffic poller crashed")

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
        except (TimeoutError, FileNotFoundError):
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
        except (TimeoutError, ValueError, FileNotFoundError):
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
        except TimeoutError:
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
