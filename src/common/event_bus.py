from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from src.common.singleton import Singleton

log = logging.getLogger("jarvis.bus")

Urgency = Literal["low", "normal", "high", "critical"]


class EventType(StrEnum):
    VOICE_INTENT = "VOICE_INTENT"
    DAEMON_ALERT = "DAEMON_ALERT"
    OS_EVENT = "OS_EVENT"
    PIXEL_EVENT = "PIXEL_EVENT"
    STATE_CHANGE = "STATE_CHANGE"
    HUD_STATE = "HUD_STATE"        # set_hud_state tool: нейросеть сама рулит визором (color+animation)
    TOKEN_STREAM = "TOKEN_STREAM"
    KWIN_ACTION = "KWIN_ACTION"
    # Aegis / Wraith / Navigator additions.
    AUDIO_FFT = "AUDIO_FFT"        # Piper output amplitude / spectral bands
    RECON_ALERT = "RECON_ALERT"    # Wraith findings (Wi-Fi vuln, intrusion, …)
    NMAP_SCAN = "NMAP_SCAN"        # nmap progress lines (host/port/state)
    HUD_OVERLAY = "HUD_OVERLAY"    # ad-hoc HUD payloads (graph nodes, projections)
    DEEP_WATCH = "DEEP_WATCH"      # Ring-0 eBPF: execve / tcp_v4_connect samples
    TRAFFIC = "TRAFFIC"            # bytes/s per interface (HUD shader distortion)
    SYSTEM_STATE = "SYSTEM_STATE"  # Symbiote: load tier transition (idle/normal/high/critical)
    # ───────── Adaptive Visor (ТЗ оператора, фаза «Ambient Aura») ─────────
    SYSTEM_WAKE = "SYSTEM_WAKE"            # resume из suspend (Sentinel ↔ login1)
    CALL_INBOUND = "CALL_INBOUND"          # высокоуровневый incoming-call (HUD-ready)
    DIAGNOSTIC_START = "DIAGNOSTIC_START"  # голосовой триггер диагностики (HUD overlay)


class SystemLoad(StrEnum):
    """Symbiote load tier. Computed by Sentinel from cpu/ram/thermal/heavy
    processes. The whole organism reacts to a tier change: HUD drops FPS,
    the embed-server is reniced, Piper speaks faster."""
    IDLE = "idle"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SystemStateSnapshot:
    load: SystemLoad
    cpu: float
    ram: float
    thermal: float
    # iGPU busy % из /sys/class/drm/card0/device/gpu_busy_percent. Заполняется
    # Sentinel.watch_igpu(); консьюмеры (HUD neural-pulse, dashboard диагностика)
    # читают как часть snapshot. Не участвует в _classify — высокая нагрузка
    # на iGPU не делает систему неотзывчивой (это часть нашего же inference),
    # так что в общий SystemLoad tier не пушим.
    gpu: float
    heavy: bool
    reason: str
    ts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "load": str(self.load),
            "cpu": round(self.cpu, 1),
            "ram": round(self.ram, 1),
            "thermal": round(self.thermal, 1),
            "gpu": round(self.gpu, 1),
            "heavy": self.heavy,
            "reason": self.reason,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class Event:
    type: EventType
    data: Any
    urgency: Urgency = "normal"
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None]]


def safe_async(fn: Handler) -> Handler:
    @functools.wraps(fn)
    async def wrapper(event: Event) -> None:
        try:
            await fn(event)
        except Exception:
            log.exception("handler %s failed on %s", getattr(fn, "__qualname__", fn), event.type)

    return wrapper


class EventBus(metaclass=Singleton):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subs: dict[EventType, list[Handler]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._subs[event_type].append(safe_async(handler))

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    def publish_threadsafe(self, event: Event) -> None:
        if self._loop is None:
            log.warning("publish_threadsafe before bind_loop; dropping %s", event.type)
            return
        asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)

    async def run(self) -> None:
        log.info("bus loop started")
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                log.info("bus loop cancelled")
                raise
            except Exception:
                log.exception("bus queue.get failed")
                await asyncio.sleep(0.1)
                continue
            for handler in self._subs.get(event.type, ()):
                asyncio.create_task(handler(event))


class SystemState(metaclass=Singleton):
    """Single source of truth for the organism's load tier.

    Sentinel ``update()``s this from its existing watchers; consumers
    (HUD's CoreSphereGL, ``core.say()``, embed-server renicer) read the latest
    snapshot via :py:meth:`snapshot` / :py:attr:`load`. A change in tier
    is the trigger to publish ``EventType.SYSTEM_STATE`` on the bus."""

    def __init__(self) -> None:
        self._snap = SystemStateSnapshot(
            load=SystemLoad.NORMAL,
            cpu=0.0, ram=0.0, thermal=0.0, gpu=0.0,
            heavy=False, reason="boot",
            ts=time.time(),
        )

    @property
    def load(self) -> SystemLoad:
        return self._snap.load

    def snapshot(self) -> SystemStateSnapshot:
        return self._snap

    @staticmethod
    def _classify(cpu: float, ram: float, thermal: float, heavy: bool) -> SystemLoad:
        # Critical: thermal throttle or near-OOM.
        if thermal >= 90.0 or ram >= 92.0:
            return SystemLoad.CRITICAL
        # High: any of: sustained CPU saturation, heavy GUI app, hot SoC,
        # RAM closing in. Designed so the N100 doesn't have to be on fire
        # before the symbiote downshifts.
        if cpu >= 75.0 or ram >= 80.0 or thermal >= 80.0 or heavy:
            return SystemLoad.HIGH
        if cpu < 15.0 and ram < 50.0:
            return SystemLoad.IDLE
        return SystemLoad.NORMAL

    def update(
        self,
        *,
        cpu: float | None = None,
        ram: float | None = None,
        thermal: float | None = None,
        gpu: float | None = None,
        heavy: bool | None = None,
        reason: str = "",
    ) -> tuple[bool, SystemStateSnapshot]:
        """Merge partial readings into the running snapshot.

        Returns ``(tier_changed, snapshot)`` so the caller can decide to
        publish a ``SYSTEM_STATE`` event on the bus only on a transition.
        Unspecified fields keep their previous value, so a thermal-only
        watcher doesn't clobber the CPU sample taken seconds earlier.

        Note: ``gpu`` сохраняется в snapshot, но НЕ участвует в _classify —
        iGPU busy при inference это наша же работа, не повод душить FPS HUD.
        Высокая iGPU-нагрузка превращается в "neural pulse" сферы отдельно
        (см. JarvisHUD._on_system_state)."""
        prev = self._snap
        new_cpu = prev.cpu if cpu is None else float(cpu)
        new_ram = prev.ram if ram is None else float(ram)
        new_thermal = prev.thermal if thermal is None else float(thermal)
        new_gpu = prev.gpu if gpu is None else float(gpu)
        new_heavy = prev.heavy if heavy is None else bool(heavy)
        tier = self._classify(new_cpu, new_ram, new_thermal, new_heavy)
        snap = SystemStateSnapshot(
            load=tier, cpu=new_cpu, ram=new_ram, thermal=new_thermal, gpu=new_gpu,
            heavy=new_heavy, reason=reason or prev.reason, ts=time.time(),
        )
        self._snap = snap
        # Tier-change ИЛИ переход через neural-порог 80% — оба повода для
        # SYSTEM_STATE события (HUD должен включить neural pulse в обоих случаях).
        gpu_crossed = (prev.gpu < 80.0) != (new_gpu < 80.0)
        return (tier != prev.load) or gpu_crossed, snap
