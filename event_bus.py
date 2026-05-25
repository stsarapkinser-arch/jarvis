from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Literal

from singleton import Singleton

log = logging.getLogger("jarvis.bus")

Urgency = Literal["low", "normal", "high", "critical"]


class EventType(StrEnum):
    VOICE_INTENT = "VOICE_INTENT"
    DAEMON_ALERT = "DAEMON_ALERT"
    OS_EVENT = "OS_EVENT"
    PIXEL_EVENT = "PIXEL_EVENT"
    STATE_CHANGE = "STATE_CHANGE"
    TOKEN_STREAM = "TOKEN_STREAM"
    KWIN_ACTION = "KWIN_ACTION"
    # Aegis / Wraith / Navigator additions.
    AUDIO_FFT = "AUDIO_FFT"        # spectral bands while Piper speaks
    RECON_ALERT = "RECON_ALERT"    # Wraith findings (Wi-Fi vuln, intrusion, …)
    NMAP_SCAN = "NMAP_SCAN"        # nmap progress lines (host/port/state)
    HUD_OVERLAY = "HUD_OVERLAY"    # ad-hoc HUD payloads (graph nodes, projections)


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
