from __future__ import annotations

import asyncio
import logging
import sys
import threading

import qasync
from PyQt6.QtWidgets import QApplication

from core import Jarvis
from daemon_swarm import DaemonSwarm
from event_bus import Event, EventBus, EventType
from jarvis_hud import JarvisHUD
from kwin import KWinOrchestrator
from main import JarvisMain
from pixel import PixelBridge
from sentinel import Sentinel

log = logging.getLogger("jarvis.boot")

HUD_REPOSITION_PERIOD_SEC = 6.0


async def hud_layout_watcher(bus: EventBus, kwin: KWinOrchestrator) -> None:
    """Periodically inspect the window layout and ask the HUD to re-anchor
    to the least-occupied screen quadrant. Best-effort: silent on Wayland
    without kdotool/wmctrl."""
    log.info("HUD layout watcher started")
    last: tuple[float, float] | None = None
    while True:
        try:
            corner = await kwin.suggest_hud_corner()
        except Exception:
            log.exception("suggest_hud_corner failed")
            corner = None
        if corner and corner != last:
            await bus.publish(Event(
                EventType.KWIN_ACTION,
                {"kind": "hud_reposition", "x_ratio": corner[0], "y_ratio": corner[1]},
            ))
            last = corner
        await asyncio.sleep(HUD_REPOSITION_PERIOD_SEC)


async def amain() -> None:
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())

    hud = JarvisHUD()
    hud.show()

    jarvis = Jarvis()
    kwin = KWinOrchestrator()
    hud.subscribe_to_bus(bus)

    bus.subscribe(EventType.VOICE_INTENT, jarvis.on_voice_intent)
    bus.subscribe(EventType.DAEMON_ALERT, jarvis.on_daemon_alert)
    bus.subscribe(EventType.OS_EVENT, jarvis.on_os_event)
    bus.subscribe(EventType.PIXEL_EVENT, jarvis.on_pixel_event)

    asyncio.create_task(bus.run(), name="event-bus")
    await DaemonSwarm().start_all()
    await Sentinel().start_all()
    await PixelBridge().start()
    asyncio.create_task(hud_layout_watcher(bus, kwin), name="hud-layout")

    voice = JarvisMain()
    threading.Thread(target=voice.run, args=(bus,), daemon=True).start()

    print("💎 JARVIS Protocol Fully Deployed.")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.create_task(amain())
        loop.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
