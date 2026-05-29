"""Pixel bridge unit tests.

These exercise the pure-helper surface of :class:`pixel.PixelBridge`:

* reminder app detection (``_is_reminder_app``)
* reminder event assembly + dedup (``_build_reminder_event``)
* call lifecycle: call_incoming registers an entry, call_ended drains it
  (``_active_calls`` / ``_publish_call_ended`` / ``_on_notification_removed``)
* TTL safety net for orphaned call notifications (``_sweep_stale_calls``)

We avoid the real D-Bus / KDE Connect stack: tests inject fake notification
payloads through ``_process_payload`` and inspect the captured events.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ui.pixel_rendererfrom src.common.event_bus import Event, EventBus, EventType
from src.common.singleton import Singleton


def _fresh_bridge() -> pixel.PixelBridge:
    """Reset the PixelBridge singleton + EventBus so each test starts clean."""
    Singleton.reset(pixel.PixelBridge)
    Singleton.reset(EventBus)
    return pixel.PixelBridge()


class _Recorder:
    """Captures every Event published on the bus during a test."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)


def _swap_bus(bridge: pixel.PixelBridge) -> _Recorder:
    rec = _Recorder()
    bridge.bus = rec  # type: ignore[assignment]
    return rec


# ──────────────────────── reminder detection ──────────────────────────────
def test_is_reminder_app_matches_common_apps():
    # _is_reminder_app expects already-lowercased app name (the call site in
    # _process_payload normalises with .lower()).
    assert pixel.PixelBridge._is_reminder_app("google calendar")
    assert pixel.PixelBridge._is_reminder_app("календарь")
    assert pixel.PixelBridge._is_reminder_app("todoist")
    assert pixel.PixelBridge._is_reminder_app("google keep")
    assert pixel.PixelBridge._is_reminder_app("будильник")
    # Empty / unrelated:
    assert not pixel.PixelBridge._is_reminder_app("")
    assert not pixel.PixelBridge._is_reminder_app("firefox")
    assert not pixel.PixelBridge._is_reminder_app("telegram")


def test_build_reminder_event_dedup_window():
    bridge = _fresh_bridge()
    payload = {
        "appName": "Google Calendar",
        "title": "Встреча с командой",
        "text": "10:00 в Konsole",
    }
    first = bridge._build_reminder_event(payload)
    assert first is not None
    assert first["kind"] == "reminder"
    assert first["app"] == "Google Calendar"
    assert first["title"] == "Встреча с командой"
    assert first["body"] == "10:00 в Konsole"

    # Identical second call within DEDUP window returns None.
    second = bridge._build_reminder_event(payload)
    assert second is None

    # Different reminder is not deduped.
    other = bridge._build_reminder_event({
        "appName": "Google Calendar",
        "title": "Обед",
        "text": "13:00",
    })
    assert other is not None
    assert other["title"] == "Обед"


def test_build_reminder_event_empty_payload_returns_none():
    bridge = _fresh_bridge()
    assert bridge._build_reminder_event({"appName": "Google Calendar"}) is None
    assert bridge._build_reminder_event({}) is None


async def test_process_payload_emits_reminder_event():
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    await bridge._process_payload({
        "appName": "Календарь",
        "title": "Запись к врачу",
        "text": "Завтра в 9 утра",
    })
    assert len(rec.events) == 1
    ev = rec.events[0]
    assert ev.type == EventType.PIXEL_EVENT
    assert ev.data["kind"] == "reminder"
    assert ev.data["title"] == "Запись к врачу"
    assert ev.data["body"] == "Завтра в 9 утра"


async def test_process_payload_does_not_emit_call_for_reminder_app():
    """A reminder containing the word "звонок" should not be misclassified
    as an incoming call, because the reminder branch returns early.
    """
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    await bridge._process_payload({
        "appName": "Будильник",
        "title": "Звонок будильника",
        "text": "07:00",
    })
    kinds = [e.data.get("kind") for e in rec.events if e.type == EventType.PIXEL_EVENT]
    assert "call_incoming" not in kinds
    assert "reminder" in kinds


# ──────────────────────── call lifecycle ─────────────────────────────────
async def test_call_incoming_registers_active_call_and_pauses_media():
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    paused = asyncio.Event()

    async def fake_pause() -> None:
        paused.set()

    bridge._pause_media = fake_pause  # type: ignore[assignment]

    await bridge._process_payload(
        {"appName": "Phone", "title": "Incoming call", "text": "from Mom"},
        dev_id="dev-1",
        public_id="pid-42",
    )

    assert paused.is_set()
    assert "pid-42" in bridge._active_calls
    info = bridge._active_calls["pid-42"]
    assert info["caller"]  # any non-empty caller string
    assert info["device"] == "dev-1"
    kinds = [e.data.get("kind") for e in rec.events if e.type == EventType.PIXEL_EVENT]
    assert "call_incoming" in kinds


async def test_notification_removed_resumes_media_and_emits_call_ended():
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    resumed = asyncio.Event()

    async def fake_resume() -> None:
        resumed.set()

    bridge._resume_media = fake_resume  # type: ignore[assignment]
    bridge._active_calls["pid-9"] = {
        "device": "dev-1",
        "caller": "Mom",
        "ts": time.time(),
    }

    await bridge._on_notification_removed("pid-9")

    assert resumed.is_set()
    assert "pid-9" not in bridge._active_calls
    call_ended = [
        e for e in rec.events
        if e.type == EventType.PIXEL_EVENT and e.data.get("kind") == "call_ended"
    ]
    assert len(call_ended) == 1
    assert call_ended[0].data["caller"] == "Mom"


async def test_notification_removed_ignores_non_call_ids():
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    resume_called = False

    async def fake_resume() -> None:
        nonlocal resume_called
        resume_called = True

    bridge._resume_media = fake_resume  # type: ignore[assignment]

    await bridge._on_notification_removed("unrelated-pid")

    assert not resume_called
    assert not rec.events


async def test_stale_call_sweep_fires_call_ended_via_ttl():
    bridge = _fresh_bridge()
    rec = _swap_bus(bridge)
    resumed = asyncio.Event()

    async def fake_resume() -> None:
        resumed.set()

    bridge._resume_media = fake_resume  # type: ignore[assignment]

    # Plant a call whose timestamp is older than the TTL.
    bridge._active_calls["stale-pid"] = {
        "device": "dev-1",
        "caller": "GhostCaller",
        "ts": time.time() - pixel.PixelBridge.CALL_NOTIFICATION_TTL_SEC - 5,
    }
    # Any unrelated payload triggers the sweep at the top of _process_payload.
    await bridge._process_payload({"appName": "Firefox", "title": "n/a"})

    assert resumed.is_set()
    assert "stale-pid" not in bridge._active_calls
    call_ended = [
        e for e in rec.events
        if e.type == EventType.PIXEL_EVENT and e.data.get("kind") == "call_ended"
    ]
    assert len(call_ended) == 1
    assert call_ended[0].data["caller"] == "GhostCaller"


# ──────────────────────── media helpers ─────────────────────────────────
async def test_resume_media_skips_when_playerctl_missing():
    """_resume_media is a no-op (and must not raise) without playerctl."""
    bridge = _fresh_bridge()
    with patch("pixel.shutil.which", return_value=None):
        await bridge._resume_media()  # must not raise


async def test_resume_media_invokes_playerctl_play(monkeypatch):
    bridge = _fresh_bridge()
    seen: dict[str, Any] = {}

    async def fake_exec(*args: str, **kwargs: Any) -> Any:
        seen["args"] = args
        seen["kwargs"] = kwargs

        class _P:
            pass

        return _P()

    with patch("pixel.shutil.which", return_value="/usr/bin/playerctl"):
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await bridge._resume_media()

    assert seen.get("args") is not None
    # playerctl --all-players play
    assert seen["args"][0].endswith("playerctl") or seen["args"][0] == "playerctl"
    assert "play" in seen["args"]
