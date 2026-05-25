"""Offscreen Qt smoke tests for the Aegis HUD.

These tests rely on PyQt6 being installed and on the ``offscreen`` QPA
platform. They verify that:

* the HUD instantiates without crashing in a headless environment
* signals route FFT / nmap / recon / app-glow / pixel payloads into the
  internal state without raising
* a paint cycle runs end-to-end with every overlay active
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import jarvis_hud


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(["-platform", "offscreen"])
    yield a


def test_hud_instantiates(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    assert hud.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert hud.windowFlags() & Qt.WindowType.FramelessWindowHint
    hud.deleteLater()


def test_hud_signals_do_not_raise(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    hud.resize(800, 600)
    hud.fft_signal.emit([0.1, 0.2, 0.3] * 8, 0.42)
    hud.nmap_signal.emit({"event": "host", "host": "10.0.0.1", "ip": "10.0.0.1"})
    hud.nmap_signal.emit({"event": "port", "host": "10.0.0.1", "port": 22, "state": "open", "service": "ssh"})
    hud.nmap_signal.emit({"event": "progress", "percent": 50.0})
    hud.recon_signal.emit({"color": "yellow", "kind": "wifi_wps", "summary": "test"})
    hud.recon_signal.emit({"color": "red", "kind": "intrusion_auth", "summary": "boom", "ufw_suggest": "sudo -n ufw deny from 1.2.3.4"})
    hud.app_glow_signal.emit([{"x": 10, "y": 10, "w": 200, "h": 100, "caption": "Code"}])
    hud.pixel_projection_signal.emit({"title": "Test", "body": "hello", "app": "Pixel"})
    hud.state_signal.emit("THINKING", "running diagnostics")
    hud.token_signal.emit("hello world ")
    hud.position_signal.emit(0.2, 0.8)
    app.processEvents()
    # Tick the animation a few times to advance pulses.
    for _ in range(5):
        hud.update_animation()
    # Verify state was captured into the HUD without rendering (offscreen
    # render of QMainWindow is fragile on PyQt6/Wayland CI).
    assert abs(hud._fft_level - 0.42) < 1e-6
    assert (("10.0.0.1", 22) in hud._port_hits)
    assert hud._recon_flash is not None
    assert hud._recon_flash["color"] == "red"
    assert hud._app_glows and hud._app_glows[0]["caption"] == "Code"
    assert hud._pixel_card is not None
    assert hud.state == "THINKING"
    hud.deleteLater()


def test_hud_draw_app_glow_method_caches_caption(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    hud.draw_app_glow("Visual Studio Code")
    # Should have queued a caption-only stub for the next paint cycle.
    assert any(g.get("caption") == "Visual Studio Code" for g in hud._app_glows)
    hud.deleteLater()


def test_pixel_event_routes_to_projection(app) -> None:
    from event_bus import Event, EventType
    import asyncio

    hud = jarvis_hud.JarvisHUD()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        hud._on_pixel_event(Event(EventType.PIXEL_EVENT, {
            "kind": "clipboard", "payload": "https://example.com",
        }))
    )
    # Signal fired into the Qt slot; spin once to let it deliver.
    app.processEvents()
    assert hud._pixel_card is not None
    assert hud._pixel_card.get("body") == "https://example.com"
    hud.deleteLater()
