"""Aegis full-screen AR HUD (cybernetic overlay).

The HUD is a borderless, click-through, full-screen ``QMainWindow`` that
draws every Jarvis subsystem on top of the live Plasma desktop.

Layers (drawn back-to-front):

1. **Scan grid** — faint angular brackets, time ticker, protocol header.
2. **App glow** — neon rectangles + corner brackets pulsing around windows
   we are currently observing (VS Code with a syntax error, konsole running
   a long command, KWin shortcut target). Populated via
   :class:`KWinOrchestrator` and surfaced through ``app_glow_signal``.
3. **Port matrix** — when ``nmap`` is running, a grid of pulsing dots
   appears in the lower band; a fresh open-port hit pops to full intensity
   and fades over a second. Closed/filtered ports stay dim.
4. **Nav graph** — five anchor nodes (CORE, MEMORY, HARDWARE, PIXEL,
   OLLAMA) connected by animated energy lines while Jarvis is THINKING /
   SPEAKING. Each ``<thought>`` token routes one line.
5. **Pixel AR card** — Pixel notifications fly in as an angled-perspective
   card on the right edge, dwell briefly, then fade.
6. **Core sphere** — pulsing wireframe sphere with 24 latitude/longitude
   bars whose radius is driven by the FFT bands of the Piper TTS stream.
   Replaces the old concentric circles.
7. **Recon halo** — a yellow vertical glow when ``RECON_ALERT.color ==
   "yellow"`` (Wi-Fi vulnerability), a red full-screen wash for
   ``"red"`` (intrusion). Both auto-decay after a few seconds.
8. **Token ticker** — same tape as before, kept on the bottom edge.

The HUD subscribes to the existing ``EventBus`` and never blocks the Qt
event loop: every cross-thread update arrives through ``pyqtSignal``.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from collections import deque
from typing import Any

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPointF, QRectF, QPropertyAnimation, QEasingCurve,
    pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import QApplication, QMainWindow

log = logging.getLogger("jarvis.hud")

# --- Palette ----------------------------------------------------------------
STATE_COLORS: dict[str, QColor] = {
    "IDLE":     QColor(0,   180, 255, 100),
    "THINKING": QColor(0,   255, 150, 200),
    "SPEAKING": QColor(255, 255, 255, 180),
    "ALERT":    QColor(255, 20,  50,  255),
}
NEON_CYAN   = QColor(0,   255, 255, 220)
NEON_AMBER  = QColor(255, 200, 60,  220)
NEON_RED    = QColor(255, 40,  60,  255)
NEON_GREEN  = QColor(80,  255, 160, 220)
NEON_VIOLET = QColor(180, 80,  255, 220)

# --- Geometry tuning --------------------------------------------------------
TICKER_MAX_CHARS = 220
TICKER_HEIGHT    = 28
FADE_IN_MS       = 900
CORE_LERP        = 0.08
FFT_BANDS        = 24
SPHERE_LAT_RINGS = 9          # horizontal rings on the wireframe sphere
SPHERE_LON_BARS  = 24         # vertical bars on the wireframe sphere
SPHERE_BASE_R    = 95.0       # base sphere radius in pixels at fullscreen 1080p
PORT_MATRIX_COLS = 32
PORT_MATRIX_ROWS = 8
APP_GLOW_TTL_SEC = 4.0
RECON_FLASH_TTL  = 5.0
NAV_NODES = ("CORE", "MEMORY", "HARDWARE", "PIXEL", "OLLAMA")
NAV_NODE_RADIUS = 22


class JarvisHUD(QMainWindow):
    # Qt-thread-only signals: cross-thread publishers call .emit().
    state_signal           = pyqtSignal(str, str)
    token_signal           = pyqtSignal(str)
    position_signal        = pyqtSignal(float, float)
    fft_signal             = pyqtSignal(list, float)
    nmap_signal            = pyqtSignal(dict)
    recon_signal           = pyqtSignal(dict)
    app_glow_signal        = pyqtSignal(list)          # list[dict(x,y,w,h,caption,color?)]
    nav_graph_signal       = pyqtSignal(str)           # route a thought token
    pixel_projection_signal = pyqtSignal(dict)         # {'app': str, 'title': str, 'body': str}

    def __init__(self) -> None:
        super().__init__()
        self.state_signal.connect(self.set_state)
        self.token_signal.connect(self._append_token)
        self.position_signal.connect(self.set_core_position)
        self.fft_signal.connect(self._on_fft)
        self.nmap_signal.connect(self._on_nmap)
        self.recon_signal.connect(self._on_recon)
        self.app_glow_signal.connect(self._on_app_glow)
        self.nav_graph_signal.connect(self._on_nav_token)
        self.pixel_projection_signal.connect(self._on_pixel_projection)
        self._bus: Any = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.0)

        # Aegis is full-screen by spec. Fall back to setGeometry on offscreen
        # platforms (used by the test suite) so the widget still has a size.
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

        self.state: str = "IDLE"
        self.pulse_radius: float = SPHERE_BASE_R
        self.rotation_angle: float = 0.0
        self.thinking_text: str = ""
        self.ticker_buffer: deque[str] = deque(maxlen=TICKER_MAX_CHARS)
        self.token_offset: float = 0.0

        # Core sphere anchor — same lerp model as before so the new layer
        # animates in tandem with the rest of the HUD.
        self._target_xr: float = 0.85
        self._target_yr: float = 0.85
        self._current_xr: float = 0.85
        self._current_yr: float = 0.85

        # FFT state
        self._fft_bands: list[float] = [0.0] * FFT_BANDS
        self._fft_level: float = 0.0
        self._fft_decay_ts: float = 0.0

        # Port matrix state: keyed by (host, port) → (state, ts).
        self._port_hits: dict[tuple[str, int], tuple[str, float]] = {}
        self._nmap_progress: float = 0.0
        self._nmap_active_host: str = ""

        # Recon flash
        self._recon_flash: dict | None = None        # last finding with color
        self._recon_pending_ufw: str | None = None   # last suggested ufw block

        # App glow (neon brackets around windows)
        self._app_glows: list[dict] = []

        # Nav graph state
        self._nav_pulses: deque[dict] = deque(maxlen=40)

        # Pixel projection card
        self._pixel_card: dict | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(FADE_IN_MS)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ----- Qt lifecycle -----
    def show_fullscreen(self) -> None:
        """Activate the Aegis fullscreen mode. Public so start_jarvis can defer."""
        self.showFullScreen()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    # ----- Public slots -----
    @pyqtSlot(str, str)
    def set_state(self, new_state: str, thought: str = "") -> None:
        self.state = new_state
        self.thinking_text = thought
        # Each new thought routes through CORE → MEMORY → HARDWARE so the
        # Navigator layer shows live cognition.
        if thought:
            self._enqueue_thought_pulse(thought)
        self.update()

    @pyqtSlot(str)
    def _append_token(self, tok: str) -> None:
        for ch in tok:
            self.ticker_buffer.append(ch)
        # One token in the LLM stream = one micro-pulse on the nav graph.
        self.nav_graph_signal.emit(tok)

    @pyqtSlot(float, float)
    def set_core_position(self, x_ratio: float, y_ratio: float) -> None:
        self._target_xr = max(0.08, min(0.92, x_ratio))
        self._target_yr = max(0.10, min(0.90, y_ratio))

    @pyqtSlot(list, float)
    def _on_fft(self, bands: list, level: float) -> None:
        if not isinstance(bands, list):
            return
        # Trim / pad to expected size so a misconfigured analyzer can't
        # break the renderer.
        if len(bands) >= FFT_BANDS:
            self._fft_bands = [float(x) for x in bands[:FFT_BANDS]]
        else:
            padded = list(bands) + [0.0] * (FFT_BANDS - len(bands))
            self._fft_bands = [float(x) for x in padded]
        self._fft_level = max(0.0, min(1.0, float(level)))
        self._fft_decay_ts = time.time()

    @pyqtSlot(dict)
    def _on_nmap(self, payload: dict) -> None:
        kind = payload.get("event") or payload.get("kind")
        if kind == "port":
            host = str(payload.get("host", "?"))
            port = int(payload.get("port", 0))
            state = str(payload.get("state", "open"))
            self._port_hits[(host, port)] = (state, time.time())
            # Cap grid to the most recent COLS*ROWS entries to bound memory.
            cap = PORT_MATRIX_COLS * PORT_MATRIX_ROWS * 4
            if len(self._port_hits) > cap:
                # Drop the oldest half.
                sortable = sorted(self._port_hits.items(), key=lambda kv: kv[1][1])
                for key, _ in sortable[: len(sortable) // 2]:
                    self._port_hits.pop(key, None)
        elif kind == "host":
            self._nmap_active_host = str(payload.get("host", ""))
        elif kind == "progress":
            try:
                self._nmap_progress = float(payload.get("percent", 0.0)) / 100.0
            except (TypeError, ValueError):
                pass

    @pyqtSlot(dict)
    def _on_recon(self, payload: dict) -> None:
        # Stamp the timestamp so the renderer can decay the flash.
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        self._recon_flash = payload
        self._recon_pending_ufw = payload.get("ufw_suggest")

    @pyqtSlot(list)
    def _on_app_glow(self, payload: list) -> None:
        now = time.time()
        glows: list[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                glows.append({
                    "x": int(item.get("x", 0)),
                    "y": int(item.get("y", 0)),
                    "w": int(item.get("w", 0)),
                    "h": int(item.get("h", 0)),
                    "caption": str(item.get("caption", ""))[:80],
                    "color": str(item.get("color", "cyan")),
                    "ts": float(item.get("ts", now)),
                })
            except (TypeError, ValueError):
                continue
        self._app_glows = glows

    @pyqtSlot(str)
    def _on_nav_token(self, _tok: str) -> None:
        # A token-level pulse: CORE → random sibling.
        if self.state == "IDLE":
            return
        self._enqueue_thought_pulse("")

    @pyqtSlot(dict)
    def _on_pixel_projection(self, payload: dict) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        self._pixel_card = payload

    # ----- Animation tick -----
    def update_animation(self) -> None:
        self.rotation_angle = (self.rotation_angle + 1.2) % 360.0
        t = time.time()

        # Core radius reacts to FFT level when available, otherwise falls back
        # to the state-driven pulse the old HUD used so IDLE still breathes.
        fft_age = t - self._fft_decay_ts
        if fft_age < 0.6 and self._fft_level > 0.01:
            target = SPHERE_BASE_R * (0.9 + 0.6 * self._fft_level)
        elif self.state == "IDLE":
            target = SPHERE_BASE_R + 5 * math.sin(t * 3)
        elif self.state == "THINKING":
            target = SPHERE_BASE_R + 14 * math.sin(t * 10)
        elif self.state == "SPEAKING":
            target = SPHERE_BASE_R + 9 * math.sin(t * 6)
        else:
            target = SPHERE_BASE_R + 30 * math.sin(t * 20)
        self.pulse_radius = self.pulse_radius * 0.8 + target * 0.2

        self.token_offset = (self.token_offset + 1.5) % 10000.0
        self._current_xr += (self._target_xr - self._current_xr) * CORE_LERP
        self._current_yr += (self._target_yr - self._current_yr) * CORE_LERP

        # Decay FFT bands towards zero between updates so silence calms the sphere.
        if fft_age > 0.12:
            self._fft_bands = [b * 0.85 for b in self._fft_bands]
            self._fft_level *= 0.85

        # Cull stale port hits beyond 60s to keep the grid fresh.
        cutoff = t - 60.0
        if self._port_hits:
            stale = [k for k, (_, ts) in self._port_hits.items() if ts < cutoff]
            for k in stale:
                self._port_hits.pop(k, None)

        # Cull old app glows.
        if self._app_glows:
            self._app_glows = [
                g for g in self._app_glows if t - g.get("ts", t) < APP_GLOW_TTL_SEC
            ]

        # Decay recon flash.
        if self._recon_flash and t - self._recon_flash.get("ts", t) > RECON_FLASH_TTL:
            self._recon_flash = None

        # Pixel projection auto-dismiss after 6 seconds.
        if self._pixel_card and t - self._pixel_card.get("ts", t) > 6.0:
            self._pixel_card = None

        self.update()

    # ----- Internals -----
    def _core_center(self, w: int, h: int) -> tuple[int, int]:
        return int(w * self._current_xr), int(h * self._current_yr)

    def _state_color(self) -> QColor:
        return STATE_COLORS.get(self.state, STATE_COLORS["IDLE"])

    def _enqueue_thought_pulse(self, text: str) -> None:
        """Push a nav-graph pulse. ``text`` empty → random node."""
        if NAV_NODES:
            # Hash to a deterministic destination so similar tokens land on the
            # same node — feels less like noise, more like routing.
            seed = (sum(ord(c) for c in text) if text else int(time.time() * 11)) % (len(NAV_NODES) - 1)
            dest = NAV_NODES[1 + seed]  # never CORE → CORE
            self._nav_pulses.append({
                "dst": dest,
                "ts": time.time(),
                "label": text[:32],
            })

    # ----- Painting -----
    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        color = self._state_color()

        self._draw_scan_grid(painter, w, h, color)
        self._draw_app_glow(painter)
        self._draw_port_matrix(painter, w, h)
        self._draw_nav_graph(painter, w, h)
        self._draw_pixel_card(painter, w, h)
        self._draw_core_sphere(painter, w, h, color)
        self._draw_ticker(painter, w, h, color)
        self._draw_recon_overlay(painter, w, h)

        painter.setPen(color)
        painter.setFont(QFont("Monospace", 10))
        host_hint = f" :: TARGET_{self._nmap_active_host}" if self._nmap_active_host else ""
        ufw_hint = f" :: UFW_PENDING" if self._recon_pending_ufw else ""
        painter.drawText(
            20, 40,
            f"JARVIS INTEL N100 :: PROTOCOL_{self.state} :: MEM_SYNC_OK{host_hint}{ufw_hint}",
        )

        if self.state == "ALERT" and not self._recon_flash:
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(255, 0, 0, 50))
            grad.setColorAt(0.1, QColor(0, 0, 0, 0))
            grad.setColorAt(0.9, QColor(0, 0, 0, 0))
            grad.setColorAt(1.0, QColor(255, 0, 0, 50))
            painter.fillRect(self.rect(), grad)

    # --- Layer 1: faint scan grid -----------------------------------------
    def _draw_scan_grid(self, painter: QPainter, w: int, h: int, color: QColor) -> None:
        grid_color = QColor(color.red(), color.green(), color.blue(), 22)
        pen = QPen(grid_color)
        pen.setWidth(1)
        painter.setPen(pen)
        step = 96
        for x in range(0, w, step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            painter.drawLine(0, y, w, y)

        # Corner brackets so it really looks like a HUD.
        bracket = QColor(color.red(), color.green(), color.blue(), 180)
        pen = QPen(bracket)
        pen.setWidth(2)
        painter.setPen(pen)
        L = 36
        for cx, cy, dx, dy in (
            (8,     8,     +1, +1),
            (w - 8, 8,     -1, +1),
            (8,     h - 8, +1, -1),
            (w - 8, h - 8, -1, -1),
        ):
            painter.drawLine(cx, cy, cx + dx * L, cy)
            painter.drawLine(cx, cy, cx, cy + dy * L)

    # --- Layer 2: neon app glow -------------------------------------------
    def draw_app_glow(self, window_title: str) -> None:
        """Public helper used by the core when it wants to glow a single
        application without composing a full payload (e.g. quick highlight on
        a failed bash command). The actual drawing happens in
        :meth:`_draw_app_glow`; this just stages a sentinel entry that the
        next paint cycle picks up. The KWin orchestrator owns the geometry —
        we only carry the caption hint until it answers."""
        self._app_glows.append({
            "x": 0, "y": 0, "w": 0, "h": 0,
            "caption": window_title[:80],
            "color": "amber",
            "ts": time.time(),
        })

    def _draw_app_glow(self, painter: QPainter) -> None:
        if not self._app_glows:
            return
        now = time.time()
        for g in self._app_glows:
            if g["w"] <= 0 or g["h"] <= 0:
                # No geometry yet (only a caption hint) — skip; KWin watcher
                # will resolve it on the next tick.
                continue
            age = now - g.get("ts", now)
            if age >= APP_GLOW_TTL_SEC:
                continue
            tail = max(0.0, 1.0 - age / APP_GLOW_TTL_SEC)
            base = NEON_AMBER if g.get("color") == "amber" else NEON_CYAN
            ring = QColor(base.red(), base.green(), base.blue(), int(240 * tail))
            pen = QPen(ring)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(g["x"], g["y"], g["w"], g["h"]), 8, 8)
            # Iron-Man style corner brackets.
            L = 24
            for cx, cy, dx, dy in (
                (g["x"],            g["y"],            +1, +1),
                (g["x"] + g["w"],   g["y"],            -1, +1),
                (g["x"],            g["y"] + g["h"],   +1, -1),
                (g["x"] + g["w"],   g["y"] + g["h"],   -1, -1),
            ):
                painter.drawLine(cx, cy, cx + dx * L, cy)
                painter.drawLine(cx, cy, cx, cy + dy * L)
            painter.setFont(QFont("Monospace", 9))
            painter.drawText(g["x"] + 6, g["y"] - 6, g.get("caption", ""))

    # --- Layer 3: port matrix --------------------------------------------
    def _port_color(self, state: str, intensity: float) -> QColor:
        if state.startswith("open"):
            c = NEON_GREEN
        elif state.startswith("closed"):
            c = QColor(255, 80, 80, 220)
        else:
            c = QColor(255, 220, 100, 220)
        i = max(0.15, min(1.0, intensity))
        return QColor(c.red(), c.green(), c.blue(), int(c.alpha() * i))

    def _draw_port_matrix(self, painter: QPainter, w: int, h: int) -> None:
        if not self._port_hits and self._nmap_progress <= 0.01:
            return
        margin_x = 60
        margin_y = h - 220
        cell_w = (w - 2 * margin_x) / PORT_MATRIX_COLS
        cell_h = 18
        now = time.time()

        # Title + progress bar.
        painter.setPen(NEON_CYAN)
        painter.setFont(QFont("Monospace", 11))
        painter.drawText(
            margin_x, margin_y - 14,
            f"PORT MATRIX :: {len(self._port_hits)} hits"
            + (f" :: scan {int(self._nmap_progress*100)}%" if self._nmap_progress else ""),
        )
        if self._nmap_progress > 0:
            bar_w = int((w - 2 * margin_x) * self._nmap_progress)
            painter.fillRect(margin_x, margin_y - 6, bar_w, 3, NEON_CYAN)

        painter.setPen(Qt.PenStyle.NoPen)
        # Sort ports so each port lands on a predictable cell (col = port % cols,
        # row = (port // cols) % rows). Stable, easy to scan visually.
        for (host, port), (state, ts) in self._port_hits.items():
            age = now - ts
            if age > 30.0:
                continue
            intensity = max(0.15, 1.0 - age / 30.0)
            col = port % PORT_MATRIX_COLS
            row = (port // PORT_MATRIX_COLS) % PORT_MATRIX_ROWS
            x = margin_x + col * cell_w + cell_w * 0.5
            y = margin_y + row * cell_h + cell_h * 0.5
            painter.setBrush(self._port_color(state, intensity))
            r = 5 + 4 * intensity
            painter.drawEllipse(QPointF(x, y), r, r)

    # --- Layer 4: nav graph ------------------------------------------------
    def _nav_node_positions(self, w: int, h: int) -> dict[str, QPointF]:
        cx, cy = self._core_center(w, h)
        # Place nav nodes in a half-ring opposite of the core (so the sphere
        # doesn't sit on top of the graph).
        offset_dir = -1 if self._current_xr > 0.5 else +1
        anchor_x = cx + offset_dir * int(min(w, h) * 0.22)
        anchor_y = cy
        positions = {"CORE": QPointF(cx, cy)}
        radius = min(w, h) * 0.18
        for i, node in enumerate(NAV_NODES[1:]):
            theta = math.radians(-65 + i * 35)
            px = anchor_x + math.cos(theta) * radius * offset_dir
            py = anchor_y + math.sin(theta) * radius
            positions[node] = QPointF(px, py)
        return positions

    def _draw_nav_graph(self, painter: QPainter, w: int, h: int) -> None:
        if self.state == "IDLE" and not self._nav_pulses:
            return
        positions = self._nav_node_positions(w, h)
        now = time.time()

        # Draw quiescent links (faint).
        link_pen = QPen(QColor(0, 220, 255, 70))
        link_pen.setWidth(1)
        painter.setPen(link_pen)
        for node, p in positions.items():
            if node == "CORE":
                continue
            painter.drawLine(positions["CORE"], p)

        # Animated pulses along links.
        for pulse in list(self._nav_pulses):
            age = now - pulse.get("ts", now)
            if age > 1.2:
                self._nav_pulses.popleft() if self._nav_pulses and self._nav_pulses[0] is pulse else None
                continue
            dst = positions.get(pulse.get("dst", ""))
            if dst is None:
                continue
            t = min(1.0, age / 0.9)
            src = positions["CORE"]
            x = src.x() + (dst.x() - src.x()) * t
            y = src.y() + (dst.y() - src.y()) * t
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 255, 200, int(220 * (1.0 - t * 0.4))))
            painter.drawEllipse(QPointF(x, y), 4 + 3 * (1.0 - t), 4 + 3 * (1.0 - t))

        # Draw nodes with labels.
        painter.setFont(QFont("Monospace", 9))
        for node, p in positions.items():
            if node == "CORE":
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 80, 120, 180))
            painter.drawEllipse(p, NAV_NODE_RADIUS, NAV_NODE_RADIUS)
            painter.setPen(QPen(QColor(0, 220, 255, 220), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(p, NAV_NODE_RADIUS, NAV_NODE_RADIUS)
            painter.setPen(QColor(220, 240, 255, 230))
            painter.drawText(
                int(p.x() - NAV_NODE_RADIUS * 2.2),
                int(p.y() + NAV_NODE_RADIUS + 14),
                int(NAV_NODE_RADIUS * 4.4), 16,
                Qt.AlignmentFlag.AlignCenter,
                node,
            )

        # Last thought caption next to CORE.
        if self.thinking_text:
            painter.setPen(QColor(220, 240, 255, 180))
            painter.setFont(QFont("Monospace", 10))
            cx, cy = self._core_center(w, h)
            painter.drawText(
                cx - 240, cy - SPHERE_BASE_R - 70, 480, 50,
                Qt.AlignmentFlag.AlignCenter,
                self.thinking_text[:160],
            )

    # --- Layer 5: Pixel AR card -------------------------------------------
    def _draw_pixel_card(self, painter: QPainter, w: int, h: int) -> None:
        card = self._pixel_card
        if card is None:
            return
        now = time.time()
        age = now - card.get("ts", now)
        if age >= 6.0:
            return
        fade = 1.0 - abs(age - 3.0) / 3.0
        fade = max(0.0, min(1.0, fade))
        card_w, card_h = 360, 140
        x = w - card_w - 80
        y = 120

        # Perspective trapezoid to fake AR depth.
        poly = QPolygonF([
            QPointF(x + 30, y),
            QPointF(x + card_w, y + 14),
            QPointF(x + card_w - 30, y + card_h),
            QPointF(x, y + card_h - 14),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 30, 50, int(180 * fade)))
        painter.drawPolygon(poly)
        painter.setPen(QPen(QColor(0, 220, 255, int(220 * fade)), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)

        painter.setPen(QColor(0, 240, 255, int(230 * fade)))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 24, y + 24, "PIXEL :: INCOMING")

        painter.setPen(QColor(255, 255, 255, int(230 * fade)))
        painter.setFont(QFont("Monospace", 12))
        title = str(card.get("title") or card.get("app") or "Pixel")[:48]
        painter.drawText(x + 24, y + 50, title)

        painter.setFont(QFont("Monospace", 10))
        body = str(card.get("body") or card.get("text") or card.get("ticker") or "")[:120]
        painter.drawText(
            x + 24, y + 64, card_w - 48, card_h - 70,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            body,
        )

    # --- Layer 6: core sphere (FFT-driven) --------------------------------
    def _draw_core_sphere(self, painter: QPainter, w: int, h: int, color: QColor) -> None:
        cx, cy = self._core_center(w, h)
        r0 = self.pulse_radius

        # Outer soft glow uses the FFT level so the whole sphere brightens
        # with voice volume.
        outer = QRadialGradient(cx, cy, r0 * 3.0)
        glow_alpha = int(90 + 110 * self._fft_level)
        outer.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), glow_alpha))
        outer.setColorAt(0.5, QColor(color.red(), color.green(), color.blue(), 30))
        outer.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(cx, cy), int(r0 * 3.0), int(r0 * 3.0))

        # Wireframe sphere: SPHERE_LAT_RINGS horizontal ellipses + SPHERE_LON_BARS
        # vertical bars whose radius is modulated by the FFT band at that angle.
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rot = math.radians(self.rotation_angle)
        bars = self._fft_bands or [0.0] * FFT_BANDS

        # Vertical longitude bars (the actual "equalizer" feel).
        for i in range(SPHERE_LON_BARS):
            band = bars[i % len(bars)]
            displaced = r0 * (1.0 + 0.35 * band)
            theta = (i / SPHERE_LON_BARS) * math.tau + rot
            # Project a vertical great-circle: cos(theta) controls horizontal
            # foreshortening, giving a depth illusion.
            depth = math.cos(theta)
            rx = abs(depth) * displaced
            if rx < 2:
                continue
            path = QPainterPath()
            path.moveTo(cx, cy - displaced)
            ctrl = QPointF(cx + (1 if depth >= 0 else -1) * rx, cy)
            path.quadTo(ctrl, QPointF(cx, cy + displaced))
            painter.drawPath(path)

        # Horizontal latitude rings.
        for i in range(1, SPHERE_LAT_RINGS):
            phi = -math.pi / 2 + (i / SPHERE_LAT_RINGS) * math.pi
            ry = abs(math.cos(phi)) * r0
            ring_r = abs(math.cos(phi)) * r0
            y_offset = math.sin(phi) * r0
            painter.drawEllipse(
                QRectF(cx - ring_r, cy + y_offset - ry * 0.18, ring_r * 2, ry * 0.36)
            )

        # Inner solid core.
        inner = QRadialGradient(cx, cy, r0 * 0.6)
        inner.setColorAt(0.0, color)
        inner.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner)
        painter.drawEllipse(QPoint(cx, cy), int(r0 * 0.6), int(r0 * 0.6))

        # Rotating angle ticks reading the upper hemisphere.
        ticks = 18
        painter.setPen(QPen(color, 1))
        for k in range(ticks):
            a = (k / ticks) * math.tau + rot * 0.5
            x1 = cx + math.cos(a) * (r0 + 8)
            y1 = cy + math.sin(a) * (r0 + 8)
            x2 = cx + math.cos(a) * (r0 + 14 + 6 * self._fft_level)
            y2 = cy + math.sin(a) * (r0 + 14 + 6 * self._fft_level)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # FFT particle ring — drawPoints is cheap on Intel N100 because Qt
        # batches them into a single OpenGL call (no per-point fill).
        if any(b > 0.02 for b in bars):
            pts: list[QPointF] = []
            for i, band in enumerate(bars):
                if band < 0.02:
                    continue
                a = (i / len(bars)) * math.tau + rot * 0.3
                rr = r0 * (1.6 + 0.4 * band)
                # Two points per band — one inner, one outer — gives a denser ring.
                pts.append(QPointF(cx + math.cos(a) * rr, cy + math.sin(a) * rr))
                pts.append(QPointF(cx + math.cos(a) * (rr - 6), cy + math.sin(a) * (rr - 6)))
            pen = QPen(QColor(color.red(), color.green(), color.blue(), 230))
            pen.setWidth(2)
            painter.setPen(pen)
            for p in pts:
                painter.drawPoint(p)

    # --- Layer 7: ticker ---------------------------------------------------
    def _draw_ticker(self, painter: QPainter, w: int, h: int, color: QColor) -> None:
        if not self.ticker_buffer:
            return
        text = "".join(self.ticker_buffer).replace("\n", " ⏎ ")
        painter.setFont(QFont("Monospace", 11))
        shadow = QColor(0, 0, 0, 160)
        y = h - TICKER_HEIGHT
        x_offset = int(-self.token_offset) % max(1, painter.fontMetrics().horizontalAdvance(text) + 60)
        painter.setPen(shadow)
        painter.drawText(20 - x_offset + 1, y + 1, text)
        painter.setPen(color)
        painter.drawText(20 - x_offset, y, text)

    # --- Layer 8: recon overlay -------------------------------------------
    def _draw_recon_overlay(self, painter: QPainter, w: int, h: int) -> None:
        flash = self._recon_flash
        if flash is None:
            return
        now = time.time()
        age = now - flash.get("ts", now)
        if age >= RECON_FLASH_TTL:
            return
        decay = 1.0 - (age / RECON_FLASH_TTL)
        decay = max(0.0, min(1.0, decay))

        color = flash.get("color", "yellow")
        blink = 0.5 + 0.5 * math.sin(now * 6.0)
        if color == "red":
            wash = QColor(255, 30, 60, int(70 * decay * (0.5 + 0.5 * blink)))
            painter.fillRect(self.rect(), wash)
            edge = QColor(255, 60, 90, int(220 * decay))
        else:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(255, 200, 50, int(40 * decay * blink)))
            grad.setColorAt(0.5, QColor(255, 200, 50, int(10 * decay)))
            grad.setColorAt(1.0, QColor(255, 200, 50, int(40 * decay * blink)))
            painter.fillRect(self.rect(), grad)
            edge = QColor(255, 220, 100, int(220 * decay))

        pen = QPen(edge)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

        # Recon line in the upper-right corner.
        painter.setFont(QFont("Monospace", 12))
        painter.setPen(QColor(255, 230, 120, int(240 * decay)) if color != "red" else QColor(255, 90, 90, int(240 * decay)))
        summary = str(flash.get("summary", ""))[:80]
        kind = str(flash.get("kind", ""))
        painter.drawText(w - 720, 70, 700, 22, Qt.AlignmentFlag.AlignRight, f"[{kind}] {summary}")

        if self._recon_pending_ufw:
            painter.setPen(QColor(255, 120, 120, int(240 * decay)))
            painter.setFont(QFont("Monospace", 10))
            painter.drawText(
                w - 720, 96, 700, 22,
                Qt.AlignmentFlag.AlignRight,
                f"UFW ready: {self._recon_pending_ufw}",
            )

    # ----- Bus wiring -----
    def subscribe_to_bus(self, bus: Any) -> None:
        from event_bus import EventType
        self._bus = bus
        bus.subscribe(EventType.STATE_CHANGE, self._on_state)
        bus.subscribe(EventType.TOKEN_STREAM, self._on_token)
        bus.subscribe(EventType.KWIN_ACTION, self._on_kwin_action)
        bus.subscribe(EventType.AUDIO_FFT, self._on_audio_fft)
        bus.subscribe(EventType.NMAP_SCAN, self._on_nmap_event)
        bus.subscribe(EventType.RECON_ALERT, self._on_recon_event)
        bus.subscribe(EventType.HUD_OVERLAY, self._on_hud_overlay)
        bus.subscribe(EventType.PIXEL_EVENT, self._on_pixel_event)

    async def _on_state(self, event) -> None:
        data = event.data
        if isinstance(data, tuple) and len(data) == 2:
            state, thought = data
        else:
            state, thought = str(data), ""
        self.state_signal.emit(str(state), str(thought))

    async def _on_token(self, event) -> None:
        self.token_signal.emit(str(event.data))

    async def _on_kwin_action(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        kind = data.get("kind")
        if kind == "hud_reposition":
            self.position_signal.emit(
                float(data.get("x_ratio", 0.85)),
                float(data.get("y_ratio", 0.85)),
            )
        elif kind == "app_glow":
            payload = data.get("windows") or []
            if isinstance(payload, list):
                self.app_glow_signal.emit(list(payload))

    async def _on_audio_fft(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        bands = data.get("bands") or []
        if not isinstance(bands, list):
            return
        try:
            level = float(data.get("level", 0.0))
        except (TypeError, ValueError):
            level = 0.0
        self.fft_signal.emit(bands, level)

    async def _on_nmap_event(self, event) -> None:
        if isinstance(event.data, dict):
            self.nmap_signal.emit(dict(event.data))

    async def _on_recon_event(self, event) -> None:
        if isinstance(event.data, dict):
            self.recon_signal.emit(dict(event.data))

    async def _on_hud_overlay(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        kind = data.get("kind")
        if kind == "app_glow":
            payload = data.get("windows") or []
            if isinstance(payload, list):
                self.app_glow_signal.emit(list(payload))
        elif kind == "pixel_projection":
            self.pixel_projection_signal.emit(dict(data))

    async def _on_pixel_event(self, event) -> None:
        """Mirror Pixel notifications onto the AR card overlay."""
        data = event.data if isinstance(event.data, dict) else {}
        kind = str(data.get("kind", ""))
        # Only project clipboard / call / quick_command — image OCR is too
        # large for the card and is already opened in konsole.
        if kind not in {"clipboard", "call_incoming", "quick_command"}:
            return
        body = (
            data.get("payload")
            or data.get("caller")
            or data.get("text")
            or ""
        )
        title_map = {
            "clipboard": "Clipboard",
            "call_incoming": "Incoming Call",
            "quick_command": "Quick Command",
        }
        self.pixel_projection_signal.emit({
            "app": title_map.get(kind, "Pixel"),
            "title": title_map.get(kind, "Pixel"),
            "body": str(body)[:200],
        })


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = JarvisHUD()
    hud.show_fullscreen()
    sys.exit(app.exec())
