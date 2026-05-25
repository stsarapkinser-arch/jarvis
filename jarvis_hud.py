from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Any

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient
)
from PyQt6.QtWidgets import QApplication, QMainWindow

STATE_COLORS: dict[str, QColor] = {
    "IDLE": QColor(0, 180, 255, 100),
    "THINKING": QColor(0, 255, 150, 200),
    "SPEAKING": QColor(255, 255, 255, 180),
    "ALERT": QColor(255, 20, 50, 255),
}
TICKER_MAX_CHARS = 220
TICKER_HEIGHT = 28
FADE_IN_MS = 900
CORE_LERP = 0.08  # higher = faster glide toward target


class JarvisHUD(QMainWindow):
    state_signal = pyqtSignal(str, str)
    token_signal = pyqtSignal(str)
    position_signal = pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.state_signal.connect(self.set_state)
        self.token_signal.connect(self._append_token)
        self.position_signal.connect(self.set_core_position)
        self._bus: Any = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.0)

        screen_geo = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geo)

        self.state: str = "IDLE"
        self.pulse_radius: float = 50.0
        self.rotation_angle: int = 0
        self.thinking_text: str = ""
        self.ticker_buffer: deque[str] = deque(maxlen=TICKER_MAX_CHARS)
        self.token_offset: float = 0.0

        # Core anchor (ratios 0..1). Live position lerps toward target every frame
        # so re-positioning is a smooth glide instead of a jump cut.
        self._target_xr: float = 0.85
        self._target_yr: float = 0.85
        self._current_xr: float = 0.85
        self._current_yr: float = 0.85

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(FADE_IN_MS)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    @pyqtSlot(str, str)
    def set_state(self, new_state: str, thought: str = "") -> None:
        self.state = new_state
        self.thinking_text = thought
        self.update()

    @pyqtSlot(str)
    def _append_token(self, tok: str) -> None:
        for ch in tok:
            self.ticker_buffer.append(ch)

    @pyqtSlot(float, float)
    def set_core_position(self, x_ratio: float, y_ratio: float) -> None:
        """Slot for new HUD core anchor (ratios 0..1 of fullscreen rect).
        The actual position eases toward this target via lerp in update_animation()."""
        self._target_xr = max(0.05, min(0.95, x_ratio))
        self._target_yr = max(0.05, min(0.95, y_ratio))

    def update_animation(self) -> None:
        self.rotation_angle = (self.rotation_angle + 3) % 360
        t = time.time()
        if self.state == "IDLE":
            target = 50 + 5 * math.sin(t * 3)
        elif self.state == "THINKING":
            target = 70 + 20 * math.sin(t * 10)
        elif self.state == "SPEAKING":
            target = 60 + 12 * math.sin(t * 6)
        else:
            target = 100 + 40 * math.sin(t * 20)
        self.pulse_radius = target
        self.token_offset = (self.token_offset + 1.5) % 10000.0

        self._current_xr += (self._target_xr - self._current_xr) * CORE_LERP
        self._current_yr += (self._target_yr - self._current_yr) * CORE_LERP

        self.update()

    def _core_center(self, w: int, h: int) -> tuple[int, int]:
        return int(w * self._current_xr), int(h * self._current_yr)

    def _draw_core(self, painter: QPainter, w: int, h: int, color: QColor) -> None:
        cx, cy = self._core_center(w, h)

        outer_glow = QRadialGradient(cx, cy, self.pulse_radius * 3.0)
        outer_glow.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 90))
        outer_glow.setColorAt(0.5, QColor(color.red(), color.green(), color.blue(), 30))
        outer_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(outer_glow)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(cx, cy), int(self.pulse_radius * 3.0), int(self.pulse_radius * 3.0))

        glow = QRadialGradient(cx, cy, self.pulse_radius * 1.5)
        glow.setColorAt(0, color)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(glow)
        painter.drawEllipse(QPoint(cx, cy), int(self.pulse_radius * 2), int(self.pulse_radius * 2))

        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(cx - 60, cy - 60, 120, 120, self.rotation_angle * 16, 200 * 16)
        painter.drawArc(cx - 80, cy - 80, 160, 160, int(-self.rotation_angle * 16 * 1.5), 120 * 16)
        painter.drawArc(cx - 100, cy - 100, 200, 200, int(self.rotation_angle * 16 * 0.6), 60 * 16)

        painter.setPen(color)
        painter.setFont(QFont("Monospace", 10))
        if self.thinking_text:
            text_left = self._current_xr > 0.5
            tx = cx - 320 if text_left else cx + 20
            painter.drawText(
                tx, cy - 130, 320, 110,
                (Qt.AlignmentFlag.AlignRight if text_left else Qt.AlignmentFlag.AlignLeft)
                | Qt.AlignmentFlag.AlignBottom,
                self.thinking_text,
            )

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

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        color = STATE_COLORS.get(self.state, STATE_COLORS["IDLE"])

        self._draw_core(painter, w, h, color)
        self._draw_ticker(painter, w, h, color)

        painter.setPen(color)
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(20, 40, f"JARVIS INTEL N100 :: PROTOCOL_{self.state} :: MEM_SYNC_OK")

        if self.state == "ALERT":
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0, QColor(255, 0, 0, 50))
            grad.setColorAt(0.1, QColor(0, 0, 0, 0))
            grad.setColorAt(0.9, QColor(0, 0, 0, 0))
            grad.setColorAt(1, QColor(255, 0, 0, 50))
            painter.fillRect(self.rect(), grad)

    def subscribe_to_bus(self, bus: Any) -> None:
        from event_bus import EventType
        self._bus = bus
        bus.subscribe(EventType.STATE_CHANGE, self._on_state)
        bus.subscribe(EventType.TOKEN_STREAM, self._on_token)
        bus.subscribe(EventType.KWIN_ACTION, self._on_kwin_action)

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
        """Reposition the HUD core when the workspace layout changes."""
        data = event.data if isinstance(event.data, dict) else {}
        if data.get("kind") == "hud_reposition":
            x = float(data.get("x_ratio", 0.85))
            y = float(data.get("y_ratio", 0.85))
            self.position_signal.emit(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = JarvisHUD()
    hud.show()
    sys.exit(app.exec())
