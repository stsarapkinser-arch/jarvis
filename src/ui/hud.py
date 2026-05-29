"""Aegis HUD — Professional Minimalist, чистый QPainter, ноль OpenGL.

ТЗ оператора (финальное, после провала с GL-стеком на Intel N100):

  • Никаких GL-виджетов, никаких шейдеров, никаких FBO. Они валили N100
    в Core Dump через vk::DeviceLostError. Сейчас всё на QPainter.
  • 1-px неоновая рамка по периметру экрана. Цвет мягко пульсирует синим
    в IDLE; плавно (QPropertyAnimation) переходит в красный при угрозе.
  • WhisperLine внизу по центру: ``[Load: ... | RAM: ...GB | Pixel: ...% | Last Event: ...]``.
  • Элегантная 2D-сфера в углу, мягко мерцает при THINKING. QPainter +
    QRadialGradient — никаких vec3 в шейдере, ничего что могло бы упасть.
  • Click-through через WindowTransparentForInput + WA_TransparentForMouseEvents.
  • Top-most, не в таскбаре, не принимает фокус.

Все cross-thread обновления — через pyqtSignal. Никаких прямых .update()
из bus-треда. Ни одного вызова, который мог бы дёрнуть QPainter из не-Qt
треда.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from collections import deque
from typing import Any

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QApplication, QMainWindow

log = logging.getLogger("jarvis.hud")

HUD_WINDOW_TITLE = "JarvisHUD"

# ───────── Frame palette (цвет состояний рамки/сферы) ─────────
# Мягкая, ненасильственная палитра — без агрессивного красного.
# ALERT заменён на тёплый янтарный — тревога без враждебности.
FRAME_IDLE_BRIGHT   = QColor( 40, 120, 200, 140)   # Приглушённый синий
FRAME_IDLE_DIM      = QColor( 20,  80, 160,  70)   # Очень тёмный синий
FRAME_THINKING      = QColor( 30, 180, 160, 180)   # Мягкий тил
FRAME_SPEAKING      = QColor(160, 210, 255, 220)   # Мягкий голубой
FRAME_ALERT         = QColor(220, 160,  40, 220)   # Тёплый янтарь
FRAME_NEUTRAL       = QColor( 40, 120, 200, 130)   # Нейтральный синий

# Голосовая пульсация рамки: лерп dim→peak по уровню голоса (FFT).
FRAME_SPEAK_DIM     = QColor( 90, 150, 210, 110)   # пауза между словами
FRAME_SPEAK_PEAK    = QColor(190, 225, 255, 235)   # пик голоса

STATE_FRAME_COLORS: dict[str, QColor] = {
    "IDLE":     FRAME_IDLE_BRIGHT,
    "THINKING": FRAME_THINKING,
    "SPEAKING": FRAME_SPEAKING,
    "ALERT":    FRAME_ALERT,
}

# ───────── set_hud_state палитра (нейросеть сама называет цвет) ─────────
# Имена из tool-схемы set_hud_state(color, animation). Неизвестное имя → cyan.
HUD_TOOL_COLORS: dict[str, QColor] = {
    "cyan":   QColor( 30, 180, 200, 200),
    "blue":   FRAME_IDLE_BRIGHT,
    "amber":  FRAME_ALERT,
    "red":    QColor(220,  70,  60, 225),
    "green":  QColor( 60, 200, 120, 215),
    "white":  QColor(200, 220, 240, 210),
}

# ───────── Geometry / timing ─────────
SPHERE_BASE_R       = 52
WHISPER_FONT_SIZE   = 9
WHISPER_BOTTOM_GAP  = 20      # отступ от низа экрана до текста
APP_GLOW_TTL_SEC    = 4.0
FRAME_TRANSITION_MS = 600     # длительность плавной смены цвета рамки
IDLE_PULSE_MS       = 2400    # период idle-пульсации (один цикл вкл/выкл)
WHISPER_REFRESH_SEC = 0.5
DIAGNOSTIC_FADE_SEC = 1.0


class JarvisHUD(QMainWindow):
    """Single fullscreen transparent overlay, отрисовка ТОЛЬКО QPainter'ом."""

    # ──── Cross-thread signals ────
    state_signal          = pyqtSignal(str, str)
    hud_state_signal      = pyqtSignal(str, str)   # set_hud_state: color, animation
    token_signal          = pyqtSignal(str)
    position_signal       = pyqtSignal(float, float)
    fft_signal            = pyqtSignal(list, float)
    app_glow_signal       = pyqtSignal(list)
    system_state_signal   = pyqtSignal(str)

    # ──── Compat signals (другие модули и тесты могут их использовать) ────
    # Все они мапятся в frame-color animation / sphere pulse — никакой
    # отдельной "ауры" в этой минималистичной версии нет.
    aura_color_signal     = pyqtSignal(int, int, int, int, float)  # r,g,b,a,ttl_sec
    aura_pulse_signal     = pyqtSignal()
    aura_intensity_signal = pyqtSignal(float)
    diagnostic_signal     = pyqtSignal(float)                       # длительность overlay
    neural_signal         = pyqtSignal(bool)                        # no-op

    def __init__(self) -> None:
        super().__init__()

        # ─── Атрибуты прозрачности/click-through ставятся ДО show ───
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAutoFillBackground(False)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.BypassWindowManagerHint
        )
        self.setWindowTitle(HUD_WINDOW_TITLE)

        # ─── Сигналы → главный Qt-тред (все Cross-thread издатели делают .emit) ───
        self.state_signal.connect(self.set_state)
        self.hud_state_signal.connect(self._apply_hud_state_cmd)
        self.token_signal.connect(self._append_token)
        self.position_signal.connect(self.set_core_position)
        self.fft_signal.connect(self._on_fft)
        self.app_glow_signal.connect(self._on_app_glow)
        self.system_state_signal.connect(self._on_system_tier)
        self.aura_color_signal.connect(self._apply_aura_color)
        self.aura_pulse_signal.connect(self._apply_aura_pulse)
        self.aura_intensity_signal.connect(self._apply_aura_intensity)
        self.diagnostic_signal.connect(self._apply_diagnostic)
        self.neural_signal.connect(self._apply_neural)
        self._bus: Any = None

        # ─── Geometry ───
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

        # ─── Internal state ───
        self.state: str = "IDLE"
        self.thinking_text: str = ""
        self.ticker_buffer: deque[str] = deque(maxlen=220)

        # Sphere position (strictly locked to native position, no drift)
        self._locked_xr: float = 0.88
        self._locked_yr: float = 0.86

        # Pulse phase (для idle/thinking мягкого мерцания сферы)
        self._pulse_phase: float = 0.0

        # FFT уровень — мерцание сферы при SPEAKING синхронно голосу
        self._fft_level: float = 0.0
        self._fft_decay_ts: float = 0.0

        # App glow brackets — 2D Iron-Man рамки вокруг окон, БЕЗ bloom
        self._app_glows: list[dict] = []

        # Диагностика — углы экрана на N секунд после голосовой команды
        self._diag_until: float = 0.0

        # WhisperLine кэш (обновляется раз в 0.5 сек, не каждый кадр)
        self._whisper_cache: dict[str, Any] = {}
        self._whisper_cache_ts: float = 0.0
        self._last_event_text: str = ""
        self._pixel_battery: int | None = None

        # ─── Frame color + анимация ───
        # Реальное хранилище цвета — pyqtProperty ниже. QPropertyAnimation
        # интерполирует QColor по умолчанию (зарегистрировано Qt'ом).
        self._frame_color: QColor = QColor(FRAME_IDLE_BRIGHT)
        self._transition_anim = QPropertyAnimation(self, b"frameColor", self)
        self._transition_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._transition_anim.setDuration(FRAME_TRANSITION_MS)
        self._idle_pulse_anim = QPropertyAnimation(self, b"frameColor", self)
        self._idle_pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._idle_pulse_anim.setDuration(IDLE_PULSE_MS)
        self._idle_pulse_anim.setLoopCount(-1)
        self._idle_pulse_anim.setStartValue(FRAME_IDLE_BRIGHT)
        self._idle_pulse_anim.setKeyValueAt(0.5, FRAME_IDLE_DIM)
        self._idle_pulse_anim.setEndValue(FRAME_IDLE_BRIGHT)
        # Стартуем pulse сразу — IDLE по дефолту
        self._idle_pulse_anim.start()

        # ─── Tick (30 FPS — достаточно для QPainter, N100 не вспотеет) ───
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    # ──────── frameColor как pyqtProperty (для QPropertyAnimation) ────────
    def _get_frame_color(self) -> QColor:
        return self._frame_color

    def _set_frame_color(self, c: QColor) -> None:
        self._frame_color = c
        self.update()

    frameColor = pyqtProperty(QColor, fget=_get_frame_color, fset=_set_frame_color)

    # ──────── Lifecycle ────────
    def show_fullscreen(self) -> None:
        self.showFullScreen()

    # ──────── Frame animation helpers ────────
    def _animate_frame_to(self, target: QColor, duration_ms: int = FRAME_TRANSITION_MS) -> None:
        """Плавный переход рамки на новый цвет. Глушит idle-пульсацию
        пока transition идёт; вызывающий должен сам перезапустить
        ``_idle_pulse_anim`` если нужно (см. ``_set_state_color``)."""
        self._idle_pulse_anim.stop()
        self._transition_anim.stop()
        self._transition_anim.setDuration(duration_ms)
        self._transition_anim.setStartValue(QColor(self._frame_color))
        self._transition_anim.setEndValue(target)
        self._transition_anim.start()

    def _set_state_color(self) -> None:
        """Анимация рамки в цвет текущего ``self.state``. Если IDLE —
        после transition перезапускаем бесконечную idle-пульсацию.
        При SPEAKING — frame пульсирует в такт голосу через FFT level,
        поэтому QPropertyAnimation глушим: им управляет только _tick→FFT."""
        if self.state == "SPEAKING":
            # Голос полностью владеет цветом рамки — никаких конкурирующих
            # анимаций, иначе они дёргают frameColor мимо FFT и появляется
            # рассинхрон/мерцание. _update_speaking_frame в _tick рулит сам.
            self._idle_pulse_anim.stop()
            self._transition_anim.stop()
            return
        target = STATE_FRAME_COLORS.get(self.state, FRAME_NEUTRAL)
        self._animate_frame_to(target, FRAME_TRANSITION_MS)
        if self.state == "IDLE":
            QTimer.singleShot(FRAME_TRANSITION_MS + 50, self._restart_idle_pulse)

    def _restart_idle_pulse(self) -> None:
        # Запускать только если за это время не пришёл другой state.
        if self.state != "IDLE":
            return
        # Если transition ещё работает — не перебиваем его.
        if self._transition_anim.state() == QPropertyAnimation.State.Running:
            return
        self._idle_pulse_anim.setStartValue(QColor(self._frame_color))
        self._idle_pulse_anim.setKeyValueAt(0.5, FRAME_IDLE_DIM)
        self._idle_pulse_anim.setEndValue(QColor(self._frame_color))
        self._idle_pulse_anim.start()

    def _update_speaking_frame(self) -> None:
        """Рамка дышит в такт голосу. Вызывается каждый _tick (30 fps),
        читает свежий self._fft_level (издаётся PiperFFTPump ~100 Hz прямо
        из PCM-потока перед aplay) — задержка между звуком и пульсом ≈ 0.

        Лерп между приглушённым и ярким голубым по уровню голоса: и цвет,
        и альфа едут вместе, поэтому рамка ощутимо «вспыхивает» на пиках
        речи и притухает в паузах между словами."""
        if self.state != "SPEAKING":
            return
        lvl = self._fft_level
        # Лёгкое фоновое «дыхание», чтобы рамка жила даже на тихих участках.
        breath = 0.10 * (0.5 + 0.5 * math.sin(self._pulse_phase))
        e = max(0.0, min(1.0, lvl + breath))
        # Лерп dim → peak.
        r = int(FRAME_SPEAK_DIM.red()   + e * (FRAME_SPEAK_PEAK.red()   - FRAME_SPEAK_DIM.red()))
        g = int(FRAME_SPEAK_DIM.green() + e * (FRAME_SPEAK_PEAK.green() - FRAME_SPEAK_DIM.green()))
        b = int(FRAME_SPEAK_DIM.blue()  + e * (FRAME_SPEAK_PEAK.blue()  - FRAME_SPEAK_DIM.blue()))
        a = int(FRAME_SPEAK_DIM.alpha() + e * (FRAME_SPEAK_PEAK.alpha() - FRAME_SPEAK_DIM.alpha()))
        self._set_frame_color(QColor(r, g, b, a))

    # ──────── Qt slots (main thread) ────────
    @pyqtSlot(str, str)
    def set_state(self, new_state: str, thought: str = "") -> None:
        self.state = new_state
        self.thinking_text = thought
        self._set_state_color()
        # WhisperLine трекает последнее событие
        if new_state == "ALERT" and thought:
            self._last_event_text = thought[:60]
        self.update()

    @pyqtSlot(str, str)
    def _apply_hud_state_cmd(self, color: str, animation: str) -> None:
        """set_hud_state(color, animation) — нейросеть прямо рулит визором.

        Мапим в существующую state-машину рамки: glitch→тревога, idle→покой,
        pulse→активная работа. Явный цвет из tool'а перекрывает палитру состояния.
        Речь (SPEAKING, FFT-пульс) приоритетнее и перехватит рамку, когда
        зазвучит голос."""
        qc = HUD_TOOL_COLORS.get(color.strip().lower(), HUD_TOOL_COLORS["cyan"])
        anim = animation.strip().lower()
        if anim == "glitch":
            self.state = "ALERT"
            self._animate_frame_to(qc, FRAME_TRANSITION_MS)
        elif anim == "idle":
            self.state = "IDLE"
            self._animate_frame_to(qc, FRAME_TRANSITION_MS)
            QTimer.singleShot(FRAME_TRANSITION_MS + 50, self._restart_idle_pulse)
        else:  # "pulse" / default — активная работа
            self.state = "THINKING"
            self._animate_frame_to(qc, FRAME_TRANSITION_MS)
        self.update()

    @pyqtSlot(str)
    def _append_token(self, tok: str) -> None:
        for ch in tok:
            self.ticker_buffer.append(ch)

    @pyqtSlot(float, float)
    def set_core_position(self, xr: float, yr: float) -> None:
        # HUD строго зафиксирован на родном месте (правый нижний угол).
        # Любые запросы на репозицию игнорируем — пятно/рамка НИКОГДА не
        # съезжают. Аргументы намеренно не используются.
        del xr, yr

    @pyqtSlot(list, float)
    def _on_fft(self, bands: list, level: float) -> None:
        try:
            self._fft_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            return
        self._fft_decay_ts = time.time()

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
    def _on_system_tier(self, tier: str) -> None:
        # Tier-transition не меняет frame color (это делает STATE_CHANGE) —
        # мы держим WhisperLine актуальной, и всё.
        pass

    @pyqtSlot(int, int, int, int, float)
    def _apply_aura_color(self, r: int, g: int, b: int, a: int, ttl_sec: float) -> None:
        """Compat-сигнал «aura» теперь анимирует рамку. После TTL возврат
        к state-цвету (ALERT/IDLE/etc)."""
        target = QColor(int(r), int(g), int(b), int(a))
        self._animate_frame_to(target, FRAME_TRANSITION_MS)
        if ttl_sec > 0:
            QTimer.singleShot(int(ttl_sec * 1000), self._set_state_color)

    @pyqtSlot()
    def _apply_aura_pulse(self) -> None:
        # Микро-пульс ауры больше не отдельная сущность. Сфера сама
        # реагирует на FFT level — игнорируем для упрощения.
        pass

    @pyqtSlot(float)
    def _apply_aura_intensity(self, value: float) -> None:
        # Meeting auto-dim: при значении < 1.0 чуть уменьшаем alpha рамки.
        try:
            v = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        c = self._frame_color
        self._set_frame_color(QColor(c.red(), c.green(), c.blue(), int(c.alpha() * v)))

    @pyqtSlot(float)
    def _apply_diagnostic(self, duration_sec: float) -> None:
        self._diag_until = time.time() + max(0.5, float(duration_sec))

    @pyqtSlot(bool)
    def _apply_neural(self, active: bool) -> None:
        # Без GL-сферы neural-pulse не имеет смысла. Игнорируем.
        del active

    # ──────── Tick (30 fps) ────────
    def _tick(self) -> None:
        self._pulse_phase = (self._pulse_phase + 0.06) % math.tau
        # Decay FFT
        if time.time() - self._fft_decay_ts > 0.1:
            self._fft_level *= 0.88
        # Update frame pulsation when speaking (zero-delay sync with audio FFT)
        self._update_speaking_frame()
        # Cull stale app glows
        now = time.time()
        before = len(self._app_glows)
        self._app_glows = [g for g in self._app_glows
                           if now - g.get("ts", now) < APP_GLOW_TTL_SEC]
        # Repaint только если есть что показывать движущегося. paintEvent
        # дешёвый, можно дёргать каждый кадр без вреда даже на N100.
        self.update()
        del before  # silence linter; кэш на будущее если оптимизировать

    # ──────── Paint ────────
    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        self._draw_frame(painter, w, h)
        self._draw_app_brackets(painter)
        self._draw_sphere(painter, w, h)
        self._draw_whisper_line(painter, w, h)
        if time.time() < self._diag_until:
            self._draw_diagnostic(painter, w, h)

        painter.end()

    # --- 1-px перiметральная рамка ---
    def _draw_frame(self, painter: QPainter, w: int, h: int) -> None:
        pen = QPen(self._frame_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # inset 1 px — пиксель полностью внутри widget.rect()
        painter.drawRect(0, 0, w - 1, h - 1)

    # --- 2D corner brackets вокруг подсвеченных окон ---
    def _draw_app_brackets(self, painter: QPainter) -> None:
        now = time.time()
        for g in self._app_glows:
            if g["w"] <= 0 or g["h"] <= 0:
                continue
            age = now - g.get("ts", now)
            if age >= APP_GLOW_TTL_SEC:
                continue
            fade = max(0.0, 1.0 - age / APP_GLOW_TTL_SEC)
            color_name = g.get("color", "cyan")
            base = {
                "cyan":   QColor(  0, 220, 255),
                "amber":  QColor(255, 200,  60),
                "red":    QColor(255,  60,  80),
                "green":  QColor( 80, 255, 160),
                "violet": QColor(180,  80, 255),
            }.get(color_name, QColor(0, 220, 255))
            ring = QColor(base.red(), base.green(), base.blue(), int(230 * fade))
            pen = QPen(ring)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            x, y = int(g["x"]), int(g["y"])
            ww, hh = int(g["w"]), int(g["h"])
            L = 22
            for cx, cy, dx, dy in (
                (x,       y,        +1, +1),
                (x + ww,  y,        -1, +1),
                (x,       y + hh,   +1, -1),
                (x + ww,  y + hh,   -1, -1),
            ):
                painter.drawLine(cx, cy, cx + dx * L, cy)
                painter.drawLine(cx, cy, cx, cy + dy * L)
            # Caption над bracket'ом
            painter.setFont(QFont("Monospace", 9))
            painter.drawText(x + 4, max(0, y - 6), g.get("caption", "")[:40])

    # --- 2D Сфера в углу (индикатор речи) ---
    def _draw_sphere(self, painter: QPainter, w: int, h: int) -> None:
        # Сфера появляется ТОЛЬКО когда Jarvis активно говорит (FFT level > threshold).
        # При отсутствии аудио — полностью невидима.
        if self.state != "SPEAKING" or self._fft_level < 0.05:
            return

        cx = int(self._locked_xr * w)
        cy = int(self._locked_yr * h)

        # Радиус пульсирует исключительно по FFT уровню голоса.
        # Мягкое фоновое дыхание (soft) только как минимальная анимация.
        soft = 0.5 + 0.5 * math.sin(self._pulse_phase)
        r = int(SPHERE_BASE_R + 2 * soft + 22 * self._fft_level)

        c = self._frame_color
        # Видимость сферы пропорциональна уровню FFT — полностью зависит от звука.
        # Появляется ТОЛЬКО когда есть активная речь.
        visibility = self._fft_level

        # Outer halo
        outer = QRadialGradient(cx, cy, r * 2.2)
        glow_alpha = int(50 * visibility + 80 * self._fft_level)
        outer.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), glow_alpha))
        outer.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer))
        painter.drawEllipse(QPoint(cx, cy), r * 2, r * 2)

        # Ring
        ring_alpha = int(160 + 70 * self._fft_level)
        ring_pen = QPen(QColor(c.red(), c.green(), c.blue(), ring_alpha))
        ring_pen.setWidth(2)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPoint(cx, cy), r, r)

        # Inner core
        inner = QRadialGradient(cx, cy, max(1, int(r * 0.65)))
        inner_alpha = int(180 + 50 * self._fft_level)
        inner.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), inner_alpha))
        inner.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(inner))
        painter.drawEllipse(QPoint(cx, cy), max(1, int(r * 0.65)), max(1, int(r * 0.65)))

    # --- WhisperLine — статус по центру внизу ---
    def _draw_whisper_line(self, painter: QPainter, w: int, h: int) -> None:
        now = time.time()
        if now - self._whisper_cache_ts > WHISPER_REFRESH_SEC:
            self._refresh_whisper_cache()
            self._whisper_cache_ts = now

        load_str = self._whisper_cache.get("load", "?")
        ram_gb = self._whisper_cache.get("ram_gb", 0.0)
        pixel = self._pixel_battery
        pixel_part = f"Pixel: {pixel}%" if pixel is not None else "Pixel: —"
        last = self._last_event_text or "—"
        text = (
            f"Load: {load_str}   |   "
            f"RAM: {ram_gb:.1f}GB   |   "
            f"{pixel_part}   |   "
            f"Last Event: {last[:40]}"
        )

        painter.setFont(QFont("Monospace", WHISPER_FONT_SIZE))
        text_w = painter.fontMetrics().horizontalAdvance(text)
        x = max(20, (w - text_w) // 2)
        y = h - WHISPER_BOTTOM_GAP

        c = self._frame_color
        ink = QColor(c.red(), c.green(), c.blue(), 160)
        painter.setPen(ink)
        painter.drawText(int(x), int(y), text)

    def _refresh_whisper_cache(self) -> None:
        out: dict[str, Any] = {}
        try:
            from src.common.event_bus import SystemState
            snap = SystemState().snapshot()
            out["load"] = str(getattr(snap.load, "value", snap.load))
            out["cpu"] = snap.cpu
            out["ram_pct"] = snap.ram
            out["gpu"] = snap.gpu
            out["thermal"] = snap.thermal
        except Exception:
            pass
        try:
            import psutil
            vm = psutil.virtual_memory()
            out["ram_gb"] = (vm.total - vm.available) / (1024 ** 3)
        except Exception:
            out.setdefault("ram_gb", 0.0)
        self._whisper_cache = out

    # --- Диагностический overlay (4 угла) ---
    def _draw_diagnostic(self, painter: QPainter, w: int, h: int) -> None:
        try:
            from src.common.event_bus import SystemState
            snap = SystemState().snapshot()
        except Exception:
            return
        # Fade-out за последнюю секунду
        remaining = self._diag_until - time.time()
        fade = max(0.0, min(1.0, remaining / DIAGNOSTIC_FADE_SEC)) if remaining < DIAGNOSTIC_FADE_SEC else 1.0

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else self.rect()
        try:
            ax, ay = avail.x(), avail.y()
            aw, ah = avail.width(), avail.height()
        except AttributeError:
            ax, ay, aw, ah = 0, 0, w, h

        c = self._frame_color
        ink = QColor(c.red(), c.green(), c.blue(), int(230 * fade))
        painter.setPen(ink)
        painter.setFont(QFont("Monospace", 11))
        margin = 30
        labels = (
            (f"CPU  {snap.cpu:.0f}%",      ax + margin,              ay + margin + 14),
            (f"RAM  {snap.ram:.0f}%",      ax + aw - margin - 110,   ay + margin + 14),
            (f"iGPU {snap.gpu:.0f}%",      ax + margin,              ay + ah - margin),
            (f"TEMP {snap.thermal:.0f}°C", ax + aw - margin - 110,   ay + ah - margin),
        )
        for text, x, y in labels:
            painter.drawText(int(x), int(y), text)

    # ──────── Bus wiring ────────
    def subscribe_to_bus(self, bus: Any) -> None:
        from src.common.event_bus import EventType
        self._bus = bus
        bus.subscribe(EventType.STATE_CHANGE,     self._on_state)
        bus.subscribe(EventType.HUD_STATE,        self._on_hud_state)
        bus.subscribe(EventType.TOKEN_STREAM,     self._on_token)
        bus.subscribe(EventType.KWIN_ACTION,      self._on_kwin_action)
        bus.subscribe(EventType.AUDIO_FFT,        self._on_audio_fft)
        bus.subscribe(EventType.HUD_OVERLAY,      self._on_hud_overlay)
        bus.subscribe(EventType.SYSTEM_STATE,     self._on_system_state)
        bus.subscribe(EventType.CALL_INBOUND,     self._on_call_inbound)
        bus.subscribe(EventType.SYSTEM_WAKE,      self._on_system_wake)
        bus.subscribe(EventType.DIAGNOSTIC_START, self._on_diagnostic_start)
        bus.subscribe(EventType.RECON_ALERT,      self._on_recon_alert)
        bus.subscribe(EventType.OS_EVENT,         self._on_os_event)
        bus.subscribe(EventType.PIXEL_EVENT,      self._on_pixel_event)

    async def _on_state(self, event) -> None:
        data = event.data
        if isinstance(data, tuple) and len(data) == 2:
            state, thought = data
        else:
            state, thought = str(data), ""
        self.state_signal.emit(str(state), str(thought))

    async def _on_hud_state(self, event) -> None:
        """set_hud_state tool → визор. Нейросеть сама называет color+animation."""
        data = event.data if isinstance(event.data, dict) else {}
        color = str(data.get("color", "cyan"))
        animation = str(data.get("animation", "pulse"))
        self.hud_state_signal.emit(color, animation)

    async def _on_token(self, event) -> None:
        self.token_signal.emit(str(event.data))

    async def _on_kwin_action(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        kind = data.get("kind")
        if kind == "hud_reposition":
            self.position_signal.emit(
                float(data.get("x_ratio", 0.88)),
                float(data.get("y_ratio", 0.86)),
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

    async def _on_hud_overlay(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        kind = data.get("kind")
        if kind == "app_glow":
            payload = data.get("windows") or []
            if isinstance(payload, list):
                self.app_glow_signal.emit(list(payload))
        elif kind == "safe_pulse":
            # Shadow Exec verified command: активная симуляция — зелёный (persistent).
            # Деактивация: следующий STATE_CHANGE восстановит правильный цвет.
            if data.get("active", True):
                self.aura_color_signal.emit(0, 255, 120, 220, 0.0)
        elif kind == "aura_intensity":
            try:
                value = float(data.get("value", 1.0))
            except (TypeError, ValueError):
                value = 1.0
            self.aura_intensity_signal.emit(value)

    async def _on_system_state(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        tier = str(data.get("load", "normal"))
        self.system_state_signal.emit(tier)

    async def _on_call_inbound(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        caller = str(data.get("caller", "Unknown"))[:40]
        self.aura_color_signal.emit(255, 50, 80, 230, 30.0)
        self._last_event_text = f"Call: {caller}"

    async def _on_system_wake(self, event) -> None:
        del event
        self.aura_color_signal.emit(100, 200, 255, 230, 2.0)
        self._last_event_text = "Wake"

    async def _on_diagnostic_start(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        try:
            duration = float(data.get("duration_sec", 10.0))
        except (TypeError, ValueError):
            duration = 10.0
        self.diagnostic_signal.emit(duration)

    async def _on_recon_alert(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        color = data.get("color", "yellow")
        if color == "red":
            self.aura_color_signal.emit(255, 30, 60, 240, 5.0)
        else:
            self.aura_color_signal.emit(255, 200, 50, 200, 5.0)
        summary = str(data.get("summary", ""))[:40]
        if summary:
            self._last_event_text = f"Recon: {summary}"

    async def _on_os_event(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        sensor = str(data.get("sensor", "?"))
        value = data.get("value", "?")
        self._last_event_text = f"{sensor}={value}"

    async def _on_pixel_event(self, event) -> None:
        data = event.data if isinstance(event.data, dict) else {}
        kind = str(data.get("kind", ""))
        if kind in ("battery", "battery_low"):
            try:
                self._pixel_battery = int(data.get("charge", 0))
            except (TypeError, ValueError):
                pass
            if kind == "battery_low":
                self._last_event_text = f"Pixel low: {self._pixel_battery}%"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("jarvis-hud")
    app.setApplicationDisplayName(HUD_WINDOW_TITLE)
    hud = JarvisHUD()
    hud.show_fullscreen()
    sys.exit(app.exec())
