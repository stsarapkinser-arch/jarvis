"""Offscreen Qt smoke tests for the Professional Minimalist HUD.

После отката OpenGL (ТЗ оператора, итерация «не плодить Core Dump'ы»):
HUD — чистый QPainter. Никаких QOpenGLWidget, шейдеров, FBO. Тесты
проверяют, что:
 * HUD инстанцируется в headless без падений
 * Все Qt-флаги выставлены (Tool / WindowDoesNotAcceptFocus / WindowTransparentForInput)
 * pyqtSignal'ы корректно эмитятся (state/aura_color/diagnostic и т.д.)
 * Frame color меняется при смене state
 * `frameColor` pyqtProperty доступен (нужен для QPropertyAnimation)
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
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

import src.ui.hud

@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(["-platform", "offscreen"])
    yield a


def test_hud_instantiates(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    flags = hud.windowFlags()
    # Все ключевые window flags из ТЗ должны быть установлены.
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.Tool
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowType.WindowTransparentForInput
    # Прозрачность и click-through на уровне атрибутов виджета.
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    # Caption — KWin pin-скрипт ищет окно по нему.
    assert hud.windowTitle() == jarvis_hud.HUD_WINDOW_TITLE
    hud.deleteLater()


def test_signals_present(app) -> None:
    """Проверка что все cross-thread сигналы определены — модули
    (start_jarvis, sentinel, core) эмитят в них."""
    hud = jarvis_hud.JarvisHUD()
    for sig in (
        "state_signal", "token_signal", "position_signal",
        "fft_signal", "app_glow_signal", "system_state_signal",
        "aura_color_signal", "aura_pulse_signal",
        "aura_intensity_signal", "diagnostic_signal", "neural_signal",
    ):
        assert hasattr(hud, sig), f"missing signal {sig}"
    hud.deleteLater()


def test_signals_do_not_raise(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    hud.resize(800, 600)
    hud.fft_signal.emit([0.1, 0.2, 0.3] * 8, 0.42)
    hud.app_glow_signal.emit([{"x": 10, "y": 10, "w": 200, "h": 100, "caption": "Code"}])
    hud.state_signal.emit("THINKING", "running diagnostics")
    hud.token_signal.emit("hello world ")
    hud.position_signal.emit(0.2, 0.8)
    hud.system_state_signal.emit("normal")
    hud.aura_color_signal.emit(255, 50, 80, 200, 5.0)
    hud.aura_pulse_signal.emit()
    hud.aura_intensity_signal.emit(0.35)
    hud.diagnostic_signal.emit(10.0)
    hud.neural_signal.emit(True)
    app.processEvents()
    for _ in range(3):
        hud._tick()
    assert abs(hud._fft_level - 0.42) < 1e-6
    assert hud._app_glows and hud._app_glows[0]["caption"] == "Code"
    assert hud.state == "THINKING"
    assert hud.thinking_text == "running diagnostics"
    assert "".join(hud.ticker_buffer).endswith("hello world ")
    hud.deleteLater()


def test_frame_color_changes_on_state(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    # IDLE → THINKING должен анимировать рамку в teal-ish.
    hud.set_state("THINKING", "")
    app.processEvents()
    # Скакать animation мы не можем без таймера, но finalValue в endValue
    # сэта — проверим что transition_anim получил правильный endValue.
    end = hud._transition_anim.endValue()
    assert isinstance(end, QColor)
    assert end.green() > end.red(), "THINKING должен быть teal-ish (green > red)"
    # ALERT → ярко-красный
    hud.set_state("ALERT", "intrusion")
    app.processEvents()
    end = hud._transition_anim.endValue()
    assert end.red() > end.green(), "ALERT должен быть red (red > green)"
    assert end.red() > 200
    hud.deleteLater()


def test_frame_color_is_pyqt_property(app) -> None:
    """frameColor должен быть зарегистрирован как pyqtProperty —
    без этого QPropertyAnimation не интерполирует QColor."""
    hud = jarvis_hud.JarvisHUD()
    # Доступен через get/set
    c0 = hud.frameColor
    assert isinstance(c0, QColor)
    test_color = QColor(123, 45, 67, 200)
    hud.frameColor = test_color
    assert hud.frameColor.red() == 123
    hud.deleteLater()


def test_no_gl_imports(app) -> None:
    """Регрессия: HUD НЕ должен импортировать GL-виджеты / шейдеры.
    Это уронило N100 в Core Dump в прошлой итерации. Проверяем только
    реальные import-statement'ы — упоминания в docstring разрешены."""
    import re

    import jarvis_hud as jh
    src = Path(jh.__file__).read_text(encoding="utf-8")
    # Грубо рубим всё кроме исполняемого кода: убираем строковые литералы.
    code_only = re.sub(r'"""(.|\n)*?"""', '', src)
    code_only = re.sub(r"'''(.|\n)*?'''", '', code_only)
    forbidden = (
        "from PyQt6.QtOpenGLWidgets",
        "from PyQt6.QtOpenGL",
        "from core_sphere_gl",
        "from bloom_overlay_gl",
        "from gl_shaders",
        "import QOpenGLWidget",
    )
    for token in forbidden:
        assert token not in code_only, f"GL import leaked back into jarvis_hud.py: {token}"


def test_diagnostic_overlay_armed(app) -> None:
    hud = jarvis_hud.JarvisHUD()
    hud.diagnostic_signal.emit(5.0)
    app.processEvents()
    import time
    assert hud._diag_until > time.time()
    hud.deleteLater()


def test_aura_color_compat_animates_frame(app) -> None:
    """aura_color_signal должен анимировать рамку (frameColor) —
    без отдельной OpenGL-ауры это единственный канал «aura»."""
    hud = jarvis_hud.JarvisHUD()
    hud.aura_color_signal.emit(255, 50, 80, 200, 8.0)
    app.processEvents()
    end = hud._transition_anim.endValue()
    assert end.red() == 255 and end.green() == 50 and end.blue() == 80
    hud.deleteLater()
