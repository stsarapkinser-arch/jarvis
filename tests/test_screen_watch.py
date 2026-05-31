"""Тесты проактивного взгляда на экран (OCR-глаза + воля).

Чистые хелперы детекции (dev-контекст по заголовку, фрагменты ошибок в OCR,
сборка предложения помощи) + env-тумблер приватности. Сам цикл — тонкая IO-
обёртка над этими хелперами и уже протестированным ProactiveGate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.skills import screen


# ───────────────────────── dev-контекст (пред-фильтр OCR) ─────────────────────────
def test_dev_context_matches_terminals_and_editors():
    for title in ("Konsole — bash", "main.py — VSCodium", "nvim user@host",
                  "pytest — Yakuake", "gdb session"):
        assert screen.looks_like_dev_context(title), title


def test_dev_context_rejects_casual_windows():
    for title in ("YouTube — Chrome", "Telegram", "Spotify", "Документ — LibreOffice"):
        assert not screen.looks_like_dev_context(title), title


# ───────────────────────── извлечение ошибок из OCR ─────────────────────────
def test_extract_error_fragments_finds_traceback():
    ocr = (
        "user@host:~$ python app.py\n"
        "Traceback (most recent call last):\n"
        "  File app.py line 10\n"
        "ValueError: invalid literal for int.\n"
        "user@host:~$"
    )
    frags = screen.extract_error_fragments(ocr)
    assert frags
    assert any("Traceback" in f or "ValueError" in f for f in frags)


def test_extract_error_fragments_russian():
    frags = screen.extract_error_fragments("Сборка завершена. Ошибка: файл не найден.")
    assert frags and any("шибк" in f.lower() or "не найден" in f.lower() for f in frags)


def test_extract_error_fragments_none_on_clean_text():
    assert screen.extract_error_fragments("Всё хорошо, сборка успешна, тестов 42.") == []


def test_build_error_offer_includes_fragment_and_question():
    offer = screen.build_error_offer(["ValueError: bad int"])
    assert "ValueError: bad int" in offer
    assert "Подсказать" in offer


def test_end_to_end_offer_on_dev_error_screen():
    title = "app.py — Konsole"
    ocr = "Traceback (most recent call last): RuntimeError: boom."
    assert screen.looks_like_dev_context(title)
    frags = screen.extract_error_fragments(ocr)
    assert frags
    assert "RuntimeError" in screen.build_error_offer(frags)


# ───────────────────────── env-тумблер приватности ─────────────────────────
def test_screen_watch_enabled_default_on(monkeypatch):
    from src.core.orchestrator import Jarvis
    monkeypatch.delenv("JARVIS_SCREEN_WATCH", raising=False)
    assert Jarvis._screen_watch_enabled() is True


def test_screen_watch_disabled_by_env(monkeypatch):
    from src.core.orchestrator import Jarvis
    for val in ("0", "false", "off", "no"):
        monkeypatch.setenv("JARVIS_SCREEN_WATCH", val)
        assert Jarvis._screen_watch_enabled() is False
