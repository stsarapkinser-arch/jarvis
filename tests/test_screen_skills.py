"""Тесты навыков «глаза на экране» (screen perception).

Без реального скриншота/OCR: SkillContext замокан, проверяем сборку команды,
разбор OCR-вывода и человекочитаемую речь."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import skills
from src.inference.router import IntentCategory
from src.skills.screen import clean_ocr, ocr_command


class _FakeCtx:
    def __init__(self, run_result=(0, "", "")):
        self.spawned: list[str] = []
        self.ran: list[str] = []
        self._run_result = run_result

    async def spawn(self, cmd: str) -> int:
        self.spawned.append(cmd)
        return 4242

    async def run(self, cmd: str):
        self.ran.append(cmd)
        return self._run_result


def test_screen_skills_registered_in_ui():
    ui = {s.id for s in skills.skills_for(IntentCategory.UI_CONTROL)}
    assert {"read_screen", "read_active_window", "describe_error_on_screen"} <= ui


def test_ocr_command_uses_tesseract_and_cleanup():
    cmd = ocr_command()
    assert "tesseract" in cmd
    assert "mktemp" in cmd and "rm -f" in cmd
    assert "grim" in cmd  # Wayland-захват в цепочке


def test_clean_ocr_collapses_noise():
    raw = "Error:   file\n\n   not   found\n\n\n"
    assert clean_ocr(raw) == "Error: file not found"


def test_read_screen_speaks_recognised_text():
    ctx = _FakeCtx(run_result=(0, "Привет оператор\nстрока два", ""))
    sk = skills.get("read_screen")
    out = asyncio.run(sk.handler(ctx, {}))
    assert ctx.ran and "tesseract" in ctx.ran[0]
    assert "Привет оператор" in out


def test_read_screen_handles_missing_tools():
    ctx = _FakeCtx(run_result=(127, "", "tesseract: not found"))
    sk = skills.get("read_screen")
    out = asyncio.run(sk.handler(ctx, {}))
    assert "не удалось" in out.lower()


def test_describe_error_extracts_error_lines():
    ctx = _FakeCtx(run_result=(0, "Всё хорошо. Permission denied: /etc/shadow. Конец.", ""))
    sk = skills.get("describe_error_on_screen")
    out = asyncio.run(sk.handler(ctx, {}))
    assert "permission denied" in out.lower()
    assert "хорошо" not in out.lower()  # не-ошибочные фрагменты отброшены


def test_read_active_window_returns_title():
    ctx = _FakeCtx(run_result=(0, "Konsole — bash\n", ""))
    sk = skills.get("read_active_window")
    out = asyncio.run(sk.handler(ctx, {}))
    assert "Konsole" in out
