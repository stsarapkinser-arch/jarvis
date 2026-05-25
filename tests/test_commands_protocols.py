"""Verify the new PROTOCOLS section parses cleanly and matches via the
CommandBook fuzzy lookup the same way the original macros do."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commands import CommandBook


def test_book_loads_battle_and_stealth_macros() -> None:
    book = CommandBook(str(Path(__file__).resolve().parents[1] / "commands.txt"))
    for phrase in ("режим боя", "боевой режим", "battle mode", "режим невидимости", "stealth mode"):
        assert phrase in book.macros, f"missing macro: {phrase}"
        assert "ufw" in book.macros[phrase] or "tor" in book.macros[phrase] or "macchanger" in book.macros[phrase]


def test_book_fuzzy_matches_protocols() -> None:
    book = CommandBook(str(Path(__file__).resolve().parents[1] / "commands.txt"))
    # Close phrasing should still hit through difflib.
    assert book.match("режим боя") is not None
    assert book.match("включи невидимость") is not None
    assert book.match("battle mode") is not None
