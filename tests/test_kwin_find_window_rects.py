"""KWinOrchestrator.find_window_rects test (pure-Python, no Plasma needed).

We monkeypatch ``KWinOrchestrator.query_windows`` to return a synthetic
layout and verify the filter returns geometry dicts that match by caption
or by X11 resource name.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kwin import KWinOrchestrator, ScreenLayout, WindowInfo
from singleton import Singleton


def _make_layout() -> ScreenLayout:
    windows = [
        WindowInfo(caption="src/core.py — Visual Studio Code", res="code", x=10, y=10, width=800, height=600),
        WindowInfo(caption="zsh — Konsole", res="org.kde.konsole", x=820, y=10, width=600, height=400),
        WindowInfo(caption="Untitled — Firefox", res="firefox", x=0, y=420, width=1024, height=600),
    ]
    return ScreenLayout(width=1920, height=1080, windows=windows)


def test_find_window_rects_matches_caption() -> None:
    Singleton.reset(KWinOrchestrator)
    kwin = KWinOrchestrator()

    async def _fake_query() -> ScreenLayout:
        return _make_layout()

    kwin.query_windows = _fake_query  # type: ignore[assignment]
    rects = asyncio.run(kwin.find_window_rects("visual studio code"))
    assert len(rects) == 1
    assert rects[0]["caption"].endswith("Visual Studio Code")
    assert rects[0]["w"] == 800


def test_find_window_rects_matches_resource() -> None:
    Singleton.reset(KWinOrchestrator)
    kwin = KWinOrchestrator()

    async def _fake_query() -> ScreenLayout:
        return _make_layout()

    kwin.query_windows = _fake_query  # type: ignore[assignment]
    rects = asyncio.run(kwin.find_window_rects("konsole"))
    assert len(rects) == 1
    assert rects[0]["caption"].endswith("Konsole")


def test_find_window_rects_empty_pattern() -> None:
    Singleton.reset(KWinOrchestrator)
    kwin = KWinOrchestrator()
    rects = asyncio.run(kwin.find_window_rects(""))
    assert rects == []
