"""Integration tests for the new ``Jarvis._shadow_then_real`` flow and
the ``_try_resolve_shadow`` confirmation gate.

We bypass Singleton state — each test gets its own ``Jarvis()`` because
the metaclass returns the cached instance. We patch the
``ShadowExec.run`` coroutine to deterministic outcomes so the suite never
touches a real sandbox binary.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shadow_exec
from singleton import Singleton


@pytest.fixture(autouse=True)
def reset_jarvis_singleton():
    """Drop the Jarvis singleton between tests so we get clean state."""
    Singleton.reset()
    yield
    Singleton.reset()


def _make_shadow_result(rc: int, engine: str = "bwrap") -> shadow_exec.ShadowResult:
    return shadow_exec.ShadowResult(rc=rc, stdout="", stderr="boom" if rc else "", engine=engine)


class _NullMemory:
    """ChromaDB stand-in so Jarvis tests never need Ollama running."""
    def remember(self, *a, **kw):  # noqa: D401
        return None

    def recall(self, *a, **kw):
        return []


def _make_jarvis_with_shadow_engine(present: bool = True):
    """Construct a Jarvis without actually running its real subsystems."""
    from core import Jarvis
    j = Jarvis()
    # Force the shadow engine choice for the test.
    j._shadow = shadow_exec.ShadowExec(
        bwrap_path="/usr/bin/true" if present else "",
        podman_path="",
    )
    j.memory = _NullMemory()
    j.say = lambda *a, **kw: None  # mute TTS spawn during tests
    return j


def test_shadow_then_real_queues_on_rc_zero():
    j = _make_jarvis_with_shadow_engine(True)
    with patch.object(j._shadow, "run", new=AsyncMock(return_value=_make_shadow_result(0))):
        out = asyncio.run(j._shadow_then_real("echo hi", "test intent", {}))
    assert out == "queued"
    assert j._pending_shadow is not None
    assert j._pending_shadow["cmd"] == "echo hi"


def test_shadow_then_real_rejects_on_failure():
    j = _make_jarvis_with_shadow_engine(True)
    with patch.object(j._shadow, "run", new=AsyncMock(return_value=_make_shadow_result(2))):
        out = asyncio.run(j._shadow_then_real("rm -rf nope", "intent", {}))
    assert out == "rejected"
    assert j._pending_shadow is None


def test_shadow_then_real_falls_through_when_no_engine():
    j = _make_jarvis_with_shadow_engine(False)
    out = asyncio.run(j._shadow_then_real("any", "intent", {}))
    assert out == "direct"
    assert j._pending_shadow is None


def test_resolve_shadow_handles_negative_response():
    j = _make_jarvis_with_shadow_engine(True)
    import time
    j._pending_shadow = {
        "cmd": "echo hi", "intent": "intent", "snap": None, "ts": time.time(),
        "engine": "bwrap",
    }
    handled = asyncio.run(j._try_resolve_shadow("отмени"))
    assert handled is True
    assert j._pending_shadow is None


def test_resolve_shadow_returns_false_when_no_pending():
    j = _make_jarvis_with_shadow_engine(True)
    handled = asyncio.run(j._try_resolve_shadow("да"))
    assert handled is False
