"""IPC protocol between Jarvis and the inference server.

Simple JSON-based protocol over a Unix socket (or TCP fallback).
Requests and responses are one JSON object per line.
"""
from __future__ import annotations

from typing import NamedTuple


class GenerateRequest(NamedTuple):
    """Ask the inference server to generate tokens."""
    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0


class GenerateChunk(NamedTuple):
    """A chunk of the generated response (streamed)."""
    token: str
    is_done: bool = False


class HealthCheck(NamedTuple):
    """Ping the server to see if it's alive."""
    pass


class HealthReply(NamedTuple):
    """Server is alive."""
    status: str = "ok"


class ServerError(NamedTuple):
    """Server encountered an error."""
    error: str
    detail: str | None = None
