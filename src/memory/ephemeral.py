"""Ephemeral Programming — self-writing AI scripts.

When a one-liner of bash can't solve the user's intent, the LLM can emit
a one-shot Python script tagged ``<python>...</python>`` (and optionally
``<args>...</args>``). This module:

1. Extracts the script from the LLM response.
2. Writes it to ``/tmp/akali_ephemeral_<rand>.py`` with mode ``0o700``.
3. Runs it through :class:`shadow_exec.ShadowExec` (network-off sandbox).
4. Parses the **last** non-empty stdout line as JSON; if the JSON has a
   ``"thought"`` field we hand it back as the assistant's
   ``<thought>...</thought>``.
5. Guarantees the script file is ``os.unlink``-ed in a ``finally`` block
   even if anything raises mid-flight.

The contract with the LLM lives in :data:`EPHEMERAL_CONTRACT` — append it
to the system prompt before asking the model to solve ad-hoc problems.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import secrets
import tempfile
from pathlib import Path

from src.security.execution import ShadowExec, ShadowResult

log = logging.getLogger("jarvis.ephemeral")

EPHEMERAL_CONTRACT = """\
EPHEMERAL PROGRAMMING:
When a bash one-liner can't express the user's intent, you MAY emit a
one-shot Python program inside <python>...</python>. The program runs
in a network-isolated sandbox (no /home, no network). At the end of
execution its LAST stdout line MUST be a single JSON object. Use keys:
  * "thought": one-sentence rationale (Russian, optional)
  * "result":  the data you want to report
  * "speak":   optional one-line phrase to say aloud
Do NOT print prose before the JSON; comments + debug prints are fine but
the final line must be valid JSON. Keep it under 80 lines.
"""

PY_BLOCK_RE = re.compile(r"<python>\s*(?P<body>.+?)\s*</python>", re.DOTALL | re.IGNORECASE)
ARGS_BLOCK_RE = re.compile(r"<args>\s*(?P<body>.+?)\s*</args>", re.DOTALL | re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class EphemeralRun:
    rc: int
    stdout: str
    stderr: str
    engine: str
    parsed: dict | None    # JSON-parsed last-line, or None on failure
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and self.parsed is not None


def extract_python(text: str) -> tuple[str, list[str]] | None:
    """Return ``(body, argv)`` if the LLM response contains a <python> tag,
    otherwise ``None``. Args are split by whitespace from <args> if
    present; an empty list otherwise."""
    if not text:
        return None
    m = PY_BLOCK_RE.search(text)
    if not m:
        return None
    body = m.group("body").strip()
    if not body:
        return None
    args_m = ARGS_BLOCK_RE.search(text)
    argv = args_m.group("body").strip().split() if args_m else []
    return body, argv


def parse_last_json_line(stdout: str) -> dict | None:
    """Pull the trailing JSON object from ``stdout`` if there is one."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                return None
            return None
        # First non-empty trailing line wasn't JSON → bail out (so we
        # don't accidentally parse a debug print way up the buffer).
        return None
    return None


class EphemeralRunner:
    """Glue between the LLM, the temp file, and Shadow Exec."""

    def __init__(self, shadow: ShadowExec | None = None, tmp_dir: str | os.PathLike = "/tmp") -> None:
        self.shadow = shadow or ShadowExec()
        self.tmp_dir = Path(tmp_dir)

    async def run_from_llm(self, llm_text: str) -> EphemeralRun | None:
        """High-level entry: pull <python> out of an LLM response, execute,
        parse. Returns ``None`` if the response carried no <python> block."""
        extracted = extract_python(llm_text)
        if not extracted:
            return None
        body, argv = extracted
        return await self.run_script(body, argv)

    async def run_script(self, body: str, argv: list[str] | None = None) -> EphemeralRun:
        path = self._write_temp(body)
        try:
            cmd = self._build_cmd(path, argv or [])
            shadow_res: ShadowResult = await self.shadow.run(cmd)
            parsed = parse_last_json_line(shadow_res.stdout)
            err = None
            if shadow_res.rc != 0 and parsed is None:
                err = (shadow_res.stderr.strip().splitlines() or ["unknown error"])[-1][:280]
            return EphemeralRun(
                rc=shadow_res.rc,
                stdout=shadow_res.stdout,
                stderr=shadow_res.stderr,
                engine=shadow_res.engine,
                parsed=parsed,
                error=err,
                timed_out=shadow_res.timed_out,
            )
        finally:
            # CRITICAL — never leave a script on disk, even if the runner
            # raised mid-flight. This is the whole point of "ephemeral".
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except Exception:
                log.exception("ephemeral unlink failed for %s", path)

    def _write_temp(self, body: str) -> str:
        suffix = secrets.token_hex(4)
        fd, path = tempfile.mkstemp(
            prefix="akali_ephemeral_",
            suffix=f"_{suffix}.py",
            dir=str(self.tmp_dir),
        )
        try:
            os.write(fd, body.encode("utf-8", errors="replace"))
        finally:
            os.close(fd)
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def _build_cmd(path: str, argv: list[str]) -> str:
        # The sandbox sees /tmp; the file path is already there. Quote each
        # arg defensively so the LLM can't smuggle a shell metacharacter.
        import shlex
        quoted = " ".join(shlex.quote(a) for a in argv)
        return f"python3 {shlex.quote(path)} {quoted}".strip()


__all__ = [
    "EphemeralRunner",
    "EphemeralRun",
    "EPHEMERAL_CONTRACT",
    "extract_python",
    "parse_last_json_line",
]
