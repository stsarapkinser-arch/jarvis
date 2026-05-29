"""Ephemeral Programming round-trip test.

We don't need bwrap/podman here — we hand the runner a stub
``ShadowExec`` whose ``run`` actually launches the script through the
*system* python and returns a ``ShadowResult``. That gives a real
end-to-end check of:

* extracting ``<python>…</python>`` from a (mock) LLM blob
* writing the script to ``/tmp`` with ``0o700``
* running it and parsing the trailing JSON line
* the ``finally`` cleanup that has to remove the temp file even if the
  script crashes
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.memory.ephemeral as ephemeral
import src.security.execution as shadow_exec


class DirectShadow(shadow_exec.ShadowExec):
    """Replaces the sandbox with an unsandboxed direct subprocess call.

    Tests only — keeps the round-trip honest without depending on a
    sandbox binary. We still go through ``asyncio.create_subprocess_exec``
    so the result decoding & timeout codepath is exercised."""

    def __init__(self) -> None:
        super().__init__(bwrap_path="", podman_path="")
        self.last_cmd: str | None = None

    async def run(self, command: str, extra_env=None):
        import shlex
        self.last_cmd = command
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return shadow_exec.ShadowResult(
            rc=proc.returncode or 0,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            engine="direct-test",
        )


def test_extract_python_picks_block():
    extracted = ephemeral.extract_python(
        "preface\n<python>print('hi')\n</python>\nafter"
    )
    assert extracted is not None
    body, argv = extracted
    assert body == "print('hi')"
    assert argv == []


def test_extract_python_with_args():
    text = "<args> a b 'c d' </args><python>import sys; print(sys.argv)</python>"
    body, argv = ephemeral.extract_python(text)
    assert "import sys" in body
    assert argv[:2] == ["a", "b"]


def test_extract_python_returns_none_when_missing():
    assert ephemeral.extract_python("nothing here") is None


def test_parse_last_json_line_picks_trailing_object():
    out = "preamble\ndebug: 1\n{\"thought\": \"ok\", \"result\": 42}\n"
    parsed = ephemeral.parse_last_json_line(out)
    assert parsed == {"thought": "ok", "result": 42}


def test_parse_last_json_line_ignores_when_trail_is_garbage():
    assert ephemeral.parse_last_json_line("garbage tail\n") is None


def test_ephemeral_round_trip_cleans_up(tmp_path: Path):
    runner = ephemeral.EphemeralRunner(shadow=DirectShadow(), tmp_dir=str(tmp_path))
    llm = """
prelude\n<python>
import json
print(json.dumps({"thought": "hello", "result": 7, "speak": "done"}))
</python>
tail
"""
    res = asyncio.run(runner.run_from_llm(llm))
    assert res is not None
    assert res.ok
    assert res.parsed == {"thought": "hello", "result": 7, "speak": "done"}
    # the /tmp file is gone — that's the whole point.
    assert not any(tmp_path.iterdir())


def test_ephemeral_unlinks_even_on_crash(tmp_path: Path):
    runner = ephemeral.EphemeralRunner(shadow=DirectShadow(), tmp_dir=str(tmp_path))
    llm = "<python>raise SystemExit(2)\n</python>"
    res = asyncio.run(runner.run_from_llm(llm))
    assert res is not None
    assert res.rc != 0
    assert res.parsed is None
    assert not any(tmp_path.iterdir()), "ephemeral runner leaked a script on crash"


def test_ephemeral_run_from_llm_returns_none_without_block():
    runner = ephemeral.EphemeralRunner(shadow=DirectShadow())
    assert asyncio.run(runner.run_from_llm("plain text only")) is None


def test_ephemeral_returns_full_stdout(tmp_path: Path):
    runner = ephemeral.EphemeralRunner(shadow=DirectShadow(), tmp_dir=str(tmp_path))
    llm = "<python>import json; print('debug'); print(json.dumps({'result': 1}))</python>"
    res = asyncio.run(runner.run_from_llm(llm))
    assert "debug" in res.stdout
    assert res.parsed == {"result": 1}
