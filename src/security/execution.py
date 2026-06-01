"""Sandboxing for LLM-emitted bash — ONE module, TWO modes.

Every bash command Jarvis runs is LLM-emitted (there is no static command
book), so it is always confined. The two modes differ only by *integration
shape*, not by security philosophy — both are namespace sandboxes that drop
the network, caps and the real $HOME. Keeping them side by side here (instead
of scattering one into common/parser.py) makes that shared intent explicit and
the hardening easy to audit in one place.

MODE 1 — :class:`ShadowExec` (isolated dry-run, captures rc/stdout).
    Used to *simulate* a command before it ever touches the host. We spawn it
    directly (argv prefix to ``create_subprocess_exec``) so we can capture the
    real exit code. Engines, in order:
      1. **bwrap** (Bubblewrap) ``--unshare-all`` (no net/user/IPC/UTS),
         ``--die-with-parent``, sealed-off home (``--tmpfs /home`` & ``/root``),
         read-only-bound root so binaries still resolve.
      2. **podman run --rm --network none --read-only** container fallback for
         minimal images that ship podman but not bwrap.
      3. Pure-Python *refuse* — better to abort than run untrusted bash bare.
    rc==0 → HUD green safe-pulse + the caller asks for voice confirmation.

MODE 2 — :func:`wrap_sandbox` (inline jail string for a REAL, caged run).
    Used for the rare command the model self-flagged risky (e.g. ``curl|bash``):
    it must actually run on the host but caged. Since that goes through the
    shell pipeline (``run_bash``), this returns a *command string* prefixing the
    payload with a jail. Engines, in order: **firejail** (namespaces + seccomp +
    cap-drop + ``--net=none`` + blacklists) → **systemd-run --user --scope**
    (cgroup Protect* knobs) → raw bash. The dry-run prefers bwrap (it captures
    rc cleanly via argv); wrap prefers firejail (it composes as a shell prefix) —
    same idea, two entry shapes.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import shlex
import shutil
from collections.abc import Iterable

log = logging.getLogger("jarvis.shadow")


@dataclasses.dataclass(frozen=True)
class ShadowResult:
    rc: int
    stdout: str
    stderr: str
    engine: str            # "bwrap" | "podman" | "refused"
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


class ShadowExec:
    """Sandbox runner. Stateless apart from the engine choice cache."""

    BWRAP_ARGS = (
        "--unshare-all",
        "--die-with-parent",
        "--clearenv",
        "--setenv", "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "HOME", "/tmp",
        # Read-only root, fresh /tmp + /home so the command can't touch
        # anything outside the sandbox.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/sbin", "/sbin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--tmpfs", "/root",
        "--proc", "/proc",
        "--dev", "/dev",
        "--symlink", "/usr/bin", "/usr/local/bin",
        "--chdir", "/tmp",
    )

    PODMAN_IMAGE = "docker.io/library/alpine:latest"
    PODMAN_ARGS = (
        "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    )

    def __init__(
        self,
        bwrap_path: str | None = None,
        podman_path: str | None = None,
        timeout: float = 8.0,
        podman_image: str | None = None,
    ) -> None:
        self.bwrap_path = bwrap_path if bwrap_path is not None else shutil.which("bwrap")
        self.podman_path = podman_path if podman_path is not None else shutil.which("podman")
        self.timeout = timeout
        if podman_image is not None:
            self.PODMAN_IMAGE = podman_image

    @property
    def available_engines(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.bwrap_path:
            out.append("bwrap")
        if self.podman_path:
            out.append("podman")
        return tuple(out)

    # ----- public API -------------------------------------------------
    async def run(self, command: str, extra_env: dict[str, str] | None = None) -> ShadowResult:
        """Run ``command`` through the highest-priority available sandbox."""
        if not command.strip():
            return ShadowResult(0, "", "", "refused")
        if self.bwrap_path:
            return await self._run_bwrap(command, extra_env)
        if self.podman_path:
            return await self._run_podman(command, extra_env)
        log.warning("no sandbox engine available; refusing to execute")
        return ShadowResult(
            rc=126, stdout="", stderr="shadow_exec: no sandbox engine (bwrap/podman) found",
            engine="refused",
        )

    # ----- engines ----------------------------------------------------
    async def _run_bwrap(self, command: str, extra_env: dict[str, str] | None) -> ShadowResult:
        argv = [self.bwrap_path, *self.BWRAP_ARGS]
        for k, v in (extra_env or {}).items():
            argv += ["--setenv", k, v]
        argv += ["/bin/bash", "-lc", command]
        return await self._spawn(argv, "bwrap")

    async def _run_podman(self, command: str, extra_env: dict[str, str] | None) -> ShadowResult:
        argv = [self.podman_path, *self.PODMAN_ARGS]
        for k, v in (extra_env or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv += [self.PODMAN_IMAGE, "sh", "-lc", command]
        return await self._spawn(argv, "podman")

    async def _spawn(self, argv: Iterable[str], engine: str) -> ShadowResult:
        argv = list(argv)
        log.debug("shadow_exec[%s]: %s", engine, " ".join(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            return ShadowResult(rc=127, stdout="", stderr=str(e), engine=engine)
        except Exception as e:
            log.exception("shadow_exec spawn failed")
            return ShadowResult(rc=126, stdout="", stderr=str(e), engine=engine)

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Дожинаем убитый процесс в текущем event loop. Иначе его
            # subprocess-transport доживает до GC уже ПОСЛЕ закрытия лупа
            # (классический "Event loop is closed" из BaseSubprocessTransport.
            # __del__) — и оставляет зомби до reap'а.
            try:
                await proc.wait()
            except Exception:
                log.debug("reaping timed-out shadow proc failed", exc_info=True)
            return ShadowResult(
                rc=124, stdout="", stderr=f"shadow_exec[{engine}]: timed out after {self.timeout}s",
                engine=engine, timed_out=True,
            )

        return ShadowResult(
            rc=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout.decode("utf-8", errors="replace")[:65536],
            stderr=stderr.decode("utf-8", errors="replace")[:8192],
            engine=engine,
        )


# Tracked for telemetry / risk-scoring only — the sandbox decision is
# uniform now (LLM-emitted bash always goes through shadow_exec).
DESTRUCTIVE_SUBSTRINGS = ("rm ", "rm -", "dd ", "mkfs", "shred", ":(){:|:&};:", "curl ", "wget ")


def needs_shadow(command: str) -> bool:
    """Every LLM-emitted bash command goes through the sandbox first.

    There is no static command book to whitelist from anymore — the
    vision is LLM-only intent resolution. Empty strings are the only
    exemption."""
    return bool(command.strip())


# ───────────────────── MODE 2: inline jail string (real, caged run) ─────────
def wrap_sandbox(cmd: str) -> str:
    """Wrap a potentially-untrusted LLM-generated command in a hardened jail.

    Returns a *command string* (not a result): the caller runs it through the
    normal shell pipeline, so the command really executes — but caged. Defaults
    are deliberately aggressive — this is for commands the model flagged risky
    itself (e.g. ``curl|bash``). Anything that truly needs real $HOME or live
    network shouldn't be sandboxed at all: it belongs on the unsandboxed path
    behind the destructive-command voice gate (core.Jarvis._gate_destructive).

    Layer order: firejail (preferred — namespace + seccomp + cap drop) →
    systemd-run user scope (cgroup + ProtectHome/ProtectSystem) → raw bash.

    Contrast with :class:`ShadowExec`, which *simulates* in full isolation and
    captures rc; this wraps a real run. Same security intent, different shape."""
    if not cmd.strip():
        return cmd
    if shutil.which("firejail"):
        flags = [
            "--quiet",
            "--noprofile",
            "--private",              # tmpfs $HOME — no leaking ~/.ssh, ~/.config, ~/jarvis
            "--private-tmp",
            "--private-dev",          # minimal /dev — no raw block devices, no /dev/mem
            "--net=none",             # no outbound packets, no DNS
            "--caps.drop=all",        # CAP_NET_RAW, CAP_SYS_ADMIN, … all gone
            "--nonewprivs",           # PR_SET_NO_NEW_PRIVS — defeats setuid escalation
            "--seccomp",              # default deny-list against ptrace, mount, kexec, etc.
            "--blacklist=/root",
            "--blacklist=/etc/shadow",
            "--blacklist=/etc/sudoers",
            "--blacklist=/etc/sudoers.d",
            "--read-only=/etc",
            "--read-only=/usr",
        ]
        return "firejail " + " ".join(flags) + " -- bash -c " + shlex.quote(cmd)
    if shutil.which("systemd-run"):
        # systemd-run --user --scope can't do user-namespace tricks, but the
        # cgroup-level Protect* knobs and RestrictAddressFamilies still give a
        # serviceable jail when firejail isn't installed.
        flags = [
            "--user", "--scope", "--quiet",
            "-p", "PrivateTmp=yes",
            "-p", "PrivateDevices=yes",
            "-p", "ProtectHome=tmpfs",
            "-p", "ProtectSystem=strict",
            "-p", "ProtectKernelModules=yes",
            "-p", "ProtectKernelTunables=yes",
            "-p", "ProtectControlGroups=yes",
            "-p", "NoNewPrivileges=yes",
            "-p", "CapabilityBoundingSet=",
            "-p", "RestrictNamespaces=yes",
            "-p", "RestrictAddressFamilies=AF_UNIX",
        ]
        return "systemd-run " + " ".join(flags) + " -- bash -c " + shlex.quote(cmd)
    return cmd


__all__ = [
    "ShadowExec",
    "ShadowResult",
    "needs_shadow",
    "wrap_sandbox",
    "DESTRUCTIVE_SUBSTRINGS",
]
