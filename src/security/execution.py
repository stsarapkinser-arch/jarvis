"""Shadow Exec — predictive safe-execution layer.

Every bash command Jarvis runs is LLM-emitted (there is no static command
book) — so every command is first dry-run inside a sandbox. We try, in
order:

1. **bwrap** (Bubblewrap, ``flatpak --version`` shipping it on Kali by
   default) with ``--unshare-all`` (no network, no user, no IPC, no UTS),
   ``--die-with-parent`` and a sealed-off home (``--tmpfs /home``,
   ``--tmpfs /root``). The root filesystem is read-only-bound so the
   command can still resolve binaries.
2. **podman run --rm --network none --read-only --userns keep-id** as a
   second-line container fallback. Slower to spin up but it works on
   minimal Kali images that ship podman but not bwrap.
3. A pure-Python fallback that simply *refuses* to run — better to abort
   than to execute untrusted bash outside a sandbox.

If the sandboxed run returns ``rc == 0`` the HUD gets a green safe-pulse
and the operator is asked to confirm by voice (handled by the caller).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
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


__all__ = ["ShadowExec", "ShadowResult", "needs_shadow", "DESTRUCTIVE_SUBSTRINGS"]
