from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import Callable, Pattern

from singleton import Singleton

log = logging.getLogger("jarvis.repair")

PatchFn = Callable[[re.Match, str], str | None]


@dataclass(frozen=True)
class Patch:
    name: str
    pattern: Pattern[str]
    fix: PatchFn


def _kill_port(m: re.Match, cmd: str) -> str | None:
    port_m = re.search(r":(\d{2,5})\b", cmd) or re.search(r"\b(\d{2,5})\b", cmd)
    if not port_m:
        return None
    return f"sudo -n fuser -k {port_m.group(1)}/tcp"


def _install_missing(m: re.Match, cmd: str) -> str | None:
    name = m.group(1).strip().strip("'\"")
    if not re.fullmatch(r"[a-zA-Z0-9._+-]{1,64}", name):
        return None
    return f"sudo -n apt-get install -y {shlex.quote(name)}"


def _prepend_sudo(m: re.Match, cmd: str) -> str | None:
    stripped = cmd.lstrip()
    if stripped.startswith("sudo "):
        return None
    return "sudo -n " + cmd


def _ensure_dir(m: re.Match, cmd: str) -> str | None:
    path_m = re.search(r"'([^']+)'", m.string) or re.search(r": (\S+?): No such file", m.string)
    if not path_m:
        return None
    path = path_m.group(1)
    parent = os.path.dirname(path) or "."
    return f"mkdir -p {shlex.quote(parent)} && {cmd}"


def _apt_update_retry(m: re.Match, cmd: str) -> str | None:
    if "apt" not in cmd and "apt-get" not in cmd:
        return None
    if "apt update" in cmd or "apt-get update" in cmd:
        return None
    return f"sudo -n apt-get update && {cmd}"


PATCHES: tuple[Patch, ...] = (
    Patch(
        "port_busy",
        re.compile(r"(Address already in use|address already in use|bind.*in use)", re.I),
        _kill_port,
    ),
    Patch(
        "command_not_found_bash",
        re.compile(r"(?:bash|sh|zsh): (\S+): command not found", re.I),
        _install_missing,
    ),
    Patch(
        "command_not_found_generic",
        re.compile(r"^([\w.+-]+): command not found", re.I | re.M),
        _install_missing,
    ),
    Patch(
        "permission_denied",
        re.compile(r"Permission denied", re.I),
        _prepend_sudo,
    ),
    Patch(
        "no_such_file",
        re.compile(r"No such file or directory", re.I),
        _ensure_dir,
    ),
    Patch(
        "apt_404",
        re.compile(r"(Failed to fetch|404 +Not Found|Unable to locate package)", re.I),
        _apt_update_retry,
    ),
)


class QuickPatcher(metaclass=Singleton):
    """Tier-1 fixer: regex on stderr → mechanical bash repair, no LLM round-trip."""

    def patch(self, cmd: str, stderr: str) -> tuple[str, str] | None:
        if not stderr:
            return None
        for p in PATCHES:
            m = p.pattern.search(stderr)
            if not m:
                continue
            try:
                fix = p.fix(m, cmd)
            except Exception:
                log.exception("patch fn %s threw", p.name)
                continue
            if fix and fix != cmd:
                log.info("quick-patch matched %s → %s", p.name, fix)
                return p.name, fix
        return None
