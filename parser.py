from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from typing import NamedTuple

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
SAY_RE = re.compile(r"<say>(.*?)</say>", re.DOTALL | re.IGNORECASE)
SANDBOX_RE = re.compile(r"<sandbox>\s*(true|1|yes|on)\s*</sandbox>", re.IGNORECASE)
FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")

SUDO_VERBS = (
    "apt", "apt-get", "dpkg", "snap", "pip", "pip3",
    "systemctl", "journalctl", "service",
    "ip", "ifconfig", "iwconfig", "nmcli", "iw", "rfkill",
    "mount", "umount", "fdisk", "parted", "mkfs", "fsck", "blkid", "cryptsetup",
    "iptables", "nft", "ufw", "ip6tables",
    "modprobe", "rmmod", "insmod",
    "nmap", "tcpdump", "tshark", "wireshark", "aircrack-ng", "airmon-ng",
    "useradd", "usermod", "userdel", "groupadd", "passwd", "chown", "chmod",
)


class ParsedResponse(NamedTuple):
    thought: str
    say: str
    bash: str
    sandbox: bool


def parse_response(raw: str) -> ParsedResponse:
    thought = ""
    say = ""
    m = THOUGHT_RE.search(raw)
    if m:
        thought = m.group(1).strip()
    m = SAY_RE.search(raw)
    if m:
        say = m.group(1).strip()
    sandbox = bool(SANDBOX_RE.search(raw))
    bash = THOUGHT_RE.sub("", raw)
    bash = SAY_RE.sub("", bash)
    bash = SANDBOX_RE.sub("", bash)
    bash = clean_bash(bash)
    return ParsedResponse(thought, say, bash, sandbox)


def clean_bash(s: str) -> str:
    s = FENCE_RE.sub("", s)
    s = s.replace("`", "")
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def inject_sudo(cmd: str) -> str:
    stripped = cmd.lstrip()
    if not stripped or stripped.startswith("sudo "):
        return cmd
    first = stripped.split(None, 1)[0]
    if first in SUDO_VERBS:
        return "sudo -n " + cmd
    return cmd


def wrap_sandbox(cmd: str) -> str:
    """Wrap a potentially-untrusted command in a sandbox.
    Prefers firejail; falls back to systemd-run --user --scope with hardening."""
    if not cmd.strip():
        return cmd
    if shutil.which("firejail"):
        return (
            "firejail --quiet --private-tmp --net=none "
            "--noprofile -- bash -c " + shlex.quote(cmd)
        )
    if shutil.which("systemd-run"):
        return (
            "systemd-run --user --scope --quiet "
            "-p PrivateTmp=yes -p ProtectHome=read-only -p NoNewPrivileges=yes "
            "-- bash -c " + shlex.quote(cmd)
        )
    return cmd


def run_bash(cmd: str, timeout: float = 5.0) -> tuple[int | None, str, str]:
    if not cmd.strip():
        return 0, "", ""
    proc = subprocess.Popen(
        cmd,
        shell=True,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return None, "", ""
