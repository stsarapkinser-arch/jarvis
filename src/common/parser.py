from __future__ import annotations

import re
import shlex
import shutil
import subprocess

# Чистка markdown-ограждений в bash-командах из LLM-heal (единственный
# оставшийся потребитель этого модуля после перехода на Native Function Calling).
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
    """Wrap a potentially-untrusted LLM-generated command in a hardened jail.

    Defaults are deliberately aggressive — the sandbox is for commands the LLM
    flagged itself as risky via ``<sandbox>true</sandbox>``. Anything that
    truly needs real $HOME or live network shouldn't be sandboxed in the first
    place: it belongs on the unsandboxed path with the destructive-command
    voice gate (see core.Jarvis._gate_destructive).

    Layer order: firejail (preferred — namespace + seccomp + cap drop) →
    systemd-run user scope (cgroup + ProtectHome/ProtectSystem) → raw bash."""
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
        # Убиваем зависший процесс, иначе он остаётся жить осиротевшим (тем более
        # с start_new_session=True — переживёт родителя). После kill() добираем
        # вывод вторым communicate(), чтобы закрыть pipe'ы и не словить зомби.
        proc.kill()
        try:
            proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return None, "", ""
