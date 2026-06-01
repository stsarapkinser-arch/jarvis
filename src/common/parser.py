from __future__ import annotations

import re
import subprocess

# NB: песочница (ShadowExec dry-run и wrap_sandbox для реального заключённого
# запуска) живёт в src/security/execution.py — одно место на весь sandboxing.
# Здесь — только текстовые утилиты bash (чистка, sudo-инъекция, простой запуск).

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
