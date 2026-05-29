from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import TypedDict

log = logging.getLogger("jarvis.snapshot")

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


class StateSnapshot(TypedDict, total=False):
    window: str
    cpu_pct: float
    ram_pct: float
    load_avg: float
    hour: int
    ts: float


_WINDOW_PROBES: tuple[list[str], ...] = (
    ["kdotool", "getactivewindow", "getwindowname"],
    ["xdotool", "getactivewindow", "getwindowname"],
)


def get_active_window_title() -> str:
    for cmd in _WINDOW_PROBES:
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.check_output(
                cmd, text=True, timeout=1, stderr=subprocess.DEVNULL
            )
            return out.strip()[:128]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        except Exception:
            log.exception("active-window probe %s failed", cmd[0])
    return ""


def _cpu_ram() -> tuple[float, float]:
    if _HAS_PSUTIL:
        try:
            return float(psutil.cpu_percent(interval=None)), float(psutil.virtual_memory().percent)
        except Exception:
            log.exception("psutil snapshot failed")
    return 0.0, 0.0


def _load() -> float:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return 0.0


def snapshot() -> StateSnapshot:
    """Capture the OS state at this moment. All probes are best-effort: failure → 0/''."""
    cpu, ram = _cpu_ram()
    return StateSnapshot(
        window=get_active_window_title(),
        cpu_pct=cpu,
        ram_pct=ram,
        load_avg=_load(),
        hour=time.localtime().tm_hour,
        ts=time.time(),
    )


def format_snapshot(snap: StateSnapshot | dict) -> str:
    if not snap:
        return ""
    parts: list[str] = []
    if snap.get("window"):
        parts.append(f"window={snap['window']!r}")
    if snap.get("cpu_pct"):
        parts.append(f"cpu={snap['cpu_pct']:.0f}%")
    if snap.get("ram_pct"):
        parts.append(f"ram={snap['ram_pct']:.0f}%")
    if snap.get("load_avg"):
        parts.append(f"load={snap['load_avg']:.2f}")
    if snap.get("hour") is not None:
        parts.append(f"hour={snap['hour']:02d}")
    return "[" + " ".join(parts) + "]" if parts else ""
