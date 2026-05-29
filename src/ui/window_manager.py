from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.common.singleton import Singleton

log = logging.getLogger("jarvis.kwin")

KWIN_SERVICE: Final = "org.kde.KWin"
KWIN_SCRIPTING_PATH: Final = "/Scripting"


@dataclass(frozen=True)
class WindowInfo:
    caption: str
    res: str
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class ScreenLayout:
    width: int
    height: int
    windows: tuple[WindowInfo, ...]

    def quadrant_load(self) -> dict[str, float]:
        """Fraction of each screen quadrant covered by window area.
        Quadrants: NW, NE, SW, SE."""
        if self.width <= 0 or self.height <= 0:
            return {"NW": 0.0, "NE": 0.0, "SW": 0.0, "SE": 0.0}
        midx = self.width / 2.0
        midy = self.height / 2.0
        load = {"NW": 0.0, "NE": 0.0, "SW": 0.0, "SE": 0.0}
        for w in self.windows:
            for label, (qx0, qy0, qx1, qy1) in (
                ("NW", (0.0, 0.0, midx, midy)),
                ("NE", (midx, 0.0, self.width, midy)),
                ("SW", (0.0, midy, midx, self.height)),
                ("SE", (midx, midy, self.width, self.height)),
            ):
                ix0 = max(w.x, qx0)
                iy0 = max(w.y, qy0)
                ix1 = min(w.x + w.width, qx1)
                iy1 = min(w.y + w.height, qy1)
                if ix1 > ix0 and iy1 > iy0:
                    area = (ix1 - ix0) * (iy1 - iy0)
                    quad_area = (qx1 - qx0) * (qy1 - qy0)
                    if quad_area > 0:
                        load[label] += area / quad_area
        return load

    def freest_quadrant(self) -> str:
        load = self.quadrant_load()
        return min(load, key=lambda k: load[k])


# Screen quadrant → (x_ratio, y_ratio) for the HUD core centre.
QUADRANT_TO_RATIO: Final[dict[str, tuple[float, float]]] = {
    "NW": (0.18, 0.22),
    "NE": (0.82, 0.22),
    "SW": (0.18, 0.78),
    "SE": (0.82, 0.78),
}


class KWinOrchestrator(metaclass=Singleton):
    """Drives KDE Plasma 6 KWin via org.kde.KWin.Scripting + kdotool/wmctrl helpers."""

    SCRIPT_DIR: Path = Path(__file__).resolve().parent / "scripts"
    INLINE_DIR: Path = Path(__file__).resolve().parent / "scripts" / "_inline"

    def __init__(self) -> None:
        self._qdbus: str | None = shutil.which("qdbus6") or shutil.which("qdbus")
        if not self._qdbus:
            log.warning("neither qdbus6 nor qdbus found; KWin disabled")
        self._kdotool: str | None = shutil.which("kdotool")
        self._wmctrl: str | None = shutil.which("wmctrl")
        try:
            self.INLINE_DIR.mkdir(exist_ok=True)
        except OSError:
            log.exception("cannot create inline scripts dir")

    async def _qdbus_call(self, *args: str) -> tuple[int, str, str]:
        if not self._qdbus:
            return 1, "", "qdbus not available"
        try:
            proc = await asyncio.create_subprocess_exec(
                self._qdbus, KWIN_SERVICE, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return proc.returncode or 0, stdout.decode().strip(), stderr.decode().strip()
        except TimeoutError:
            return 124, "", "qdbus timeout"
        except Exception as exc:
            log.exception("qdbus call failed")
            return 1, "", str(exc)

    async def load_script(self, path: Path, plugin_name: str = "jarvis") -> int | None:
        rc, out, err = await self._qdbus_call(
            KWIN_SCRIPTING_PATH, "loadScript", str(path), plugin_name
        )
        if rc != 0:
            log.error("loadScript failed: rc=%s err=%s", rc, err)
            return None
        try:
            return int(out)
        except ValueError:
            log.error("loadScript returned non-int: %r", out)
            return None

    async def run_script(self, script_id: int) -> bool:
        rc, _, err = await self._qdbus_call(f"/Scripting/Script{script_id}", "run")
        if rc != 0:
            log.error("run script %s failed: %s", script_id, err)
            return False
        return True

    async def unload_script(self, plugin_name: str = "jarvis") -> None:
        rc, _, err = await self._qdbus_call(
            KWIN_SCRIPTING_PATH, "unloadScript", plugin_name
        )
        if rc != 0:
            log.debug("unloadScript %s: %s", plugin_name, err)

    async def execute(self, name: str) -> bool:
        path = self.SCRIPT_DIR / f"{name}.js"
        if not path.is_file():
            log.error("kwin script not found: %s", path)
            return False
        plugin = f"jarvis-{name}"
        script_id = await self.load_script(path, plugin_name=plugin)
        if script_id is None:
            return False
        try:
            return await self.run_script(script_id)
        finally:
            await self.unload_script(plugin)

    async def run_inline(self, js: str, label: str = "inline") -> bool:
        """Write JS to a temp file in scripts/_inline/, load+run+unload, clean up."""
        plugin = f"jarvis-inline-{label}-{os.getpid()}"
        try:
            fd, name = tempfile.mkstemp(
                prefix=f"{label}-", suffix=".js", dir=str(self.INLINE_DIR)
            )
            os.close(fd)
            path = Path(name)
            path.write_text(js, encoding="utf-8")
        except OSError:
            log.exception("inline JS write failed")
            return False
        try:
            script_id = await self.load_script(path, plugin_name=plugin)
            if script_id is None:
                return False
            return await self.run_script(script_id)
        finally:
            await self.unload_script(plugin)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    async def prepare_workspace(self) -> bool:
        return await self.execute("prepare_workspace")

    async def pin_jarvis_hud(self, caption_substr: str = "JarvisHUD") -> bool:
        """Force-pin HUD-окно на Plasma 6 Wayland.

        Qt-флаги ``WindowStaysOnTopHint`` / ``Tool`` под Wayland —
        advisory: KWin волен опустить окно при смене workspace, входе
        в полноэкранное приложение и т.п. Этот метод через KWin Script
        ставит native-флаги (``keepAbove`` / ``skipTaskbar`` /
        ``skipPager`` / ``skipSwitcher``), которые компоновщик
        обязан соблюдать.

        Идемпотентно: повторный вызов просто перезаписывает те же флаги.
        Подбирается по подстроке caption (см. HUD_WINDOW_TITLE).
        """
        js = (
            f"const target = {json.dumps(caption_substr)};\n"
            "const wins = workspace.windowList ? workspace.windowList()\n"
            "    : (workspace.clientList ? workspace.clientList() : []);\n"
            "let pinned = 0;\n"
            "for (let i = 0; i < wins.length; i++) {\n"
            "    const w = wins[i];\n"
            "    if (!w || !w.caption) continue;\n"
            "    if (w.caption.indexOf(target) === -1) continue;\n"
            "    try { w.keepAbove = true; } catch (e) {}\n"
            "    try { w.skipTaskbar = true; } catch (e) {}\n"
            "    try { w.skipPager = true; } catch (e) {}\n"
            "    try { w.skipSwitcher = true; } catch (e) {}\n"
            "    try { w.onAllDesktops = true; } catch (e) {}\n"
            "    // Plasma 6: попытаться поднять окно в notification-слой\n"
            "    // (выше Plasma-панелей). API менялось между релизами KWin —\n"
            "    // оборачиваем КАЖДУЮ попытку в try/catch, чтобы любая\n"
            "    // несовместимость не валила пин остальных свойств.\n"
            "    try {\n"
            "        if (typeof KWin !== 'undefined' && KWin.NotificationLayer !== undefined) {\n"
            "            w.layer = KWin.NotificationLayer;\n"
            "        }\n"
            "    } catch (e) {}\n"
            "    try {\n"
            "        if (typeof KWin !== 'undefined' && KWin.OnScreenDisplayLayer !== undefined && w.layer === undefined) {\n"
            "            w.layer = KWin.OnScreenDisplayLayer;\n"
            "        }\n"
            "    } catch (e) {}\n"
            "    pinned++;\n"
            "}\n"
            "print('jarvis-hud pin: ' + pinned + ' window(s)');\n"
        )
        return await self.run_inline(js, label="pin-hud")

    async def highlight_window(self, pattern: str) -> bool:
        """Pulse any windows whose caption/resource matches the pattern (substring, case-insensitive)."""
        pattern = (pattern or "").strip()
        if not pattern:
            return False
        template_path = self.SCRIPT_DIR / "highlight_window.js"
        if not template_path.is_file():
            log.warning("highlight_window.js missing")
            return False
        template = template_path.read_text(encoding="utf-8")
        js = template.replace("__PATTERN__", json.dumps(pattern))
        return await self.run_inline(js, label="highlight")

    async def query_windows(self) -> ScreenLayout | None:
        """Best-effort window enumeration on Plasma 6 Wayland.
        Primary: kdotool (KWin-script-backed, works on Wayland).
        Fallback: wmctrl (X11 only) — returns None on pure Wayland without kdotool."""
        sw, sh = await asyncio.to_thread(self._screen_size_sync)
        if self._kdotool:
            windows = await self._enumerate_kdotool()
            if windows is not None:
                return ScreenLayout(sw, sh, tuple(windows))
        if self._wmctrl:
            windows = await self._enumerate_wmctrl()
            if windows is not None:
                return ScreenLayout(sw, sh, tuple(windows))
        return None

    @staticmethod
    def _screen_size_sync() -> tuple[int, int]:
        try:
            out = subprocess.check_output(
                ["xrandr"], text=True, timeout=1, stderr=subprocess.DEVNULL
            )
            m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return 1920, 1080

    async def _enumerate_kdotool(self) -> list[WindowInfo] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._kdotool, "search", "--name", ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            ids_b, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except (TimeoutError, FileNotFoundError):
            return None
        except Exception:
            log.exception("kdotool search failed")
            return None
        ids = [s for s in ids_b.decode().split() if s]
        if not ids:
            return []
        result: list[WindowInfo] = []
        for wid in ids[:64]:
            info = await self._kdotool_window(wid)
            if info is not None:
                result.append(info)
        return result

    async def _kdotool_window(self, wid: str) -> WindowInfo | None:
        async def _q(*args: str) -> str:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._kdotool, *args, wid,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
                return out.decode(errors="replace").strip()
            except (TimeoutError, FileNotFoundError):
                return ""

        caption = await _q("getwindowname")
        geom_raw = await _q("getwindowgeometry", "--shell")
        if not geom_raw:
            return None
        coords = {}
        for line in geom_raw.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                coords[k.strip()] = v.strip()
        try:
            x = int(coords.get("X", "0"))
            y = int(coords.get("Y", "0"))
            w = int(coords.get("WIDTH", "0"))
            h = int(coords.get("HEIGHT", "0"))
        except ValueError:
            return None
        if w <= 1 or h <= 1:
            return None
        return WindowInfo(caption=caption[:128], res="", x=x, y=y, width=w, height=h)

    async def _enumerate_wmctrl(self) -> list[WindowInfo] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._wmctrl, "-l", "-G",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except (TimeoutError, FileNotFoundError):
            return None
        result: list[WindowInfo] = []
        for line in out_b.decode(errors="replace").splitlines():
            parts = line.split(None, 7)
            if len(parts) < 8:
                continue
            try:
                x, y, w, h = (int(p) for p in parts[2:6])
            except ValueError:
                continue
            caption = parts[7].strip()
            if w > 1 and h > 1:
                result.append(WindowInfo(caption=caption[:128], res="", x=x, y=y, width=w, height=h))
        return result

    async def suggest_hud_corner(self) -> tuple[float, float] | None:
        """Returns (x_ratio, y_ratio) for HUD core position, or None if undetectable."""
        layout = await self.query_windows()
        if layout is None:
            return None
        quad = layout.freest_quadrant()
        return QUADRANT_TO_RATIO[quad]

    async def find_window_rects(self, pattern: str) -> list[dict]:
        """Return geometry dicts (x, y, w, h, caption) for every visible window
        whose caption (or resource name) contains ``pattern`` (case-insensitive).

        Used by the HUD app-glow layer: we paint Iron-Man-style neon brackets
        around the matched windows. Empty list when nothing matches or on
        Wayland sessions where neither kdotool nor wmctrl can enumerate the
        layout."""
        pat = (pattern or "").strip().lower()
        if not pat:
            return []
        layout = await self.query_windows()
        if layout is None:
            return []
        hits: list[dict] = []
        for w in layout.windows:
            caption_l = (w.caption or "").lower()
            res_l = (w.res or "").lower()
            if pat not in caption_l and pat not in res_l:
                continue
            hits.append({
                "x": int(w.x),
                "y": int(w.y),
                "w": int(w.width),
                "h": int(w.height),
                "caption": w.caption,
            })
        return hits
