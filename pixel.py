from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.pixel")

KDECONNECT_SERVICE = "org.kde.kdeconnect"
NOTIFICATIONS_IFACE = "org.kde.kdeconnect.device.notifications"
NOTIFICATION_IFACE = "org.kde.kdeconnect.device.notifications.notification"
SHARE_IFACE = "org.kde.kdeconnect.device.share"

CALL_HINTS_RE = re.compile(
    r"(incoming|ringing|missed|call|вход\w*\s+звон|пропущ\w*\s+вызов|звонит|телефон)",
    re.IGNORECASE,
)
QUICK_CMD_RE = re.compile(
    r"^\s*(?:\[(?:JARVIS|АКАЛИ|ДЖАРВИС)\]|@?(?:jarvis|джарвис|акали))[:\s]+(.+)$",
    re.IGNORECASE,
)
FILE_RECEIVED_RE = re.compile(
    r"(?:received|получен(?:о|а)?)[:\s].*?([^\s]+\.(?:png|jpg|jpeg|webp|bmp|tiff))",
    re.IGNORECASE,
)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
DOWNLOADS_DIRS = (
    Path.home() / "Downloads",
    Path.home() / "Загрузки",
    Path.home() / "KDE Connect",
)


class PixelBridge(metaclass=Singleton):
    """Subscribes to KDE Connect D-Bus signals via dbus-next.
    Three roles:
      1. Incoming-call detection → publish PIXEL_EVENT call_incoming + pause media.
      2. Quick-command notifications (prefixed [JARVIS] / @jarvis) → publish PIXEL_EVENT quick_command.
      3. Shared images → tesseract OCR → publish PIXEL_EVENT image_ocr + open konsole."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self._task: asyncio.Task | None = None
        self._mbus: Any = None
        self._handled_files: set[str] = set()

    async def _pause_media(self) -> None:
        if not shutil.which("playerctl"):
            log.info("playerctl not installed; skipping media pause")
            return
        try:
            await asyncio.create_subprocess_exec(
                "playerctl", "--all-players", "pause",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            log.exception("playerctl pause failed")

    @staticmethod
    def _devices() -> list[str]:
        if not shutil.which("kdeconnect-cli"):
            return []
        try:
            out = subprocess.check_output(
                ["kdeconnect-cli", "-l", "--id-only"],
                text=True, timeout=2, stderr=subprocess.DEVNULL,
            )
            return [ln for ln in out.strip().splitlines() if ln]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []

    async def _connect(self) -> Any:
        try:
            from dbus_next.aio import MessageBus
            from dbus_next import BusType
        except ImportError:
            log.warning("dbus-next not installed; PixelBridge disabled")
            return None
        try:
            return await MessageBus(bus_type=BusType.SESSION).connect()
        except Exception:
            log.exception("session bus connect failed")
            return None

    async def _run_ocr(self, image_path: Path) -> str:
        if not shutil.which("tesseract"):
            log.info("tesseract not installed; skipping OCR")
            return ""
        try:
            proc = await asyncio.create_subprocess_exec(
                "tesseract", str(image_path), "-", "-l", "rus+eng",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            return out.decode(errors="replace").strip()
        except (asyncio.TimeoutError, FileNotFoundError):
            return ""
        except Exception:
            log.exception("tesseract failed on %s", image_path)
            return ""

    async def _open_in_konsole(self, title: str, body: str) -> None:
        if not shutil.which("konsole"):
            return
        try:
            fd, tmp_path = await asyncio.to_thread(_mkstemp_text, body, title)
            del fd
            cmd = (
                "konsole", "-e", "bash", "-c",
                f"less {tmp_path}; rm -f {tmp_path}"
            )
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            log.exception("open in konsole failed")

    async def _handle_received_image(self, path: Path) -> None:
        key = str(path)
        if key in self._handled_files:
            return
        self._handled_files.add(key)
        if not path.is_file():
            return
        text = await self._run_ocr(path)
        await self.bus.publish(Event(
            EventType.PIXEL_EVENT,
            {"kind": "image_ocr", "path": str(path), "text": text[:4000]},
        ))
        if text:
            header = f"=== Pixel OCR: {path.name} ===\n"
            await self._open_in_konsole(path.name, header + text)

    @staticmethod
    def _scan_recent_image(filename_hint: str) -> Path | None:
        now = __import__("time").time()
        for d in DOWNLOADS_DIRS:
            if not d.is_dir():
                continue
            try:
                exact = d / filename_hint
                if exact.is_file() and exact.suffix.lower() in IMAGE_EXTS:
                    return exact
                for p in d.iterdir():
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if filename_hint and filename_hint.lower() in p.name.lower():
                        return p
                    try:
                        if now - p.stat().st_mtime < 60:
                            return p
                    except OSError:
                        continue
            except OSError:
                continue
        return None

    async def _process_payload(self, payload: dict[str, Any]) -> None:
        text_blob = " ".join(
            str(v) for k, v in payload.items()
            if k in ("appName", "ticker", "title", "text", "summary")
        )

        m = QUICK_CMD_RE.search(text_blob)
        if m:
            cmd_text = m.group(1).strip()
            if cmd_text:
                await self.bus.publish(Event(
                    EventType.PIXEL_EVENT,
                    {"kind": "quick_command", "payload": cmd_text, "raw": payload},
                    urgency="normal",
                ))
                return

        file_m = FILE_RECEIVED_RE.search(text_blob)
        if file_m:
            hint = file_m.group(1)
            path = await asyncio.to_thread(self._scan_recent_image, hint)
            if path is not None:
                await self._handle_received_image(path)
                return

        if CALL_HINTS_RE.search(text_blob):
            caller = (
                payload.get("title")
                or payload.get("appName")
                or payload.get("ticker")
                or "Unknown"
            )
            await self._pause_media()
            await self.bus.publish(Event(
                EventType.PIXEL_EVENT,
                {"kind": "call_incoming", "caller": str(caller)[:120], "raw": payload},
                urgency="high",
            ))

    async def _subscribe_device(self, dev_id: str) -> None:
        if self._mbus is None:
            return

        notif_path = f"/modules/kdeconnect/devices/{dev_id}/notifications"
        try:
            introspection = await self._mbus.introspect(KDECONNECT_SERVICE, notif_path)
            obj = self._mbus.get_proxy_object(KDECONNECT_SERVICE, notif_path, introspection)
            iface = obj.get_interface(NOTIFICATIONS_IFACE)

            def on_posted(public_id: str) -> None:
                asyncio.create_task(self._fetch_and_dispatch(dev_id, public_id))

            iface.on_notification_posted(on_posted)
            log.info("subscribed to notifications on %s", dev_id)
        except Exception:
            log.exception("notifications subscribe failed on %s", dev_id)

        share_path = f"/modules/kdeconnect/devices/{dev_id}/share"
        try:
            introspection = await self._mbus.introspect(KDECONNECT_SERVICE, share_path)
            obj = self._mbus.get_proxy_object(KDECONNECT_SERVICE, share_path, introspection)
            share_iface = obj.get_interface(SHARE_IFACE)

            def on_share(file_path: str) -> None:
                asyncio.create_task(self._on_share_received(file_path))

            for handler_name in ("on_share_received", "on_file_received"):
                handler = getattr(share_iface, handler_name, None)
                if callable(handler):
                    handler(on_share)
                    log.info("subscribed to share on %s via %s", dev_id, handler_name)
                    break
        except Exception:
            log.debug("share interface not available on %s", dev_id, exc_info=True)

    async def _on_share_received(self, file_path: str) -> None:
        try:
            path = Path(file_path).expanduser()
        except Exception:
            log.exception("bad share path %r", file_path)
            return
        if path.suffix.lower() in IMAGE_EXTS:
            await self._handle_received_image(path)

    async def _fetch_and_dispatch(self, dev_id: str, public_id: str) -> None:
        if self._mbus is None:
            return
        path = f"/modules/kdeconnect/devices/{dev_id}/notifications/{public_id}"
        try:
            introspection = await self._mbus.introspect(KDECONNECT_SERVICE, path)
            obj = self._mbus.get_proxy_object(KDECONNECT_SERVICE, path, introspection)
            iface = obj.get_interface(NOTIFICATION_IFACE)
        except Exception:
            log.debug("fetch notification %s/%s failed", dev_id, public_id, exc_info=True)
            return
        payload: dict[str, Any] = {}
        for attr in ("appName", "ticker", "title", "text"):
            try:
                getter = getattr(iface, f"get_{attr.lower()}", None)
                if getter is None:
                    continue
                val = await getter()
                payload[attr] = val
            except Exception:
                continue
        if not payload:
            return
        await self._process_payload(payload)

    async def run(self) -> None:
        self._mbus = await self._connect()
        if self._mbus is None:
            return
        for dev_id in self._devices():
            await self._subscribe_device(dev_id)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            log.info("PixelBridge cancelled")
            raise

    async def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="pixel-bridge")
        return self._task


def _mkstemp_text(body: str, label: str) -> tuple[int, str]:
    import tempfile
    fd, name = tempfile.mkstemp(prefix=f"jarvis-pixel-{label}-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    return -1, name
