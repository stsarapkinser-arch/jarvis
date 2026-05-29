from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from event_bus import Event, EventBus, EventType
from singleton import Singleton

log = logging.getLogger("jarvis.pixel")

KDECONNECT_SERVICE = "org.kde.kdeconnect"
NOTIFICATIONS_IFACE = "org.kde.kdeconnect.device.notifications"
NOTIFICATION_IFACE = "org.kde.kdeconnect.device.notifications.notification"
SHARE_IFACE = "org.kde.kdeconnect.device.share"
BATTERY_IFACE = "org.kde.kdeconnect.device.battery"

# Pixel battery thresholds (per spec). We re-arm the alarm when the phone
# climbs back above HIGH so the operator can be warned again on the next
# discharge cycle without spamming.
BATTERY_LOW_PCT = 15
BATTERY_REARM_PCT = 30

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
# SMS one-time-password detector. We only match notifications that *also*
# carry a keyword cue ("код", "code", "verification", "OTP", "пароль", "пин"),
# otherwise random 4-8 digit substrings in news/Telegram chat would spam the
# HUD. The captured group is the digit sequence.
OTP_KEYWORD_RE = re.compile(
    r"(?:\b(?:код|code|verification|otp|пароль|пин|pin|подтвержд\w*|confirm\w*)\b)",
    re.IGNORECASE,
)
OTP_DIGITS_RE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")
SMS_APP_HINTS = (
    "sms", "messaging", "messages", "сообщ", "google messages",
    "telegram", "whatsapp", "viber", "signal",
)
# Pixel apps that ship reminders / calendar events we want mirrored
# verbatim to Kali via notify-send. Matched substring-wise on the lower-
# cased appName so locale variants ("Календарь", "Calendar") both hit.
REMINDER_APP_HINTS = (
    "calendar", "календар",
    "reminder", "напомин",
    "tasks", "todoist", "any.do", "todo",
    "clock", "часы", "alarm", "будильник",
    "keep",  # Google Keep reminders
)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
DOWNLOADS_DIRS = (
    Path.home() / "Downloads",
    Path.home() / "Загрузки",
    Path.home() / "KDE Connect",
)


class PixelBridge(metaclass=Singleton):
    """Subscribes to KDE Connect D-Bus signals via dbus-next.
    Five roles:
      1. Incoming-call detection → publish PIXEL_EVENT call_incoming + pause media.
      2. Call-end detection (notification removed) → publish call_ended + resume media.
      3. Quick-command notifications (prefixed [JARVIS] / @jarvis) → publish PIXEL_EVENT quick_command.
      4. Reminder / calendar / alarm notifications → publish PIXEL_EVENT reminder (mirrored to KDE notify-send by core).
      5. Shared images → tesseract OCR → publish PIXEL_EVENT image_ocr + open konsole."""

    # Call notification TTL safety net. If KDE Connect never sends the
    # ``notification_removed`` signal (some Android builds drop it), we still
    # need to resume media. Anything older than this in ``_active_calls`` is
    # swept on every new notification and treated as call_ended.
    CALL_NOTIFICATION_TTL_SEC = 600.0
    # Per-OTP / per-reminder dedup window. The same notification can fire
    # twice from KDE Connect (one for the app, one for the tray).
    DEDUP_WINDOW_SEC = 300.0

    def __init__(self) -> None:
        self.bus = EventBus()
        self._task: asyncio.Task | None = None
        self._mbus: Any = None
        self._handled_files: set[str] = set()
        # per-device battery state so we only fire the LOW alert once per
        # discharge cycle. {dev_id: "low" | "ok"}
        self._battery_state: dict[str, str] = {}
        # per-OTP code deduplication — the same notification can fire twice
        # (one for the SMS app, one for the system tray).
        self._recent_otps: dict[str, float] = {}
        # Active call tracking: maps the KDE Connect notification public_id
        # → {dev_id, caller, ts}. Populated on call_incoming, drained by the
        # notification_removed signal (which fires call_ended + resume).
        self._active_calls: dict[str, dict[str, Any]] = {}
        # Per-reminder dedup: digest (title|body) → last_seen_ts.
        self._recent_reminders: dict[str, float] = {}

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

    async def _resume_media(self) -> None:
        """Resume any media player paused on the incoming call.

        ``playerctl --all-players play`` is a no-op for players that are
        already playing, so this is safe to fire even when nothing was paused.
        """
        if not shutil.which("playerctl"):
            log.info("playerctl not installed; skipping media resume")
            return
        try:
            await asyncio.create_subprocess_exec(
                "playerctl", "--all-players", "play",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            log.exception("playerctl play failed")

    def _sweep_stale_calls(self) -> list[dict[str, Any]]:
        """Drop call entries older than the TTL and return them so the caller
        can publish call_ended for each. Defensive: some Android builds never
        emit notification_removed when the call is dismissed."""
        if not self._active_calls:
            return []
        cutoff = time.time() - self.CALL_NOTIFICATION_TTL_SEC
        stale: list[dict[str, Any]] = []
        for pid in list(self._active_calls.keys()):
            info = self._active_calls[pid]
            if float(info.get("ts", 0.0)) < cutoff:
                stale.append(self._active_calls.pop(pid))
        return stale

    async def _publish_call_ended(self, info: dict[str, Any]) -> None:
        """Emit call_ended event and resume media. Safe to call from anywhere.
        """
        caller = str(info.get("caller", "Unknown"))
        await self._resume_media()
        await self.bus.publish(Event(
            EventType.PIXEL_EVENT,
            {
                "kind": "call_ended",
                "caller": caller,
                "device": str(info.get("device", "")),
                "started_ts": float(info.get("ts", 0.0)),
            },
            urgency="normal",
        ))

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
            from dbus_next import BusType
            from dbus_next.aio import MessageBus
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
        except (TimeoutError, FileNotFoundError):
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

    async def _process_payload(
        self,
        payload: dict[str, Any],
        *,
        dev_id: str = "",
        public_id: str = "",
    ) -> None:
        text_blob = " ".join(
            str(v) for k, v in payload.items()
            if k in ("appName", "ticker", "title", "text", "summary")
        )
        app_lower = str(payload.get("appName", "")).lower()

        # Sweep stale calls on every notification: if a call notification
        # never got its removed-signal (some Android builds), TTL it here so
        # media isn't stuck paused indefinitely.
        for stale in self._sweep_stale_calls():
            await self._publish_call_ended(stale)

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

        # SMS one-time-password — check BEFORE file/call so an OTP delivered
        # via SMS doesn't get swallowed by another classifier. We only fire
        # when both a keyword cue and a 4-8 digit number are present.
        otp_code = self._extract_otp(text_blob, app_lower)
        if otp_code is not None:
            await self.bus.publish(Event(
                EventType.PIXEL_EVENT,
                {
                    "kind": "otp",
                    "code": otp_code,
                    "app": payload.get("appName", ""),
                    "raw": payload,
                },
                urgency="high",
            ))
            return

        file_m = FILE_RECEIVED_RE.search(text_blob)
        if file_m:
            hint = file_m.group(1)
            path = await asyncio.to_thread(self._scan_recent_image, hint)
            if path is not None:
                await self._handle_received_image(path)
                return

        # Reminders / calendar / alarms — mirror to KDE notify-send via core.
        # Detected purely by appName so the operator can add more apps to
        # ``REMINDER_APP_HINTS`` without touching matching logic. We check
        # this BEFORE call so an alarm titled "Звонок будильника" doesn't
        # get misclassified as a phone call.
        if self._is_reminder_app(app_lower):
            reminder = self._build_reminder_event(payload)
            if reminder is not None:
                await self.bus.publish(Event(
                    EventType.PIXEL_EVENT,
                    reminder,
                    urgency="normal",
                ))
            return

        if CALL_HINTS_RE.search(text_blob):
            caller = (
                payload.get("title")
                or payload.get("appName")
                or payload.get("ticker")
                or "Unknown"
            )
            caller_str = str(caller)[:120]
            # Track this call so the notification_removed signal (and the
            # TTL safety net above) can fire call_ended later.
            if public_id:
                self._active_calls[public_id] = {
                    "device": dev_id,
                    "caller": caller_str,
                    "ts": time.time(),
                }
            await self._pause_media()
            # PIXEL_EVENT — для core.py (память + STATE_CHANGE + TTS).
            await self.bus.publish(Event(
                EventType.PIXEL_EVENT,
                {"kind": "call_incoming", "caller": caller_str, "raw": payload},
                urgency="high",
            ))
            # CALL_INBOUND — высокоуровневая обёртка для HUD: подписка
            # напрямую сюда переключает aura color без цепочки через
            # core._state и без зависимости от STATE_CHANGE("ALERT").
            await self.bus.publish(Event(
                EventType.CALL_INBOUND,
                {"caller": caller_str, "ts": time.time()},
                urgency="high",
            ))

    @staticmethod
    def _is_reminder_app(app_lower: str) -> bool:
        """Substring match against REMINDER_APP_HINTS."""
        if not app_lower:
            return False
        return any(hint in app_lower for hint in REMINDER_APP_HINTS)

    def _build_reminder_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Assemble a ``kind=reminder`` event payload from a KDE Connect
        notification dict. Returns ``None`` when the notification is empty
        or duplicates a reminder we forwarded in the last DEDUP_WINDOW_SEC.
        """
        title = str(payload.get("title", "")).strip()
        text = str(payload.get("text", "")).strip()
        ticker = str(payload.get("ticker", "")).strip()
        app = str(payload.get("appName", "")).strip()
        body = text or ticker
        if not title and not body:
            return None
        digest = f"{app}|{title}|{body}"[:512]
        now = time.time()
        last = self._recent_reminders.get(digest, 0.0)
        if now - last < self.DEDUP_WINDOW_SEC:
            return None
        self._recent_reminders[digest] = now
        # Cap dict size — reminders are infrequent, but be defensive.
        if len(self._recent_reminders) > 64:
            cutoff = now - self.DEDUP_WINDOW_SEC * 2
            self._recent_reminders = {
                k: v for k, v in self._recent_reminders.items() if v >= cutoff
            }
        return {
            "kind": "reminder",
            "app": app,
            "title": title or app or "Pixel",
            "body": body,
            "raw": payload,
        }

    def _extract_otp(self, text: str, app_lower: str) -> str | None:
        """Return the OTP digit sequence iff this notification looks like a
        verification code. Two-of-three rule: keyword cue OR SMS-app origin,
        plus a 4-8 digit number not part of a longer number/date."""
        if not text:
            return None
        from_sms_app = any(hint in app_lower for hint in SMS_APP_HINTS)
        has_keyword = bool(OTP_KEYWORD_RE.search(text))
        if not (from_sms_app or has_keyword):
            return None
        m = OTP_DIGITS_RE.search(text)
        if not m:
            return None
        # OTP-y digits are 4..8 contiguous; reject obvious years/dates.
        code = m.group(1)
        if code.startswith(("19", "20")) and len(code) == 4:
            # Likely a year; require keyword to override.
            if not has_keyword:
                return None
        # Dedup: same code within 5 minutes? skip.
        import time as _t
        now = _t.time()
        last = self._recent_otps.get(code, 0.0)
        if now - last < 300.0:
            return None
        self._recent_otps[code] = now
        # Cap dict size.
        if len(self._recent_otps) > 32:
            cutoff = now - 600.0
            self._recent_otps = {
                k: v for k, v in self._recent_otps.items() if v >= cutoff
            }
        return code

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

            def on_removed(public_id: str) -> None:
                asyncio.create_task(self._on_notification_removed(public_id))

            iface.on_notification_posted(on_posted)
            log.info("subscribed to notifications on %s", dev_id)
            # `notificationRemoved` is the signal KDE Connect fires when a
            # notification disappears from the phone — for an incoming call
            # this means the call was answered, dismissed, or missed. We
            # use it to resume media. Some older KDE Connect / Android
            # builds lack this signal; the TTL sweep in _process_payload is
            # the safety net.
            for handler_name in ("on_notification_removed", "on_notificationRemoved"):
                handler = getattr(iface, handler_name, None)
                if callable(handler):
                    handler(on_removed)
                    log.info(
                        "subscribed to notification_removed on %s via %s",
                        dev_id, handler_name,
                    )
                    break
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

        # Battery: subscribe to the `refreshed` signal (charge int, charging
        # bool). KDE Connect fires this on every change the phone reports,
        # roughly once a minute when charge moves and on plug/unplug — no
        # polling on our side.
        battery_path = f"/modules/kdeconnect/devices/{dev_id}/battery"
        try:
            introspection = await self._mbus.introspect(KDECONNECT_SERVICE, battery_path)
            obj = self._mbus.get_proxy_object(KDECONNECT_SERVICE, battery_path, introspection)
            batt_iface = obj.get_interface(BATTERY_IFACE)

            def on_refreshed(charge: int, charging: bool) -> None:
                asyncio.create_task(
                    self._on_battery(dev_id, int(charge), bool(charging))
                )

            for handler_name in ("on_refreshed", "on_state_changed"):
                handler = getattr(batt_iface, handler_name, None)
                if callable(handler):
                    handler(on_refreshed)
                    log.info("subscribed to battery on %s via %s", dev_id, handler_name)
                    break
            # Snapshot the current charge once on connect so a phone that's
            # already low triggers the alarm without waiting for the next
            # change event.
            charge = await self._read_battery_once(batt_iface)
            if charge is not None:
                await self._on_battery(dev_id, charge[0], charge[1])
        except Exception:
            log.debug("battery interface not available on %s", dev_id, exc_info=True)

    async def _read_battery_once(self, iface: Any) -> tuple[int, bool] | None:
        try:
            for charge_getter, charging_getter in (
                ("get_charge", "get_is_charging"),
                ("get_charge", "get_charging"),
            ):
                cg = getattr(iface, charge_getter, None)
                ig = getattr(iface, charging_getter, None)
                if callable(cg) and callable(ig):
                    return int(await cg()), bool(await ig())
        except Exception:
            log.debug("battery property read failed", exc_info=True)
        return None

    async def _on_battery(self, dev_id: str, charge: int, charging: bool) -> None:
        """Edge-trigger LOW once per discharge cycle. Re-arm at REARM%.

        Every refresh also publishes a non-urgent ``kind=battery`` event so
        the Context Weaver can quote the current charge in its prompt
        without having to peek into PixelBridge's private state."""
        if charge < 0 or charge > 100:
            return
        # Routine snapshot — fires on every KDE Connect refresh signal.
        try:
            await self.bus.publish(Event(
                EventType.PIXEL_EVENT,
                {
                    "kind": "battery",
                    "device": dev_id,
                    "charge": charge,
                    "charging": charging,
                },
                urgency="low",
            ))
        except Exception:
            log.exception("battery snapshot publish failed")

        prev = self._battery_state.get(dev_id, "ok")
        if charge >= BATTERY_REARM_PCT and prev == "low":
            self._battery_state[dev_id] = "ok"
            return
        if charge <= BATTERY_LOW_PCT and prev != "low" and not charging:
            self._battery_state[dev_id] = "low"
            await self.bus.publish(Event(
                EventType.PIXEL_EVENT,
                {
                    "kind": "battery_low",
                    "device": dev_id,
                    "charge": charge,
                    "charging": charging,
                },
                urgency="high",
            ))

    async def _on_share_received(self, file_path: str) -> None:
        try:
            path = Path(file_path).expanduser()
        except Exception:
            log.exception("bad share path %r", file_path)
            return
        if path.suffix.lower() in IMAGE_EXTS:
            await self._handle_received_image(path)

    async def _on_notification_removed(self, public_id: str) -> None:
        """KDE Connect dropped a notification on the phone. If it was an
        active call we tracked on call_incoming, treat this as call_ended
        and resume media.
        """
        info = self._active_calls.pop(public_id, None)
        if info is None:
            return
        await self._publish_call_ended(info)

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
        await self._process_payload(payload, dev_id=dev_id, public_id=public_id)

    async def push_to_phone(self, title: str, body: str) -> None:
        """Send a notification to the phone via KDE Connect ping-msg.

        Best-effort: fails silently if kdeconnect-cli is not installed or
        no device is paired. Used by core.py for critical alerts (disk full,
        thermal critical, intrusion detected)."""
        cli = shutil.which("kdeconnect-cli")
        if not cli:
            return
        msg = f"{title}: {body}" if body else title
        for dev_id in self._devices():
            try:
                proc = await asyncio.create_subprocess_exec(
                    cli, "-d", dev_id, "--ping-msg", msg[:200],
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                log.debug("push_to_phone failed for %s", dev_id, exc_info=True)

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
