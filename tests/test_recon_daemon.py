"""Parser and helper tests for the Wraith ReconDaemon.

Headless: no subprocess spawning, no network. We exercise the public
classifier helpers and verify the rate-limiting / UFW-suggestion logic
without touching tshark / auth.log.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.services.recon_daemon as rd
from src.common.event_bus import EventBus
from src.common.singleton import Singleton


def test_parse_intrusion_failed_password() -> None:
    line = "Nov 21 12:33:45 host sshd[1234]: Failed password for root from 10.0.0.5 port 22 ssh2"
    finding = rd.parse_intrusion_line(line)
    assert finding is not None
    assert finding.kind == "intrusion_auth"
    assert finding.color == "red"


def test_parse_intrusion_invalid_user() -> None:
    line = "sshd[5]: invalid user admin from 192.168.1.50"
    finding = rd.parse_intrusion_line(line)
    assert finding is not None
    assert finding.kind == "intrusion_auth"
    assert finding.color == "red"


def test_parse_intrusion_portscan() -> None:
    line = "kernel: TCP: drop open request from 8.8.8.8/4444"
    finding = rd.parse_intrusion_line(line)
    assert finding is not None
    assert finding.kind == "intrusion_portscan"


def test_parse_intrusion_usb() -> None:
    line = "usb 1-1: new full-speed USB device number 7 using xhci_hcd"
    finding = rd.parse_intrusion_line(line)
    assert finding is not None
    assert finding.kind == "intrusion_usb"
    # USB events are a warning, not an alert.
    assert finding.severity == "warn"


def test_parse_intrusion_benign_line_is_none() -> None:
    line = "systemd[1]: Started timer.service."
    assert rd.parse_intrusion_line(line) is None


def test_parse_tshark_wep_beacon() -> None:
    line = (
        "1.123 1 aa:bb:cc:dd:ee:ff -> ff:ff:ff:ff:ff:ff 802.11 220 Beacon frame, "
        "SN=0, FN=0, Flags=........C, BI=100, SSID: HOME_NET, Privacy: WEP"
    )
    finding = rd.parse_tshark_line(line)
    assert finding is not None
    assert finding.kind == "wifi_wep"
    assert finding.color == "yellow"
    assert "HOME_NET" in finding.detail


def test_parse_tshark_handshake_msg() -> None:
    line = "EAPOL Key (Message 3 of 4) Pairwise"
    finding = rd.parse_tshark_line(line)
    assert finding is not None
    assert finding.kind == "weak_handshake"


def test_parse_iw_scan_detects_wep() -> None:
    out = """\
BSS aa:bb:cc:dd:ee:ff (on wlan0)
\tSSID: GuestNet
\tcapability: ESS Privacy ShortSlotTime (0x0411)
\tWEP:
\t\t * Group cipher: WEP-40
"""
    findings = rd.parse_iw_scan(out)
    assert any(f.kind == "wifi_wep" for f in findings)


def test_parse_iw_scan_detects_wps() -> None:
    out = """\
BSS 11:22:33:44:55:66 (on wlan0)
\tSSID: VulnHome
\tWPS:\t * Version: 1.0
\t\t * Wi-Fi Protected Setup State: 2 (Configured)
\t\t * Locked: No
"""
    findings = rd.parse_iw_scan(out)
    assert any(f.kind == "wifi_wps" for f in findings)


def test_suggest_ufw_block_extracts_ip() -> None:
    finding = rd.ReconFinding(
        kind="intrusion_auth", summary="x",
        detail="Failed password for root from 10.0.0.5 port 22",
        color="red", severity="alert", ts=time.time(),
    )
    assert rd.suggest_ufw_block(finding) == "sudo -n ufw deny from 10.0.0.5"


def test_suggest_ufw_block_skips_loopback() -> None:
    finding = rd.ReconFinding(
        kind="intrusion_portscan", summary="x", detail="from 127.0.0.1",
        color="red", severity="alert", ts=time.time(),
    )
    assert rd.suggest_ufw_block(finding) is None


def test_suggest_ufw_block_yellow_returns_none() -> None:
    finding = rd.ReconFinding(
        kind="wifi_wps", summary="x", detail="BSSID=aa:bb:cc:dd:ee:ff",
        color="yellow", severity="warn", ts=time.time(),
    )
    assert rd.suggest_ufw_block(finding) is None


def test_should_emit_rate_limits() -> None:
    Singleton.reset(EventBus)
    Singleton.reset(rd.ReconDaemon)
    daemon = rd.ReconDaemon()
    assert daemon._should_emit("k") is True
    assert daemon._should_emit("k") is False
    # Different key bypasses cooldown.
    assert daemon._should_emit("k2") is True
