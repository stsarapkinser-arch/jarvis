"""Tests for the nmap parser used by the Port Matrix overlay."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.network.scanneras ns


def test_is_nmap_command_basic() -> None:
    assert ns.is_nmap_command("nmap -sS 10.0.0.1")
    assert ns.is_nmap_command("sudo nmap -A target.lan")
    assert ns.is_nmap_command("sudo -n nmap -p- 192.168.1.1")
    assert not ns.is_nmap_command("ls -la")
    assert not ns.is_nmap_command("masscan -p80 0.0.0.0/0")
    assert not ns.is_nmap_command("")


def test_parse_host_line() -> None:
    kind, payload = ns.parse_nmap_line("Nmap scan report for example.com (93.184.216.34)", "")
    assert kind == "host"
    assert payload == {"host": "example.com", "ip": "93.184.216.34", "ts": payload["ts"]}


def test_parse_host_line_no_ip() -> None:
    kind, payload = ns.parse_nmap_line("Nmap scan report for router.lan", "")
    assert kind == "host"
    assert payload["host"] == "router.lan"
    assert payload["ip"] is None


def test_parse_port_open() -> None:
    kind, payload = ns.parse_nmap_line("22/tcp   open   ssh", "router.lan")
    assert kind == "port"
    assert payload["port"] == 22
    assert payload["proto"] == "tcp"
    assert payload["state"] == "open"
    assert payload["service"] == "ssh"
    assert payload["host"] == "router.lan"


def test_parse_port_filtered() -> None:
    kind, payload = ns.parse_nmap_line("443/tcp  filtered  https", "10.0.0.1")
    assert kind == "port"
    assert payload["state"] == "filtered"


def test_parse_progress() -> None:
    kind, payload = ns.parse_nmap_line(
        "Stats: 0:00:10 elapsed; About 42.50% done; ETC: 12:30 (0:00:14 remaining)", ""
    )
    assert kind == "progress"
    assert payload["percent"] == 42.5


def test_parse_blank_line_returns_none() -> None:
    kind, _ = ns.parse_nmap_line("", "x")
    assert kind is None
    kind, _ = ns.parse_nmap_line("\n", "x")
    assert kind is None
