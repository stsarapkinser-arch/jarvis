"""Тесты бэкпрешера журнального наблюдателя DaemonSwarm.

Корневая поломка из лога оператора: journalctl -f -p 3 публиковал КАЖДУЮ строку
с «fail» как [DAEMON_ALERT] → on_daemon_alert → process_intent → 3B. Строки
самого Jarvis («jarvis.service: Failed…») влетали обратно → петля, затопившая и
уронившая llama-server. Проверяем три гарда: само-фильтр, дедуп, рейт-лимит.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.singleton import Singleton
from src.common.event_bus import EventBus
import src.services.daemon_swarm as swarm
from src.services.daemon_swarm import ALERT_RE, _SELF_RE, DaemonSwarm


def _fresh() -> DaemonSwarm:
    Singleton.reset(DaemonSwarm)
    Singleton.reset(EventBus)
    return DaemonSwarm()


def test_self_lines_are_filtered_out():
    """Строки о самом Jarvis/мозге не должны проходить (петля обратной связи)."""
    for line in (
        "июн 01 18:08:58 kali systemd[1210]: Failed to start jarvis.service",
        "июн 01 18:08:58 kali (python)[7823]: jarvis.service: Failed at step",
        "kali llama-server[1257]: failed to allocate",
    ):
        assert ALERT_RE.search(line), "строка должна матчить ALERT_RE"
        assert _SELF_RE.search(line), "строка о Jarvis должна ловиться _SELF_RE"


def test_external_alert_not_self_filtered():
    line = "июн 01 kali sshd[999]: Failed password invalid user root"
    assert ALERT_RE.search(line)
    assert not _SELF_RE.search(line)


def test_dedup_blocks_repeat_within_window():
    sw = _fresh()
    line = "kali sshd[1]: Failed password for invalid user admin"
    assert sw._alert_allowed(line) is True
    # Тот же смысл (другой pid/время) — дедуп по ядру строки → блок.
    assert sw._alert_allowed("kali sshd[2]: Failed password for invalid user admin") is False


def test_rate_limit_caps_burst():
    sw = _fresh()
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
    allowed = sum(
        1 for w in words
        if sw._alert_allowed(f"kali svc: failed unit {w} distinct reason {w}")
    )
    assert allowed == swarm._ALERT_MAX_PER_MIN  # потолок в минуту соблюдён


def test_dedup_key_strips_digits_and_space():
    k1 = DaemonSwarm._dedup_key("июн 01 18:08:58 kali systemd[1210]: Failed start X")
    k2 = DaemonSwarm._dedup_key("июн 02 19:09:59 kali systemd[3333]: Failed start X")
    assert k1 == k2 and "1210" not in k1
