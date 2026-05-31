"""Тесты SkillLearner — наблюдателя, превращающего привычку в навык."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.skills import learned
from src.skills.learning import Candidate, SkillLearner, _canonical, _slug


def _learner(tmp_path, threshold=3) -> SkillLearner:
    return SkillLearner(usage_path=tmp_path / "usage.json", threshold=threshold)


def test_canonical_collapses_whitespace():
    assert _canonical("  ip   a  ") == "ip a"


def test_slug_transliterates_cyrillic():
    s = _slug("Покажи адреса")
    assert s and all(c.isascii() for c in s)
    assert " " not in s


def test_observe_proposes_after_threshold(tmp_path):
    learner = _learner(tmp_path, threshold=3)
    assert learner.observe("покажи адреса", "ip -br a", destructive=False) is None
    assert learner.observe("покажи адреса", "ip -br a", destructive=False) is None
    cand = learner.observe("покажи адреса", "ip -br a", destructive=False)
    assert isinstance(cand, Candidate)
    assert cand.command == "ip -br a"
    assert cand.skill_id.startswith(learned.LEARNED_PREFIX)
    assert cand.count == 3


def test_observe_proposes_only_once(tmp_path):
    learner = _learner(tmp_path, threshold=2)
    learner.observe("t", "echo hi", destructive=False)
    assert learner.observe("t", "echo hi", destructive=False) is not None
    # После предложения — молчим, даже если команда снова повторяется.
    assert learner.observe("t", "echo hi", destructive=False) is None


def test_destructive_never_learned(tmp_path):
    learner = _learner(tmp_path, threshold=1)
    assert learner.observe("снеси", "rm -rf /tmp/x", destructive=True) is None


def test_sudo_not_learned(tmp_path):
    learner = _learner(tmp_path, threshold=1)
    assert learner.observe("ставь", "sudo -n apt update", destructive=False) is None


def test_counts_persist_across_instances(tmp_path):
    path = tmp_path / "usage.json"
    a = SkillLearner(usage_path=path, threshold=3)
    a.observe("t", "uptime -p", destructive=False)
    a.observe("t", "uptime -p", destructive=False)
    # Новый инстанс (как после os.execv-перезагрузки) видит накопленное.
    b = SkillLearner(usage_path=path, threshold=3)
    cand = b.observe("t", "uptime -p", destructive=False)
    assert cand is not None and cand.count == 3


def test_promote_writes_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(learned, "LEARNED_PATH", tmp_path / "learned_skills.json")
    monkeypatch.setattr(learned, "_CONFIG_DIR", tmp_path)
    learner = _learner(tmp_path, threshold=1)
    cand = learner.observe("покажи аптайм", "uptime -p", destructive=False)
    assert cand is not None
    assert learner.promote(cand) is True
    data = json.loads((tmp_path / "learned_skills.json").read_text(encoding="utf-8"))
    assert data[0]["command"] == "uptime -p"
    assert data[0]["id"] == cand.skill_id
    # После промоушена повторное наблюдение не предлагает снова.
    assert learner.observe("покажи аптайм", "uptime -p", destructive=False) is None
