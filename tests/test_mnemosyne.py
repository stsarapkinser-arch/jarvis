"""Mnemosyne harvester tests.

We test the pure helpers exhaustively (they own most of the smarts) and
exercise the async harvest + clipboard assimilation paths with a stub
memory + LLM so we don't touch ChromaDB or Ollama from the test suite.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.memory.storage as mnemosyne

# --- pure helpers ----------------------------------------------------------
def test_parse_firefox_url_picks_url_only():
    assert mnemosyne.parse_firefox_url("https://kali.org/news — Mozilla Firefox") == "https://kali.org/news"
    assert mnemosyne.parse_firefox_url("Welcome page - Mozilla Firefox") is None
    assert mnemosyne.parse_firefox_url("https://example.com - Google Chrome") == "https://example.com"
    assert mnemosyne.parse_firefox_url("Terminal — Konsole") is None
    assert mnemosyne.parse_firefox_url("") is None


def test_parse_editor_file_supports_common_editors():
    raw = "vim\x00/etc/hosts\x00"
    assert mnemosyne.parse_editor_file(raw) == "/etc/hosts"
    raw_nvim = "nvim\x00-p\x00/etc/passwd\x00"
    assert mnemosyne.parse_editor_file(raw_nvim) == "/etc/passwd"
    raw_code = "/usr/bin/code\x00--user-data-dir=/tmp/c\x00/home/u/proj/main.py\x00"
    assert mnemosyne.parse_editor_file(raw_code) == "/home/u/proj/main.py"
    raw_ls = "ls\x00-la\x00"
    assert mnemosyne.parse_editor_file(raw_ls) is None


def test_nearest_symbol_finds_def_above_line():
    src = "\n".join([
        "import os",
        "",
        "def alpha():",
        "    pass",
        "",
        "def beta(arg):",
        "    return arg + 1",
        "",
    ])
    assert mnemosyne.nearest_symbol(src, near_line=4) == "def alpha"
    assert mnemosyne.nearest_symbol(src, near_line=7) == "def beta"
    assert mnemosyne.nearest_symbol(src, near_line=1) is None


def test_detect_assimilable_categories():
    payload = """
    Suspicious IP 10.0.0.42 plus a loopback 127.0.0.1
    See CVE-2024-1234 advisory.
    https://exploit-db.example/123
    def detonate():
        import os
        os.system("ls")
    """
    hits = mnemosyne.detect_assimilable(payload)
    assert "10.0.0.42" in hits["ip"]
    assert "127.0.0.1" not in hits["ip"]   # loopback filtered out
    assert "CVE-2024-1234" in hits["cve"]
    assert any("exploit-db.example" in u for u in hits["url"])
    assert "code" in hits


def test_detect_assimilable_empty_returns_empty():
    assert mnemosyne.detect_assimilable("") == {}
    assert mnemosyne.detect_assimilable("hello world") == {}


# --- async harvester -------------------------------------------------------
class _FakeMemory:
    def __init__(self) -> None:
        self.added: list[dict] = []
        self.recalls: list[str] = []

    def remember(self, intent, command, result, kind, snapshot=None):  # noqa: D401
        self.added.append({"intent": intent, "command": command, "result": result, "kind": kind})

    def recall(self, query, n_results=3, since_days=14):
        self.recalls.append(query)
        return [{"document": "prior note about 10.0.0.42", "metadata": {}}]


class _FakeLLM:
    def __init__(self, response: str = "Это похоже на сканер портов."):
        self.calls: list[dict] = []
        self.response = response

    async def generate(self, model, prompt):
        self.calls.append({"model": model, "prompt": prompt})
        return {"response": self.response}


def test_assimilate_clipboard_runs_full_pipeline():
    mem = _FakeMemory()
    llm = _FakeLLM("Подозрительный IP — рекомендую блок.")
    mn = mnemosyne.Mnemosyne(memory=mem, llm_client=llm)

    async def go():
        return await mn.assimilate_clipboard("Look at 10.0.0.42 carefully")

    result = asyncio.run(go())
    assert result is not None
    assert "10.0.0.42" in result["hits"]["ip"]
    assert result["thought"].startswith("Подозрительный")
    # We hit the memory recall + add paths.
    assert mem.recalls
    assert any(item["kind"] == "assimilation" for item in mem.added)
    # The LLM was invoked exactly once.
    assert len(llm.calls) == 1


def test_assimilate_clipboard_skips_uninteresting_payloads():
    mem = _FakeMemory()
    mn = mnemosyne.Mnemosyne(memory=mem, llm_client=_FakeLLM())

    async def go():
        return await mn.assimilate_clipboard("hello there")

    assert asyncio.run(go()) is None
    assert not mem.added


def test_assimilate_clipboard_respects_cooldown():
    mem = _FakeMemory()
    mn = mnemosyne.Mnemosyne(memory=mem, llm_client=_FakeLLM())

    async def go():
        first = await mn.assimilate_clipboard("CVE-2024-9999 is hot")
        second = await mn.assimilate_clipboard("CVE-2024-9999 is hot")
        return first, second

    first, second = asyncio.run(go())
    assert first is not None
    assert second is None    # cooldown blocks the repeat


def test_capture_slice_works_without_kwin(tmp_path: Path):
    """When kwin is absent the slice should still contain editor / history."""
    mem = _FakeMemory()
    mn = mnemosyne.Mnemosyne(memory=mem, kwin=None, llm_client=_FakeLLM())

    # Plant a history file so the harvester pulls something.
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_history").write_text("ls\nps auxf\nnmap -sV 10.0.0.0/24\n", encoding="utf-8")

    with patch.object(mnemosyne.Path, "home", classmethod(lambda cls: home)):
        slice_ = asyncio.run(mn._capture_slice())
    assert slice_.last_shell == "nmap -sV 10.0.0.0/24"
    assert slice_.window == ""
    assert isinstance(slice_.to_document(), str)


def test_reality_slice_document_skips_blanks():
    slice_ = mnemosyne.RealitySlice(window="konsole", last_shell="whoami")
    doc = slice_.to_document()
    assert "WINDOW: konsole" in doc
    assert "URL" not in doc
    assert "SHELL: whoami" in doc


# --- resource fixes: clipboard provider cache + dedup cap ------------------
from types import SimpleNamespace  # noqa: E402

from src.common.singleton import Singleton  # noqa: E402


def test_read_local_clipboard_caches_provider_and_skips_dead_binaries(monkeypatch):
    """Опрос буфера форкает один subprocess, а не перебирает все три каждые 2с:
    провайдер определяется один раз и кэшируется."""
    Singleton.reset(mnemosyne.Mnemosyne)
    mn = mnemosyne.Mnemosyne()
    calls: list[str] = []

    def fake_run(cmd, **_kw):
        calls.append(cmd[0])
        if cmd[0] == "qdbus":          # klipper отсутствует → rc!=0, не кэшируем
            return SimpleNamespace(returncode=1, stdout="")
        if cmd[0] == "wl-paste":       # рабочий провайдер
            return SimpleNamespace(returncode=0, stdout="clip-text")
        return SimpleNamespace(returncode=0, stdout="")

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert mn._read_local_clipboard() == "clip-text"
    assert mn._clip_cmd is not None and mn._clip_cmd[0] == "wl-paste"
    # Второй опрос зовёт ТОЛЬКО кэшированный провайдер — qdbus больше не форкается.
    calls.clear()
    assert mn._read_local_clipboard() == "clip-text"
    assert calls == ["wl-paste"]


def test_read_local_clipboard_reprobes_when_cached_binary_vanishes(monkeypatch):
    """Если кэшированный бинарь исчез (FileNotFoundError) — кэш сбрасывается и
    провайдер зондируется заново, функциональность не теряется."""
    Singleton.reset(mnemosyne.Mnemosyne)
    mn = mnemosyne.Mnemosyne()
    mn._clip_cmd = ["wl-paste", "--no-newline"]

    def fake_run(cmd, **_kw):
        if cmd[0] == "wl-paste":
            raise FileNotFoundError("wl-paste removed")
        if cmd[0] == "qdbus":
            return SimpleNamespace(returncode=0, stdout="from-klipper")
        return SimpleNamespace(returncode=1, stdout="")

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Кэш (wl-paste) бросает FileNotFoundError → сброс кэша и немедленное
    # перезондирование в том же вызове находит работающий qdbus (без потери цикла).
    assert mn._read_local_clipboard() == "from-klipper"
    assert mn._clip_cmd is not None and mn._clip_cmd[0] == "qdbus"


def test_assimilate_dedup_dict_is_capped():
    """``_last_assim`` не растёт всю сессию — при переполнении дропаются старые."""
    Singleton.reset(mnemosyne.Mnemosyne)
    mn = mnemosyne.Mnemosyne(memory=None, llm_client=None)

    async def go():
        # Уникальные интересные payload'ы (разные IP) → разные ключи, без cooldown.
        for i in range(mnemosyne.ASSIM_DEDUP_MAX + 50):
            await mn.assimilate_clipboard(f"adres 10.{i // 256}.{i % 256}.7")

    asyncio.run(go())
    assert len(mn._last_assim) <= mnemosyne.ASSIM_DEDUP_MAX
