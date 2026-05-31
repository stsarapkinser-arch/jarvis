"""Тесты Reflector — консолидации памяти в постоянные факты."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory import reflection


def test_build_prompt_includes_docs():
    p = reflection.build_prompt(["INTENT: открой chrome", "INTENT: поставь dark theme"])
    assert "chrome" in p and "dark theme" in p
    assert "NONE" in p  # инструкция про пустой случай


def test_parse_facts_extracts_labelled_lines():
    text = (
        "PREFERENCE: оператор предпочитает тёмную тему\n"
        "FACT: оператора зовут Терон\n"
        "болтовня без метки\n"
        "PROJECT: проект jarvis живёт в ~/jarvis\n"
    )
    facts = reflection.parse_facts(text)
    kinds = {f.kind for f in facts}
    assert kinds == {"preference", "identity", "project"}
    assert any("тёмную тему" in f.text for f in facts)


def test_parse_facts_handles_none_and_junk():
    assert reflection.parse_facts("NONE") == []
    assert reflection.parse_facts("") == []
    assert reflection.parse_facts("просто текст без меток") == []


def test_parse_facts_dedups_and_caps():
    text = "\n".join(["PREFERENCE: любит кофе"] * 10)
    assert len(reflection.parse_facts(text)) == 1


def test_parse_facts_skips_too_short():
    assert reflection.parse_facts("FACT: ок") == []


def test_novel_facts_filters_existing():
    facts = [
        reflection.Fact("preference", "оператор любит тёмную тему"),
        reflection.Fact("identity", "оператора зовут Терон"),
    ]
    existing = ["Оператор любит тёмную тему интерфейса"]
    fresh = reflection.novel_facts(facts, existing)
    assert len(fresh) == 1 and "Терон" in fresh[0].text


def test_consolidate_writes_novel_core_facts():
    written: list[tuple[str, str]] = []

    async def fake_llm(_prompt: str) -> str:
        return "PREFERENCE: оператор работает по ночам\nFACT: основной язык python"

    def fake_core(fact: str, kind: str):
        written.append((kind, fact))

    out = asyncio.run(reflection.consolidate(
        docs=["INTENT: что-то"], existing_core_docs=[],
        llm_complete=fake_llm, remember_core=fake_core,
    ))
    assert len(out) == 2 and len(written) == 2
    assert ("preference", "оператор работает по ночам") in written


def test_consolidate_empty_docs_noop():
    async def fake_llm(_p: str) -> str:  # pragma: no cover - не должен вызваться
        raise AssertionError("LLM не должен вызываться на пустых данных")

    out = asyncio.run(reflection.consolidate([], [], fake_llm, lambda f, k: None))
    assert out == []
