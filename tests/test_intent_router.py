"""Тесты семантического маршрутизатора (Phase 2).

Проверяем: классификацию по 4 категориям, детерминированные правила для
служебных интентов, лимит микро-промптов (≤100 слов), стабильный core-префикс
(критично для KV-cache reuse) и устойчивость к мусору."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.router import IntentCategory, IntentRouter


def test_route_system_ops():
    r = IntentRouter()
    for phrase in (
        "установи пакет nginx",
        "покажи запущенные процессы",
        "сколько свободно на диске",
        "убей зависший процесс firefox",
        "обнови систему через apt",
    ):
        assert r.route(phrase).category == IntentCategory.SYSTEM_OPS, phrase


def test_route_ui_control():
    r = IntentRouter()
    for phrase in (
        "сделай ярче экран",
        "сверни все окна",
        "убавь громкость",
        "переключи рабочий стол",
        "сделай скриншот",
    ):
        assert r.route(phrase).category == IntentCategory.UI_CONTROL, phrase


def test_route_pentest_recon():
    r = IntentRouter()
    for phrase in (
        "просканируй сеть nmap",
        "перехвати трафик в wireshark",
        "проверь открытые порты на хосте",
        "поищи уязвимости cve",
        "запусти разведку по подсети",
    ):
        assert r.route(phrase).category == IntentCategory.PENTEST_RECON, phrase


def test_route_conversation_default():
    r = IntentRouter()
    for phrase in (
        "расскажи анекдот",
        "как дела",
        "что ты думаешь о свободе воли",
        "кто ты такой",
        "привет",
    ):
        assert r.route(phrase).category == IntentCategory.CONVERSATION, phrase


def test_empty_is_conversation():
    assert IntentRouter().route("").category == IntentCategory.CONVERSATION
    assert IntentRouter().route("   \n ").category == IntentCategory.CONVERSATION


def test_system_event_rules_route_deterministically():
    r = IntentRouter()
    assert r.route("[SYSTEM_EVENT: thermal=92C critical]").category == IntentCategory.SYSTEM_OPS
    assert r.route("[DAEMON_ALERT] disk almost full").category == IntentCategory.SYSTEM_OPS
    assert r.route("[OS_EVENT] memory pressure").category == IntentCategory.SYSTEM_OPS
    assert r.route("[RECON_ALERT/red] intrusion detected").category == IntentCategory.PENTEST_RECON


def test_micro_prompts_under_word_limit():
    """ТЗ: микро-промпт ≤100 слов. Считаем токены, содержащие буквы/цифры."""
    r = IntentRouter()
    for cat in IntentCategory:
        prompt = r.system_prompt_for(cat)
        words = [w for w in prompt.split() if any(ch.isalnum() for ch in w)]
        assert len(words) <= 100, f"{cat.value}: {len(words)} слов (>100)"


def test_stable_core_prefix_for_kv_cache():
    """KV-cache reuse требует байт-идентичного префикса для всех категорий."""
    r = IntentRouter(core_identity="STABLE_CORE_IDENTITY")
    prompts = [r.system_prompt_for(c) for c in IntentCategory]
    assert all(p.startswith("STABLE_CORE_IDENTITY") for p in prompts)


def test_route_decision_carries_metadata():
    d = IntentRouter().route("просканируй порты nmap")
    assert d.category == IntentCategory.PENTEST_RECON
    assert d.backend in ("regex", "rule")
    assert d.score > 0


def test_route_never_raises_on_junk():
    r = IntentRouter()
    for junk in ("!!!", "123 456 789", "\n\t\r", "ＵＮＩＣＯＤＥ", "a" * 5000, "—"):
        r.route(junk)  # must not raise


def test_embedding_backend_without_embedder_falls_back_to_regex():
    r = IntentRouter(backend="embedding", embedder=None)
    assert r.backend == "regex"
    assert r.route("установи пакет").category == IntentCategory.SYSTEM_OPS


def test_action_intents_get_a_category_with_execute_bash():
    """Регресс: «открой настройки» раньше падал в CONVERSATION (tools без
    execute_bash) — модель не могла действовать и генерила прозу до таймаута
    клиента. Команды-действия обязаны попадать в категорию с execute_bash."""
    from src.inference.tools import tools_for_category

    r = IntentRouter()
    for phrase in (
        "открой настройки",
        "открой приложение",
        "запусти приложение",
        "открой меню",
    ):
        d = r.route(phrase)
        assert d.category != IntentCategory.CONVERSATION, f"{phrase!r} → {d.category}"
        names = {s["function"]["name"] for s in tools_for_category(d.category)}
        assert "execute_bash" in names, f"{phrase!r} → {d.category} без execute_bash"


def test_unknown_command_falls_back_to_action_not_chat():
    """ГЛОБАЛЬНЫЙ фикс (не костыль из слов): неузнанная КОМАНДА без ключевого
    слова идёт в action-категорию с execute_bash, а не в CONVERSATION (где модель
    была бы разоружена и генерила прозу до таймаута)."""
    from src.inference.tools import tools_for_category

    r = IntentRouter()
    for phrase in ("почини интернет", "перезагрузи роутер", "разбуди систему"):
        d = r.route(phrase)
        assert d.backend == "fallback_command", f"{phrase!r} → {d.backend}"
        assert d.category != IntentCategory.CONVERSATION
        names = {s["function"]["name"] for s in tools_for_category(d.category)}
        assert "execute_bash" in names, f"{phrase!r} → {d.category} без execute_bash"


def test_unknown_question_falls_back_to_conversation():
    """Обратная сторона фикса: вопросительная ФОРМА (без доменных слов) → чат."""
    r = IntentRouter()
    for phrase in (
        "в чём смысл жизни",
        "сколько звёзд на небе",
        "что важнее свобода или порядок",
    ):
        assert r.route(phrase).category == IntentCategory.CONVERSATION, phrase
