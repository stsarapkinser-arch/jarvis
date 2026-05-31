"""Тесты сегментации речи и стриминговой подачи в TTS (Tier0 #2).

Сегментатор — чистая функция: проверяем границы предложений, ловушки (сокращения,
десятичные числа, версии, инициалы) и КЛЮЧЕВОЕ для Джарвиса — что многоточие …
(его просодическая пауза) не рвёт фразу. Плюс интеграция: при JARVIS_STREAM_TTS
движок кладёт предложения в очередь по одному, иначе — прежний единичный путь.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.segmentation import split_sentences


# ───────────────────────── базовые границы ─────────────────────────
def test_empty_and_blank():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence_no_split():
    assert split_sentences("Открываю терминал, сэр.") == ["Открываю терминал, сэр."]


def test_splits_on_period_question_exclaim():
    out = split_sentences("Цель захвачена. Готов к работе? Действуйте!")
    assert out == ["Цель захвачена.", "Готов к работе?", "Действуйте!"]


def test_keeps_closing_quote_with_sentence():
    out = split_sentences('Он сказал «готово». Я проверил.')
    assert out == ['Он сказал «готово».', "Я проверил."]


# ───────────────────────── ловушки ─────────────────────────
def test_does_not_split_on_decimal_numbers():
    out = split_sentences("Загрузка 3.14 процента сейчас.")
    assert out == ["Загрузка 3.14 процента сейчас."]


def test_does_not_split_on_version():
    out = split_sentences("Версия 1.2 установлена.")
    assert out == ["Версия 1.2 установлена."]


@pytest.mark.parametrize("text", [
    "Это и т.д. и прочее в одной фразе.",
    "Смотри рис. 5 внизу страницы.",
    "Иванов И. пришёл на встречу сегодня.",
])
def test_does_not_split_on_abbreviations_and_initials(text):
    # Ровно одно предложение — сокращение/инициал не рвёт фразу.
    assert len(split_sentences(text)) == 1


# ───────────────────────── многоточие как просодия ─────────────────────────
def test_ellipsis_midphrase_is_pause_not_boundary():
    # … у Джарвиса — пауза для анализа, не конец предложения (за ним строчная).
    out = split_sentences("Цель… один девять два движется.")
    assert out == ["Цель… один девять два движется."]


def test_ellipsis_before_capital_is_boundary():
    out = split_sentences("Анализирую… Цель захвачена.")
    assert out == ["Анализирую…", "Цель захвачена."]


def test_three_dots_normalized_to_ellipsis():
    out = split_sentences("Минуту... Готово.")
    assert out == ["Минуту…", "Готово."]


# ───────────────────────── склейка коротких ─────────────────────────
def test_merges_short_fragments():
    out = split_sentences("Да. Сэр, цель захвачена и готова к работе.", min_chars=24)
    # «Да.» слишком коротко → склеено со следующим, не отдельным куском.
    assert len(out) == 1
    assert out[0].startswith("Да.")


def test_min_chars_zero_keeps_all():
    out = split_sentences("Да. Нет. Возможно.", min_chars=0)
    assert out == ["Да.", "Нет.", "Возможно."]


# ───────────────────────── интеграция с движком ─────────────────────────
class _FakeBus:
    def publish_threadsafe(self, ev):
        return None

    def publish(self, ev):
        return None


def _make_engine(monkeypatch, stream: bool):
    monkeypatch.setenv("JARVIS_STREAM_TTS", "1" if stream else "0")
    from src.audio.audio_engine import AcousticEngine
    eng = AcousticEngine(
        _FakeBus(), piper_path="/nonexistent/piper",
        voice_model="/nonexistent/model.onnx", voice_config="/nonexistent/cfg.json",
    )
    # Подменяем очередь на список-перехватчик, чтобы не гонять реальный воркер.
    captured: list = []
    eng._queue.put = lambda item: captured.append(item)  # type: ignore[assignment]
    return eng, captured


def test_streaming_enqueues_sentences_separately(monkeypatch):
    eng, captured = _make_engine(monkeypatch, stream=True)
    eng.speak("Цель захвачена. Готов к работе, сэр.", "normal")
    texts = [c[0] for c in captured if c is not None]
    assert texts == ["Цель захвачена.", "Готов к работе, сэр."]


def test_non_streaming_enqueues_single_item(monkeypatch):
    eng, captured = _make_engine(monkeypatch, stream=False)
    eng.speak("Цель захвачена. Готов к работе, сэр.", "normal")
    texts = [c[0] for c in captured if c is not None]
    assert texts == ["Цель захвачена. Готов к работе, сэр."]


def test_streaming_single_sentence_stays_single(monkeypatch):
    eng, captured = _make_engine(monkeypatch, stream=True)
    eng.speak("Открываю терминал, сэр.", "normal")
    texts = [c[0] for c in captured if c is not None]
    assert texts == ["Открываю терминал, сэр."]
