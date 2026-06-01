"""Кодовая фраза подтверждения разрушительного: точный матч целиком.

Гейт на rm -rf/sudo/destructive-навык снимается ТОЛЬКО кодовой фразой как
самостоятельной репликой — не подстрокой и не общим «да». Это защита от
Vosk-дрейфа (WER до ~32%): случайное «давай» в фоне не должно ничего спустить.
"""
from __future__ import annotations

from src.core.orchestrator import is_confirm_phrase


def test_exact_codephrase_confirms():
    assert is_confirm_phrase("джарвис подтверждаю") is True


def test_normalization_punctuation_case_yo():
    # Регистр, запятая, ё→е, лишние пробелы — нормализуются.
    assert is_confirm_phrase("  Джарвис, подтверждаю!  ") is True
    assert is_confirm_phrase("ДЖАРВИС ПОДТВЕРЖДАЮ") is True


def test_word_order_swap_allowed():
    assert is_confirm_phrase("подтверждаю джарвис") is True


def test_bare_yes_does_not_confirm():
    for s in ("да", "да, давай", "конечно", "ок", "выполни", "давай уже"):
        assert is_confirm_phrase(s) is False, s


def test_codephrase_as_substring_does_not_confirm():
    # Фраза-подстрока внутри длинной реплики НЕ считается подтверждением.
    assert is_confirm_phrase("ну джарвис подтверждаю наверное потом") is False


def test_empty_is_false():
    assert is_confirm_phrase("") is False
    assert is_confirm_phrase("   ") is False
