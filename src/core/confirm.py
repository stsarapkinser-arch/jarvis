"""Логика голосового подтверждения — отдельный модуль, чтобы её могли разделять
и оркестратор, и mixin лестницы распознавания без циклического импорта.

Главная идея: гейт разрушительного снимается НЕ общим «да/ок/давай», а отдельной
КОДОВОЙ ФРАЗОЙ как самостоятельной репликой (точный матч целиком). Vosk small
ошибается до ~32% WER (см. model/README), и случайное «давай» в фоне или
галлюцинация распознавателя не должны спускать необратимую команду на хост.
Отмена (NEGATIVE_RE) остаётся широкой — ложная отмена лишь прерывает, это
безопасно. AFFIRMATIVE_RE остаётся для НЕ-опасных подтверждений (промоушен
выученного навыка), где цена ошибки нулевая.
"""
from __future__ import annotations

import os
import re

AFFIRMATIVE_RE = re.compile(
    r"\b(да|конечно|подтверждаю|подтвердить|выполни(?:ть)?|давай|ок|окей|yes|confirm|do\s+it|go\s+ahead)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(нет|отмени(?:ть)?|стоп|остановить|cancel|no|stop|abort|don'?t)\b",
    re.IGNORECASE,
)

# Кодовую фразу можно сменить через JARVIS_CONFIRM_PHRASE.
CONFIRM_PHRASE = os.getenv("JARVIS_CONFIRM_PHRASE", "джарвис подтверждаю").strip()


def _normalize_confirm(text: str) -> str:
    """Нормализация реплики под точный матч: нижний регистр, ё→е, пунктуация →
    пробел, схлопнутые пробелы по краям и внутри."""
    s = (text or "").lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def is_confirm_phrase(text: str) -> bool:
    """True, только если реплика ЦЕЛИКОМ совпадает с кодовой фразой (с точностью
    до нормализации и перестановки слов двухсловной фразы). Подстрока внутри
    длинной реплики подтверждением НЕ считается — это и есть защита от STT-дрейфа."""
    norm = _normalize_confirm(text)
    if not norm:
        return False
    target = _normalize_confirm(CONFIRM_PHRASE)
    if norm == target:
        return True
    parts = target.split()
    if len(parts) == 2 and norm == f"{parts[1]} {parts[0]}":
        return True
    return False


# Подсказка оператору, что именно произнести. Собирается из живой кодовой фразы,
# чтобы переопределение JARVIS_CONFIRM_PHRASE отражалось в речи автоматически.
CONFIRM_HINT = f"Скажите «{CONFIRM_PHRASE}» для подтверждения или «отмени»."


__all__ = [
    "AFFIRMATIVE_RE",
    "NEGATIVE_RE",
    "CONFIRM_PHRASE",
    "CONFIRM_HINT",
    "is_confirm_phrase",
]
