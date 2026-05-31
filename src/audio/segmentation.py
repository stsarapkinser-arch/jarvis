"""Сегментация речи на предложения — для потоковой подачи в TTS.

Стриминг TTS (Tier0 #2): длинную реплику режем на предложения и подаём в
очередь озвучивания по одному, чтобы КОРОТКОЕ первое предложение начало
синтезироваться/звучать раньше, чем целая реплика (piper-латентность до первого
PCM растёт с длиной текста). Чистая функция без зависимостей — тестируема в
отрыве от аудио-стека.

Сегментатор консервативен (лучше не разбить, чем разбить криво посреди мысли):
  * границы — . ! ? … и их сочетания (?!, !..), с переносом закрывающих кавычек/
    скобок в конец предложения;
  * НЕ режем на распространённых сокращениях (т.д., т.п., т.е., др., см., рис.)
    и на инициалах/одиночных заглавных перед точкой;
  * НЕ режем десятичные числа и нумерацию версий (3.14, v1.2) — точка между
    цифрами не граница;
  * многоточие … (или ...) — это пауза-просодия Джарвиса, НЕ конец фразы:
    границей считаем только если за ним пробел и заглавная/конец строки.
"""
from __future__ import annotations

import re

# Сокращения, после которых точка НЕ конец предложения (нормализуем ё→е, lower).
_ABBREV: frozenset[str] = frozenset({
    "т", "т.д", "т.п", "т.е", "т.к", "др", "пр", "см", "рис", "табл", "стр",
    "г", "гг", "в", "вв", "н.э", "руб", "коп", "млн", "млрд", "тыс",
    "проф", "доц", "акад", "ул", "д", "корп", "кв",
})

# Кандидат-граница: один+ терминатор (. ! ? …), затем закрывающие кавычки/скобки,
# затем пробел(ы). Многоточие из точек (...) сворачиваем в … заранее.
_BOUNDARY_RE = re.compile(r'([.!?…]+)(["»”’)\]]*)(\s+)')
_TRAILING_NUM_DOT = re.compile(r"\d$")
_LEADING_NUM = re.compile(r"^\d")
_WORD_BEFORE_DOT = re.compile(r"([^\s.]*)$")


def _is_false_boundary(head: str, term: str, after: str) -> bool:
    """True, если кандидат-точка НЕ конец предложения (сокращение/число/инициал)."""
    # Десятичные/версии: цифра до точки и цифра сразу после (через пробел уже
    # граница). after здесь — текст ПОСЛЕ пробела, поэтому проверяем смежность
    # отдельно (см. вызов): тут — только когда term это ровно '.'.
    if term == ".":
        m = _WORD_BEFORE_DOT.search(head)
        word = (m.group(1) if m else "").lower().replace("ё", "е")
        if word in _ABBREV:
            return True
        # Одиночная заглавная-инициал («А.», «И.») — не граница.
        if len(word) == 1 and word.isalpha():
            return True
    return False


def split_sentences(text: str, *, min_chars: int = 0) -> list[str]:
    """Разбить текст на предложения. Пустой/однопредложный → список из одного
    элемента (или пустой). ``min_chars`` склеивает слишком короткие хвосты с
    предыдущим предложением (не плодим обрывки в очередь TTS)."""
    text = (text or "").strip()
    if not text:
        return []
    # Нормализуем многоточие из точек, чтобы не спутать с концом предложения.
    norm = re.sub(r"\.{3,}", "…", text)

    pieces: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(norm):
        term, _closing, _ws = m.group(1), m.group(2), m.group(3)
        head = norm[start:m.start()]
        after = norm[m.end():]
        # Многоточие — граница только если дальше заглавная/конец (иначе пауза).
        if term == "…" and after[:1] and not (after[:1].isupper() or after[:1].isspace()):
            continue
        # Десятичное/версия: цифра до точки и цифра сразу после терминатора.
        if _TRAILING_NUM_DOT.search(head) and _LEADING_NUM.search(after):
            continue
        if _is_false_boundary(head, term, after):
            continue
        sentence = norm[start:m.start() + len(m.group(1)) + len(m.group(2))].strip()
        if sentence:
            pieces.append(sentence)
        start = m.end()
    tail = norm[start:].strip()
    if tail:
        pieces.append(tail)

    if not pieces:
        return [norm]
    return _merge_short(pieces, min_chars) if min_chars > 0 else pieces


def _merge_short(pieces: list[str], min_chars: int) -> list[str]:
    """Склеить слишком короткие фрагменты с соседом — не плодим обрывки.

    Короткий фрагмент цепляем к предыдущему (естественнее интонационно); если
    предыдущего нет — он ждёт склейки со следующим."""
    out: list[str] = []
    for p in pieces:
        if out and len(p) < min_chars:
            out[-1] = f"{out[-1]} {p}"
        elif not out and len(p) < min_chars:
            out.append(p)  # первый и короткий — подождёт склейки со следующим
        else:
            if out and len(out[-1]) < min_chars:
                out[-1] = f"{out[-1]} {p}"
            else:
                out.append(p)
    return out


__all__ = ["split_sentences"]
