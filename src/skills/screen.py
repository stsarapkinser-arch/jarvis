"""SCREEN навыки — «глаза» Джарвиса на рабочем столе.

Дешёвый, но трансформирующий level-up: до сих пор Джарвис был слеп к тому, что
на мониторе. Полноценный VLM на N100 неподъёмен, поэтому идём дешёвым путём —
**скриншот → OCR (tesseract)** + заголовок активного окна. Этого хватает для
«что это за ошибка?», «прочитай экран», «какое окно активно».

Конвенции — те же, что в ui_control: всё через ``ctx.run`` (ждём результат),
OR-цепочки бинарей ради переносимости между Wayland (grim/spectacle) и X11
(scrot/maim). При отсутствии инструментов навык деградирует в честную фразу, а
не падает.

Навыки помечены ``speaks_result=True``: содержимое экрана знает только хендлер
(прочитал его через OCR), поэтому озвучиваем СТРОКУ хендлера, а не выдуманный
моделью ``reply``.

# VERIFY on target: набор скриншот-утилит зависит от сессии оператора. Цепочка
# ниже покрывает Wayland (grim, spectacle) и X11 (scrot, maim, import); если у
# оператора стоит что-то одно — лишние ветки просто не сработают (|| идёт дальше).
"""
from __future__ import annotations

import re
from typing import Any

from src.inference.router import IntentCategory
from src.skills.registry import SkillContext, skill

_UI = IntentCategory.UI_CONTROL

# Сколько символов распознанного текста максимум отдаём в речь. Экран — это
# простыня; для голоса берём осмысленный кусок, полный текст уходит в память.
_OCR_SPEAK_LIMIT = 320

# Скриншот всего экрана во временный PNG, затем OCR. Цепочка утилит покрывает
# Wayland и X11; один из вариантов да сработает на машине оператора.
_GRAB_CHAIN = (
    "grim \"$F\" "
    "|| spectacle -b -n -o \"$F\" "
    "|| scrot -o \"$F\" "
    "|| maim \"$F\" "
    "|| import -window root \"$F\""
)


def _ocr_command(region: str = "") -> str:
    """Собрать одну shell-команду: захват экрана → tesseract (rus+eng) → cleanup.

    ``region`` (если задан) подменяет цепочку захвата на захват области (geometry
    в формате grim/slurp ``WxH+X+Y``), иначе берётся весь экран."""
    grab = _GRAB_CHAIN
    if region:
        # Геометрия уже провалидирована вызывающим — здесь только подстановка.
        grab = f"grim -g \"{region}\" \"$F\" || import -window root \"$F\""
    return (
        'F="$(mktemp --suffix=.png)"; '
        f"{{ {grab}; }} >/dev/null 2>&1; "
        # tesseract на stdout (вывод "-"); rus+eng → откат на eng, если нет языка.
        'tesseract "$F" - -l rus+eng 2>/dev/null || tesseract "$F" - 2>/dev/null; '
        'rm -f "$F"'
    )


def _clean_ocr(text: str) -> str:
    """Схлопнуть OCR-шум: пустые строки, повторные пробелы, обрезки-артефакты."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    joined = " ".join(lines)
    return re.sub(r"\s{2,}", " ", joined).strip()


@skill(id="read_screen", category=_UI, speaks_result=True,
       description="прочитать текст с экрана (скриншот + OCR)",
       aliases=("что на экране", "прочитай экран", "распознай экран",
                "что написано", "прочитай что на экране"))
async def read_screen(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(_ocr_command())
    text = _clean_ocr(out)
    if rc != 0 and not text:
        return ("Не удалось прочитать экран, сэр — нет утилиты скриншота или "
                "tesseract. Установите grim и tesseract.")
    if not text:
        return "Экран распознан, но читаемого текста на нём нет."
    if len(text) > _OCR_SPEAK_LIMIT:
        head = text[:_OCR_SPEAK_LIMIT].rsplit(" ", 1)[0]
        return f"На экране, сэр: {head}…"
    return f"На экране, сэр: {text}"


@skill(id="read_active_window", category=_UI, speaks_result=True,
       description="прочитать заголовок активного окна",
       aliases=("какое окно", "что за окно", "активное окно", "что открыто"))
async def read_active_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    # Заголовок активного окна: kdotool (KWin/Wayland) → xdotool (X11).
    rc, out, _ = await ctx.run(
        "kdotool getactivewindow getwindowname 2>/dev/null "
        "|| xdotool getactivewindow getwindowname 2>/dev/null"
    )
    title = out.strip().splitlines()[0].strip() if out.strip() else ""
    if rc != 0 and not title:
        return "Не удалось определить активное окно, сэр."
    if not title:
        return "Активное окно без заголовка."
    return f"Активно окно: {title}."


@skill(id="describe_error_on_screen", category=_UI, speaks_result=True,
       description="найти и зачитать сообщение об ошибке на экране",
       aliases=("что за ошибка", "прочитай ошибку", "какая ошибка на экране",
                "что за ошибка на экране"))
async def describe_error_on_screen(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(_ocr_command())
    text = _clean_ocr(out)
    if rc != 0 and not text:
        return ("Не удалось прочитать экран, сэр — нет утилиты скриншота или "
                "tesseract.")
    if not text:
        return "Текста на экране не вижу, сэр."
    # Выцепляем строки, похожие на ошибку (рус/eng маркеры).
    err_re = re.compile(
        r"(error|exception|traceback|failed|fatal|denied|not found|cannot|"
        r"ошибк\w*|сбой|отказ\w*|не удал\w*|не найден\w*)",
        re.IGNORECASE,
    )
    # OCR уже схлопнут в одну строку — режем по предложениям для поиска маркеров.
    fragments = re.split(r"(?<=[.!?:;])\s+", text)
    hits = [f for f in fragments if err_re.search(f)]
    if not hits:
        return "Явных сообщений об ошибке на экране не вижу, сэр."
    joined = " ".join(hits)[:_OCR_SPEAK_LIMIT]
    return f"Похоже на ошибку, сэр: {joined}"
