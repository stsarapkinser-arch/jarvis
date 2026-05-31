"""SYSTEM_OPS навыки — диагностика и безопасные операции.

Телеметрические навыки помечены ``speaks_result=True``: живые цифры знает
только хендлер (читает SystemState / shutil), поэтому озвучиваем СТРОКУ
хендлера, а не выдуманный моделью ``reply``.

Никаких разрушительных действий в стартовом наборе — только чтение состояния
и безопасные сводки. Длинный хвост (apt/systemctl/rm …) идёт через
gated-fallback ``execute_bash``, а не через эти навыки.
"""
from __future__ import annotations

import shutil
from typing import Any

from src.common.event_bus import SystemState
from src.inference.router import IntentCategory
from src.skills.registry import SkillContext, skill

_SYS = IntentCategory.SYSTEM_OPS


@skill(id="report_cpu", category=_SYS, speaks_result=True,
       description="доложить загрузку процессора и температуру",
       aliases=("нагрузка процессора", "температура", "загрузка цп"))
async def report_cpu(ctx: SkillContext, args: dict[str, Any]) -> str:
    snap = SystemState().snapshot()
    parts = [f"Процессор загружен на {snap.cpu:.0f} процентов"]
    if snap.thermal:
        parts.append(f"температура {snap.thermal:.0f} градусов")
    if snap.gpu:
        parts.append(f"видеоядро {snap.gpu:.0f} процентов")
    return ", ".join(parts) + "."


@skill(id="report_memory", category=_SYS, speaks_result=True,
       description="доложить использование оперативной памяти",
       aliases=("память", "сколько памяти", "озу"))
async def report_memory(ctx: SkillContext, args: dict[str, Any]) -> str:
    snap = SystemState().snapshot()
    return f"Оперативная память занята на {snap.ram:.0f} процентов."


@skill(id="report_disk", category=_SYS, speaks_result=True,
       description="доложить свободное место на диске",
       aliases=("диск", "сколько места", "место на диске"))
async def report_disk(ctx: SkillContext, args: dict[str, Any]) -> str:
    try:
        usage = shutil.disk_usage("/")
    except OSError:
        return "Не удалось прочитать состояние диска."
    free_gb = usage.free / (1024 ** 3)
    pct = usage.used / usage.total * 100 if usage.total else 0.0
    return f"Диск заполнен на {pct:.0f} процентов, свободно {free_gb:.0f} гигабайт."


@skill(id="report_uptime", category=_SYS, speaks_result=True,
       description="доложить время работы системы",
       aliases=("аптайм", "сколько работает", "время работы"))
async def report_uptime(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run("uptime -p")
    text = out.strip()
    if rc == 0 and text:
        return f"Система работает: {text}."
    return "Не удалось получить время работы."


@skill(id="list_top_processes", category=_SYS, speaks_result=True,
       description="доложить процессы, сильнее всего грузящие процессор",
       aliases=("топ процессов", "что грузит", "тяжёлые процессы"))
async def list_top_processes(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(
        "ps -eo comm,pcpu --sort=-pcpu --no-headers | head -3"
    )
    if rc != 0 or not out.strip():
        return "Не удалось получить список процессов."
    names = []
    for line in out.strip().splitlines():
        cols = line.split()
        if len(cols) >= 2:
            names.append(f"{cols[0]} {cols[1]} процентов")
    if not names:
        return "Активных тяжёлых процессов нет."
    return "Сильнее всего грузят: " + ", ".join(names) + "."
