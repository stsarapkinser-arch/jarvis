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
import time
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


@skill(id="report_battery", category=_SYS, speaks_result=True,
       description="доложить заряд аккумулятора",
       aliases=("батарея", "заряд", "сколько заряда"))
async def report_battery(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(
        "cat /sys/class/power_supply/BAT0/capacity 2>/dev/null "
        "|| cat /sys/class/power_supply/BAT1/capacity 2>/dev/null"
    )
    pct = out.strip()
    if rc != 0 or not pct.isdigit():
        return "Аккумулятор не обнаружен — вероятно, питание от сети."
    return f"Заряд аккумулятора {pct} процентов."


@skill(id="report_load", category=_SYS, speaks_result=True,
       description="доложить среднюю нагрузку (load average)",
       aliases=("средняя нагрузка", "load average", "лоад"))
async def report_load(ctx: SkillContext, args: dict[str, Any]) -> str:
    try:
        with open("/proc/loadavg", encoding="ascii") as f:
            one, five, fifteen = f.read().split()[:3]
    except OSError:
        return "Не удалось прочитать нагрузку."
    return f"Средняя нагрузка: {one} за минуту, {five} за пять, {fifteen} за пятнадцать."


@skill(id="report_ip_address", category=_SYS, speaks_result=True,
       description="доложить локальный IP-адрес",
       aliases=("мой айпи", "ip адрес", "какой у меня ip"))
async def report_ip_address(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(
        "ip -4 -br addr show scope global | awk '{print $1\": \"$3}'"
    )
    text = out.strip()
    if rc != 0 or not text:
        return "Активного сетевого адреса не найдено."
    first = text.splitlines()[0]
    return f"Локальный адрес: {first}."


@skill(id="report_network_status", category=_SYS, speaks_result=True,
       description="проверить доступ в интернет",
       aliases=("есть ли интернет", "проверь сеть", "интернет работает"))
async def report_network_status(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run("ping -c 1 -W 2 1.1.1.1")
    return "Интернет доступен." if rc == 0 else "Интернета нет — пинг не прошёл."


@skill(id="report_kernel", category=_SYS, speaks_result=True,
       description="доложить версию ядра",
       aliases=("версия ядра", "ядро", "uname"))
async def report_kernel(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run("uname -r")
    return f"Ядро Linux {out.strip()}." if rc == 0 and out.strip() else "Не удалось прочитать версию ядра."


@skill(id="report_distro", category=_SYS, speaks_result=True,
       description="доложить название дистрибутива",
       aliases=("какая система", "дистрибутив", "версия системы"))
async def report_distro(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(". /etc/os-release 2>/dev/null && echo \"$PRETTY_NAME\"")
    name = out.strip()
    return f"Система: {name}." if rc == 0 and name else "Не удалось определить дистрибутив."


@skill(id="report_datetime", category=_SYS, speaks_result=True,
       description="доложить текущие дату и время",
       aliases=("который час", "сколько времени", "какое сегодня число", "дата"))
async def report_datetime(ctx: SkillContext, args: dict[str, Any]) -> str:
    now = time.localtime()
    return time.strftime("Сейчас %H:%M, сегодня %d.%m.%Y.", now)


@skill(id="report_logged_users", category=_SYS, speaks_result=True,
       description="доложить, кто сейчас в системе",
       aliases=("кто залогинен", "кто в системе", "активные пользователи"))
async def report_logged_users(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run("who | awk '{print $1}' | sort -u | tr '\\n' ' '")
    users = out.strip()
    if rc != 0 or not users:
        return "Активных пользовательских сессий не вижу."
    return f"В системе: {users}."


@skill(id="report_running_services", category=_SYS, speaks_result=True,
       description="доложить число запущенных служб",
       aliases=("сколько служб", "запущенные сервисы", "активные службы"))
async def report_running_services(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(
        "systemctl list-units --type=service --state=running --no-legend --no-pager | wc -l"
    )
    count = out.strip()
    if rc != 0 or not count.isdigit():
        return "Не удалось получить список служб."
    return f"Сейчас запущено служб: {count}."


@skill(id="report_top_memory", category=_SYS, speaks_result=True,
       description="доложить процессы, сильнее всего занимающие память",
       aliases=("что ест память", "топ по памяти", "память процессов"))
async def report_top_memory(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, out, _ = await ctx.run(
        "ps -eo comm,pmem --sort=-pmem --no-headers | head -3"
    )
    if rc != 0 or not out.strip():
        return "Не удалось получить список процессов."
    names = []
    for line in out.strip().splitlines():
        cols = line.split()
        if len(cols) >= 2:
            names.append(f"{cols[0]} {cols[1]} процентов")
    return "Больше всего памяти у: " + ", ".join(names) + "." if names else "Нет данных по памяти."
