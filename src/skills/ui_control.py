"""UI_CONTROL навыки — KDE Plasma 6 / Wayland.

Здесь живут ВЫВЕРЕННЫЕ команды управления средой. Именно тут раньше болело
сильнее всего: модель угадывала синтаксис qdbus и падала. Теперь синтаксис
захардкожен и тестируется один раз на машине оператора.

Соглашения:
  * GUI-приложения (konsole/chrome/dolphin/spectacle) запускаем detached через
    ``ctx.spawn`` — не ждём, окно живёт само.
  * Короткие управляющие вызовы (wpctl/brightnessctl/qdbus6) — через ``ctx.run``
    (ждём rc, возвращаем краткий статус).
  * OR-цепочки (``a || b``) дают устойчивость к разным именам бинарей в дистрибутиве.

# VERIFY on target: строки, помеченные так, наиболее вероятны для Plasma 6, но
# DBus-интерфейсы Night Light и глобальные ярлыки KWin между версиями менялись.
# Ручная правка БОЛЬШЕ НЕ ОБЯЗАТЕЛЬНА: при падении из-за имени бинаря qdbus ядро
# само лечит команду и запоминает рабочий вариант (см. skills/healing.py и
# orchestrator._run_skill_command). Метка осталась как указатель «здесь хрупко».
"""
from __future__ import annotations

from typing import Any

from src.inference.router import IntentCategory
from src.skills.registry import SkillContext, skill

_UI = IntentCategory.UI_CONTROL


def _clamp_pct(value: Any, default: int, lo: int = 0, hi: int = 100) -> int:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ───────────────────────── Запуск приложений ─────────────────────────
@skill(id="open_terminal", category=_UI,
       description="открыть терминал (Konsole)",
       aliases=("терминал", "консоль"))
async def open_terminal(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("konsole || xterm")
    return f"terminal launched pid={pid}"


@skill(id="open_browser", category=_UI,
       description="открыть браузер (Chrome/Chromium)",
       aliases=("браузер", "хром", "chrome"))
async def open_browser(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn(
        "google-chrome-stable || google-chrome || chromium || chromium-browser || xdg-open https://"
    )
    return f"browser launched pid={pid}"


@skill(id="open_files", category=_UI,
       description="открыть файловый менеджер (Dolphin)",
       aliases=("файлы", "проводник", "долфин", "dolphin"))
async def open_files(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("dolphin || nautilus || pcmanfm")
    return f"file manager launched pid={pid}"


@skill(id="open_settings", category=_UI,
       description="открыть системные настройки KDE",
       aliases=("настройки", "параметры системы"))
async def open_settings(ctx: SkillContext, args: dict[str, Any]) -> str:
    # Plasma 6 — systemsettings; Plasma 5 — systemsettings5.
    pid = await ctx.spawn("systemsettings || systemsettings5")
    return f"settings launched pid={pid}"


@skill(id="open_screenshot_tool", category=_UI,
       description="сделать скриншот (Spectacle)",
       aliases=("скриншот", "снимок экрана", "screenshot"))
async def open_screenshot_tool(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("spectacle")
    return f"spectacle launched pid={pid}"


# ───────────────────────── Звук (PipeWire / wpctl) ─────────────────────────
@skill(id="volume_up", category=_UI, description="увеличить громкость",
       aliases=("громче", "сделай громче"))
async def volume_up(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%+")
    return "volume +5%" if rc == 0 else f"volume_up rc={rc} {err[:80]}"


@skill(id="volume_down", category=_UI, description="уменьшить громкость",
       aliases=("тише", "сделай тише"))
async def volume_down(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%-")
    return "volume -5%" if rc == 0 else f"volume_down rc={rc} {err[:80]}"


@skill(id="volume_mute", category=_UI, description="приглушить/включить звук (toggle)",
       aliases=("выключи звук", "mute", "приглуши"))
async def volume_mute(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle")
    return "mute toggled" if rc == 0 else f"volume_mute rc={rc} {err[:80]}"


@skill(id="set_volume", category=_UI,
       description="установить громкость в процентах (args.percent 0-100)",
       params={"percent": {"type": "integer", "description": "0-100"}})
async def set_volume(ctx: SkillContext, args: dict[str, Any]) -> str:
    pct = _clamp_pct(args.get("percent"), default=50)
    rc, _, err = await ctx.run(f"wpctl set-volume @DEFAULT_AUDIO_SINK@ {pct}%")
    return f"volume={pct}%" if rc == 0 else f"set_volume rc={rc} {err[:80]}"


# ───────────────────────── Яркость (brightnessctl) ─────────────────────────
@skill(id="brightness_up", category=_UI, description="увеличить яркость экрана",
       aliases=("ярче", "сделай ярче"))
async def brightness_up(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("brightnessctl set 10%+")
    return "brightness +10%" if rc == 0 else f"brightness_up rc={rc} {err[:80]}"


@skill(id="brightness_down", category=_UI, description="уменьшить яркость экрана",
       aliases=("темнее", "убавь яркость"))
async def brightness_down(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("brightnessctl set 10%-")
    return "brightness -10%" if rc == 0 else f"brightness_down rc={rc} {err[:80]}"


@skill(id="set_brightness", category=_UI,
       description="установить яркость в процентах (args.percent 1-100)",
       params={"percent": {"type": "integer", "description": "1-100"}})
async def set_brightness(ctx: SkillContext, args: dict[str, Any]) -> str:
    pct = _clamp_pct(args.get("percent"), default=70, lo=1)
    rc, _, err = await ctx.run(f"brightnessctl set {pct}%")
    return f"brightness={pct}%" if rc == 0 else f"set_brightness rc={rc} {err[:80]}"


# ───────────────────────── Night Light (фильтр синего) ─────────────────────────
@skill(id="enable_night_mode", category=_UI,
       description="включить ночной режим / фильтр синего света",
       aliases=("ночной режим", "фильтр синего", "глаза устали", "тёплый экран"))
async def enable_night_mode(ctx: SkillContext, args: dict[str, Any]) -> str:
    # VERIFY on target (Plasma 6 Night Light DBus). Резервно — переключение
    # через глобальный ярлык KWin (если прямой метод недоступен).
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.KWin /org/kde/KWin/NightLight org.kde.KWin.NightLight.setEnabled true "
        "|| qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Toggle Night Color'"
    )
    return "night mode on" if rc == 0 else "night mode toggle attempted"


@skill(id="disable_night_mode", category=_UI,
       description="выключить ночной режим / фильтр синего света",
       aliases=("выключи ночной режим", "убери фильтр"))
async def disable_night_mode(ctx: SkillContext, args: dict[str, Any]) -> str:
    # VERIFY on target.
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.KWin /org/kde/KWin/NightLight org.kde.KWin.NightLight.setEnabled false "
        "|| qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Toggle Night Color'"
    )
    return "night mode off" if rc == 0 else "night mode toggle attempted"


# ───────────────────────── Окна / рабочие столы (KWin) ─────────────────────────
@skill(id="lock_screen", category=_UI, description="заблокировать экран",
       aliases=("заблокируй", "блокировка"))
async def lock_screen(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "loginctl lock-session || qdbus6 org.kde.ScreenSaver /ScreenSaver Lock"
    )
    return "screen locked" if rc == 0 else "lock attempted"


@skill(id="show_desktop", category=_UI,
       description="свернуть все окна / показать рабочий стол",
       aliases=("сверни всё", "покажи рабочий стол", "сверни окна"))
async def show_desktop(ctx: SkillContext, args: dict[str, Any]) -> str:
    # VERIFY on target — имя ярлыка KWin.
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Show Desktop'"
    )
    return "show desktop" if rc == 0 else "show_desktop attempted"


@skill(id="close_active_window", category=_UI,
       description="закрыть активное окно",
       aliases=("закрой окно", "закрой это"))
async def close_active_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    # VERIFY on target — имя ярлыка KWin.
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Window Close'"
    )
    return "active window closed" if rc == 0 else "close attempted"


@skill(id="switch_desktop_next", category=_UI,
       description="переключиться на следующий рабочий стол",
       aliases=("следующий рабочий стол",))
async def switch_desktop_next(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Switch to Next Desktop'"
    )
    return "desktop next" if rc == 0 else "switch attempted"


@skill(id="switch_desktop_prev", category=_UI,
       description="переключиться на предыдущий рабочий стол",
       aliases=("предыдущий рабочий стол",))
async def switch_desktop_prev(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Switch to Previous Desktop'"
    )
    return "desktop prev" if rc == 0 else "switch attempted"


# ───────────────────────── Запуск приложений (доп.) ─────────────────────────
@skill(id="open_text_editor", category=_UI,
       description="открыть текстовый редактор (Kate)",
       aliases=("редактор", "блокнот", "kate"))
async def open_text_editor(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("kate || kwrite || gedit || nano")
    return f"editor launched pid={pid}"


@skill(id="open_calculator", category=_UI,
       description="открыть калькулятор",
       aliases=("калькулятор", "посчитать"))
async def open_calculator(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("kcalc || gnome-calculator || qalculate-gtk")
    return f"calculator launched pid={pid}"


@skill(id="open_system_monitor", category=_UI,
       description="открыть системный монитор",
       aliases=("монитор системы", "диспетчер задач", "ksysguard"))
async def open_system_monitor(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("plasma-systemmonitor || ksysguard || gnome-system-monitor")
    return f"system monitor launched pid={pid}"


@skill(id="open_app_launcher", category=_UI,
       description="открыть поиск приложений (KRunner)",
       aliases=("запуск приложений", "поиск", "krunner"))
async def open_app_launcher(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn(
        "qdbus6 org.kde.krunner /App org.kde.krunner.App.display 2>/dev/null || krunner"
    )
    return f"launcher opened pid={pid}"


# ───────────────────────── Медиа (playerctl) ─────────────────────────
@skill(id="media_play_pause", category=_UI,
       description="воспроизведение/пауза медиа",
       aliases=("пауза", "плей", "поставь на паузу", "продолжи"))
async def media_play_pause(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("playerctl play-pause")
    return "media toggled" if rc == 0 else f"media rc={rc} {err[:80]}"


@skill(id="media_next", category=_UI, description="следующий трек",
       aliases=("следующий трек", "переключи трек", "дальше"))
async def media_next(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("playerctl next")
    return "media next" if rc == 0 else f"media rc={rc} {err[:80]}"


@skill(id="media_previous", category=_UI, description="предыдущий трек",
       aliases=("предыдущий трек", "верни трек", "назад"))
async def media_previous(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("playerctl previous")
    return "media prev" if rc == 0 else f"media rc={rc} {err[:80]}"


@skill(id="media_stop", category=_UI, description="остановить воспроизведение",
       aliases=("останови музыку", "стоп музыка"))
async def media_stop(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("playerctl stop")
    return "media stopped" if rc == 0 else f"media rc={rc} {err[:80]}"


@skill(id="mic_mute_toggle", category=_UI,
       description="приглушить/включить микрофон (toggle)",
       aliases=("выключи микрофон", "включи микрофон", "мьют микрофона"))
async def mic_mute_toggle(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, err = await ctx.run("wpctl set-mute @DEFAULT_AUDIO_SOURCE@ toggle")
    return "mic toggled" if rc == 0 else f"mic rc={rc} {err[:80]}"


# ───────────────────────── Скриншоты (доп.) ─────────────────────────
@skill(id="screenshot_region", category=_UI,
       description="скриншот выделенной области",
       aliases=("скриншот области", "выдели и сними"))
async def screenshot_region(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("spectacle -r -b")
    return f"region screenshot pid={pid}"


@skill(id="screenshot_active_window", category=_UI,
       description="скриншот активного окна",
       aliases=("скриншот окна", "сними окно"))
async def screenshot_active_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    pid = await ctx.spawn("spectacle -a -b")
    return f"window screenshot pid={pid}"


# ───────────────────────── Окна (доп., KWin shortcuts) ─────────────────────────
# VERIFY on target — имена ярлыков KWin между версиями Plasma менялись.
@skill(id="maximize_window", category=_UI, description="развернуть активное окно",
       aliases=("разверни окно", "на весь экран"))
async def maximize_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Window Maximize'"
    )
    return "window maximized" if rc == 0 else "maximize attempted"


@skill(id="minimize_window", category=_UI, description="свернуть активное окно",
       aliases=("сверни окно",))
async def minimize_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Window Minimize'"
    )
    return "window minimized" if rc == 0 else "minimize attempted"


@skill(id="toggle_fullscreen", category=_UI,
       description="полноэкранный режим активного окна",
       aliases=("полный экран", "фуллскрин"))
async def toggle_fullscreen(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Window Fullscreen'"
    )
    return "fullscreen toggled" if rc == 0 else "fullscreen attempted"


@skill(id="switch_window", category=_UI,
       description="переключиться между окнами (Alt-Tab)",
       aliases=("переключи окно", "следующее окно"))
async def switch_window(ctx: SkillContext, args: dict[str, Any]) -> str:
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.kglobalaccel /component/kwin invokeShortcut 'Walk Through Windows'"
    )
    return "window switched" if rc == 0 else "switch attempted"


# ───────────────────────── Сессия ─────────────────────────
@skill(id="log_out", category=_UI, destructive=True,
       description="выйти из сессии (закрывает все приложения)",
       aliases=("выйти из системы", "разлогинься", "выход из сессии"))
async def log_out(ctx: SkillContext, args: dict[str, Any]) -> str:
    # VERIFY on target. destructive=True → ядро спросит подтверждение голосом.
    rc, _, _ = await ctx.run(
        "qdbus6 org.kde.Shutdown /Shutdown org.kde.Shutdown.logout"
    )
    return "logout requested" if rc == 0 else "logout attempted"

