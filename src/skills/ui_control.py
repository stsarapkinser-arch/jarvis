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
# DBus-интерфейсы Night Light и глобальные ярлыки KWin между версиями менялись —
# при первом прогоне на машине оператора их стоит подтвердить (и поправить ОДНУ
# строку, если что — в этом и весь смысл архитектуры).
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
