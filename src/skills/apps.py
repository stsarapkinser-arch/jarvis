"""Резолвер приложений (L1a) — «открой X» без 3B.

Ступень L1a над шаблонами со слотами: распознаёт повелительное «открой/запусти
{приложение}» и резолвит произнесённое имя в ВЫВЕРЕННУЮ команду запуска из
курируемого каталога — детерминированно, минуя 3B.

Почему это отдельный слой, а не просто ещё один regex в patterns.py: открытый
слот {приложение} нельзя сматчить таблицей фиксированных фраз (комбинаторный
взрыв) и нельзя извлечь чистым regex (нужны синонимы + устойчивость к ошибкам
Vosk). Поэтому: regex ловит ГЛАГОЛ запуска и забирает хвост как имя, а
сопоставление имени с приложением делает резолвер (точный синоним → fuzzy).

Безопасность (важно): произнесённое имя НЕ попадает в shell. Резолвер отображает
его в ЗАХАРДКОЖЕННУЮ ``command`` каталога; навык ``open_app`` принимает только имя
(ключ/синоним), команду берёт сам из каталога. Спутанный Vosk-ом слот в худшем
случае не зарезолвится (→ 3B), но никогда не станет произвольной командой. Это
сохраняет принцип Skill Registry «нейросеть не пишет bash».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.inference.router import IntentCategory
from src.skills.registry import SkillContext, normalize_phrase, skill

log = logging.getLogger("jarvis.apps")

# Минимальная длина имени для fuzzy-сопоставления: короткие токены («код», «вс»)
# матчатся только точно, иначе шум Vosk ловит случайное приложение.
_FUZZY_MIN_LEN = 4
# Порог близости (difflib ratio). Строгий — лучше уйти к 3B, чем открыть не то.
_FUZZY_THRESHOLD = 0.84


@dataclass(frozen=True, slots=True)
class AppEntry:
    """Одно приложение каталога.

    ``command`` — выверенная OR-цепочка бинарей (устойчивость к набору пакетов
    дистрибутива), запускается detached через ``ctx.spawn``. ``synonyms`` — как
    оператор называет приложение голосом (нормализованные формы; ключ добавляется
    в индекс автоматически)."""
    key: str
    label: str
    command: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)


# ─────────────────────────── Каталог приложений ───────────────────────────
# OR-цепочки покрывают разные DE/пакеты; первый существующий бинарь выигрывает.
_CATALOG: tuple[AppEntry, ...] = (
    AppEntry("terminal", "терминал", "konsole || xterm || x-terminal-emulator",
             ("терминал", "консоль", "konsole", "терминалку", "коммандную строку",
              "командную строку", "терминал кали", "эмулятор терминала")),
    AppEntry("browser", "браузер",
             "google-chrome-stable || google-chrome || chromium || chromium-browser || xdg-open https://",
             ("браузер", "хром", "chrome", "хромиум", "гугл хром", "интернет", "обозреватель",
              "веб браузер", "интернет браузер", "браузер хром")),
    AppEntry("firefox", "Firefox", "firefox || firefox-esr",
             ("фаерфокс", "файрфокс", "firefox", "лиса", "огненная лиса", "фф")),
    AppEntry("files", "файловый менеджер", "dolphin || nautilus || pcmanfm || thunar",
             ("файлы", "проводник", "файловый менеджер", "dolphin", "долфин", "папки",
              "менеджер файлов", "диспетчер файлов", "файловый проводник")),
    AppEntry("editor", "редактор", "kate || kwrite || gedit || mousepad || nano",
             ("редактор", "блокнот", "текстовый редактор", "kate", "кейт",
              "редактор текста", "текстовик")),
    AppEntry("code", "редактор кода", "code || codium || code-oss",
             ("код", "вс код", "vs code", "vscode", "студия кода", "код редактор")),
    AppEntry("calculator", "калькулятор", "kcalc || gnome-calculator || qalculate-gtk",
             ("калькулятор", "калькулятор", "посчитать", "счёты")),
    AppEntry("settings", "настройки", "systemsettings || systemsettings5 || gnome-control-center",
             ("настройки", "параметры", "параметры системы", "системные настройки",
              "настройки системы", "настройки системные", "панель управления",
              "центр управления", "конфигурация системы")),
    AppEntry("screenshot", "снимок экрана", "spectacle || flameshot gui || gnome-screenshot",
             ("скриншот", "снимок экрана", "spectacle", "спектакль", "заскринь")),
    AppEntry("system_monitor", "системный монитор",
             "plasma-systemmonitor || ksysguard || gnome-system-monitor",
             ("монитор системы", "системный монитор", "диспетчер задач", "ksysguard",
              "монитор ресурсов", "системный мониторинг", "диспетчер процессов")),
    AppEntry("telegram", "Telegram", "telegram-desktop || Telegram || telegram",
             ("телеграм", "телега", "telegram", "тг")),
    AppEntry("discord", "Discord", "discord || Discord",
             ("дискорд", "discord", "диск")),
    AppEntry("spotify", "Spotify", "spotify",
             ("спотифай", "spotify", "спотик")),
    AppEntry("vlc", "медиаплеер", "vlc || mpv || celluloid",
             ("плеер", "медиаплеер", "видеоплеер", "vlc", "вэлси")),
    AppEntry("gimp", "GIMP", "gimp",
             ("гимп", "gimp", "фоторедактор")),
    AppEntry("image_viewer", "просмотрщик изображений", "gwenview || eog || feh || nomacs",
             ("просмотр фото", "галерея", "картинки", "просмотрщик", "gwenview")),
    AppEntry("writer", "текстовый процессор", "libreoffice --writer || lowriter",
             ("ворд", "текстовый процессор", "либре офис райтер", "документ", "writer")),
    AppEntry("spreadsheet", "таблицы", "libreoffice --calc || localc",
             ("эксель", "таблицы", "табличный процессор", "либре офис калк", "calc")),
    AppEntry("impress", "презентации", "libreoffice --impress || loimpress",
             ("презентация", "повер поинт", "слайды", "impress")),
    AppEntry("mail", "почта", "thunderbird",
             ("почта", "почтовый клиент", "тандерберд", "thunderbird")),
    AppEntry("obsidian", "Obsidian", "obsidian",
             ("обсидиан", "obsidian", "заметки")),
)


def _build_index() -> dict[str, AppEntry]:
    index: dict[str, AppEntry] = {}
    for entry in _CATALOG:
        # Ключ — тоже валидное имя (резолвер должен принимать его round-trip).
        for name in (entry.key, *entry.synonyms):
            norm = normalize_phrase(name)
            if norm:
                index.setdefault(norm, entry)
    return index


_SYNONYM_INDEX: dict[str, AppEntry] = _build_index()


def get(key: str) -> AppEntry | None:
    """Каталожная запись по точному ключу (canonical id)."""
    norm = normalize_phrase(key)
    entry = _SYNONYM_INDEX.get(norm)
    return entry if entry is not None and entry.key == norm else _by_key(norm)


def _by_key(norm: str) -> AppEntry | None:
    for entry in _CATALOG:
        if entry.key == norm:
            return entry
    return None


def resolve(name: str) -> AppEntry | None:
    """Произнесённое имя → приложение каталога, иначе None.

    Точный синоним → fuzzy (difflib) с порогом. None означает «не уверен» —
    интент уходит 3B, а не открывает наугад не то приложение."""
    key = normalize_phrase(name)
    if not key:
        return None
    exact = _SYNONYM_INDEX.get(key)
    if exact is not None:
        return exact
    if len(key) < _FUZZY_MIN_LEN:
        return None
    best: AppEntry | None = None
    best_ratio = 0.0
    for syn, entry in _SYNONYM_INDEX.items():
        ratio = SequenceMatcher(None, key, syn).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, entry
    return best if best_ratio >= _FUZZY_THRESHOLD else None


# ─────────────────────────── Навык запуска ───────────────────────────
@skill(id="open_app", category=IntentCategory.UI_CONTROL,
       description="открыть приложение по имени (args.app: имя/синоним из каталога)",
       params={"app": {"type": "string", "description": "имя приложения, напр. 'браузер', 'телеграм'"}})
async def open_app(ctx: SkillContext, args: dict[str, Any]) -> str:
    """Запустить приложение по имени. Команду берём ИЗ КАТАЛОГА (резолвер), а не
    из аргумента — произнесённое имя в shell не попадает (анти-инъекция)."""
    entry = resolve(str(args.get("app", "")))
    if entry is None:
        return "Не нашёл такого приложения в каталоге, сэр."
    pid = await ctx.spawn(entry.command)
    return f"{entry.label} launched pid={pid}"


def all_apps() -> tuple[AppEntry, ...]:
    return _CATALOG


__all__ = [
    "AppEntry",
    "resolve",
    "get",
    "open_app",
    "all_apps",
]
