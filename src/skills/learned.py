"""Learned skills — каталог, который Джарвис дописывает себе сам.

Замыкание петли самораширения: когда gated-fallback ``execute_bash`` успешно
отрабатывает один и тот же интент несколько раз (см. ``src/skills/learning.py``),
Джарвис предлагает оператору закрепить это как постоянный навык. После
голосового подтверждения команда записывается сюда — в ``config/learned_skills``
— и **горячая перезагрузка** (FileChangeWatcher следит за ``config/``) поднимает
процесс со свежим кодом, где навык уже зарегистрирован. Длинный хвост перестаёт
быть проблемой: каталог растёт из реальной эксплуатации, без ручного труда.

Модель безопасности (важно):
  * Выученный навык — это ФИКСИРОВАННАЯ команда, которая УЖЕ прошла полный
    конвейер безопасности (ShadowExec dry-run + гейт разрушительных) при первых
    запусках. Повторный её запуск не даёт новых прав.
  * Разрушительные команды НИКОГДА не учатся (см. learning.py).
  * Никаких аргументов/подстановок в v1 → нулевая поверхность инъекций: строка
    исполняется ровно так, как была проверена.
  * Промоушен только по явному голосовому «да» оператора.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.inference.router import IntentCategory
from src.skills.registry import Skill, SkillContext, get as _get, register

log = logging.getLogger("jarvis.skills.learned")

# config/ под наблюдением FileChangeWatcher → запись сюда триггерит os.execv
# и перезагрузку, на которой навык и регистрируется. jarvis_memory/ — НЕ под
# наблюдением (там же счётчики использования), чтобы обучение не дёргало reload.
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
LEARNED_PATH = _CONFIG_DIR / "learned_skills.json"

# Префикс id всех выученных навыков — чтобы их было видно в каталоге и нельзя
# было спутать со встроенными.
LEARNED_PREFIX = "learned_"

# Потолок: не плодим бесконечный каталог, который раздует промпт категории.
MAX_LEARNED_SKILLS = 64


def _read_raw() -> list[dict[str, Any]]:
    """Прочитать сырой список записей. Любая ошибка → пустой список (не падаем)."""
    try:
        text = LEARNED_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        log.exception("чтение %s не удалось", LEARNED_PATH)
        return []
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        log.warning("%s повреждён — игнорирую", LEARNED_PATH)
        return []
    return data if isinstance(data, list) else []


def _valid_entry(e: Any) -> bool:
    """Минимальная валидация записи: id с правильным префиксом и непустая команда."""
    if not isinstance(e, dict):
        return False
    sid = e.get("id", "")
    cmd = e.get("command", "")
    return (
        isinstance(sid, str) and sid.startswith(LEARNED_PREFIX) and len(sid) > len(LEARNED_PREFIX)
        and isinstance(cmd, str) and bool(cmd.strip())
    )


def _make_handler(command: str, kind: str):
    """Построить async-хендлер навыка над ФИКСИРОВАННОЙ командой.

    ``kind="spawn"`` — detached GUI-запуск (возвращаем pid), иначе ``run`` с
    ожиданием и кратким статусом результата."""
    async def handler(ctx: SkillContext, args: dict[str, Any]) -> str:
        if kind == "spawn":
            pid = await ctx.spawn(command)
            return f"launched pid={pid}"
        rc, out, err = await ctx.run(command)
        tail = (out.strip() or err.strip())[:200]
        return f"rc={rc}; {tail}" if tail else f"rc={rc}"
    return handler


def _build_skill(e: dict[str, Any]) -> Skill | None:
    """Собрать Skill из валидной записи. None — если категория/поля битые."""
    try:
        category = IntentCategory(str(e.get("category", IntentCategory.SYSTEM_OPS)))
    except ValueError:
        category = IntentCategory.SYSTEM_OPS
    sid = str(e["id"]).strip()
    command = str(e["command"]).strip()
    kind = "spawn" if str(e.get("kind", "run")) == "spawn" else "run"
    aliases = tuple(str(a) for a in (e.get("aliases") or ()) if isinstance(a, str))
    description = str(e.get("description") or sid[len(LEARNED_PREFIX):].replace("_", " "))
    return Skill(
        id=sid,
        category=category,
        description=description,
        handler=_make_handler(command, kind),
        # Выученные навыки никогда не разрушительны (learning.py их не учит) и
        # озвучивают reply модели, а не сырой вывод команды.
        destructive=False,
        speaks_result=False,
        aliases=aliases,
    )


def load_learned() -> int:
    """Зарегистрировать все валидные выученные навыки. Возвращает их число.

    Идемпотентна по отношению к дублям: навык с уже занятым id пропускается
    (встроенный каталог всегда главнее выученного)."""
    count = 0
    for e in _read_raw()[:MAX_LEARNED_SKILLS]:
        if not _valid_entry(e):
            log.warning("пропускаю невалидную выученную запись: %r", e)
            continue
        sid = str(e["id"]).strip()
        if _get(sid) is not None:
            log.info("выученный навык %s уже зарегистрирован — пропуск", sid)
            continue
        sk = _build_skill(e)
        if sk is None:
            continue
        try:
            register(sk)
            count += 1
        except ValueError:
            # Гонка дублей внутри файла — не фатально.
            log.warning("дубль id в learned_skills: %s", sid)
    if count:
        log.info("загружено выученных навыков: %d", count)
    return count


def append_learned(entry: dict[str, Any]) -> bool:
    """Дописать запись и сохранить файл (вызывается промоушеном после «да»).

    Запись в config/ намеренно триггерит горячую перезагрузку — на ней навык
    подхватится ``load_learned``. Возвращает False при невалидной записи или
    переполнении каталога."""
    if not _valid_entry(entry):
        log.error("отказ записать невалидный выученный навык: %r", entry)
        return False
    existing = _read_raw()
    if any(isinstance(e, dict) and e.get("id") == entry["id"] for e in existing):
        log.info("выученный навык %s уже в файле", entry["id"])
        return False
    if len(existing) >= MAX_LEARNED_SKILLS:
        log.warning("каталог выученных навыков полон (%d) — не добавляю", MAX_LEARNED_SKILLS)
        return False
    existing.append(entry)
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LEARNED_PATH.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log.info("выученный навык записан: %s", entry["id"])
        return True
    except OSError:
        log.exception("запись %s не удалась", LEARNED_PATH)
        return False


# Side-effect импорта: регистрируем выученные навыки (как и встроенные модули).
load_learned()
