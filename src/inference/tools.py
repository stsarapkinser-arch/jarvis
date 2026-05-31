"""Native Function-Calling toolset — смерть текстового парсинга.

Здесь живут СТРОГИЕ определения инструментов (OpenAI Tool-Use / JSON Schema),
которые мы отдаём ``llama-server`` в массиве ``tools``. Модель больше не пишет
``<thought>``/``<say>`` в тексте — она возвращает детерминированный JSON с
вызовами функций, а Python-оркестратор их исполняет.

Модуль НАМЕРЕННО чистый и без side-effect'ов: только данные (схемы) и типы
(dataclass-обёртки над аргументами). Диспетчеризация вызовов — в
``src.core.orchestrator`` (там, где есть доступ к шине/ShadowExec/TTS), а
транспортный цикл — в ``src.inference.agent``.

Пять инструментов из ТЗ оператора:
    * ``internal_monologue`` — Chain-of-Thought; Python логирует, но НЕ озвучивает.
    * ``speak_response``     — единственный путь голоса наружу (mood → SOX-профиль).
    * ``set_hud_state``      — нейросеть сама управляет визором Aegis.
    * ``read_telemetry``     — чтение живых сенсоров (фидбэк уходит обратно в модель).
    * ``execute_bash``       — команда в реальную систему (через Shadow Exec + гейты).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final


# ─────────────────────────── Enums (строгие домены) ───────────────────────────
class SpeakMood(StrEnum):
    """Тон голоса. Маппится на SOX-DSP-профиль в orchestrator._tone_for_mood."""
    PROFESSIONAL = "professional"
    ALERT = "alert"
    IRONIC = "ironic"


class HudAnimation(StrEnum):
    """Анимация рамки/сферы визора Aegis."""
    IDLE = "idle"
    PULSE = "pulse"
    GLITCH = "glitch"


class TelemetrySensor(StrEnum):
    """Доступные сенсоры для read_telemetry."""
    CPU = "cpu"
    RAM = "ram"
    NETWORK = "network"
    PIXEL_PHONE = "pixel_phone"


# ─────────────────────────── Имена инструментов ───────────────────────────────
class ToolName(StrEnum):
    INTERNAL_MONOLOGUE = "internal_monologue"
    SPEAK_RESPONSE = "speak_response"
    SET_HUD_STATE = "set_hud_state"
    READ_TELEMETRY = "read_telemetry"
    EXECUTE_BASH = "execute_bash"
    RUN_SKILL = "run_skill"


# ─────────────────────────── JSON Schemas (OpenAI Tool-Use) ───────────────────
# Формат строго соответствует тому, что ждёт /v1/chat/completions у llama-server:
#   {"type": "function", "function": {"name", "description", "parameters": <JSON Schema>}}
_INTERNAL_MONOLOGUE_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": ToolName.INTERNAL_MONOLOGUE.value,
        "description": (
            "Внутреннее рассуждение Джарвиса (Chain-of-Thought). Используй, когда "
            "нужно подумать перед действием. Это НЕ озвучивается оператору — только "
            "пишется в лог. Одно-два предложения."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Краткая мысль на русском. Видишь только ты и лог.",
                }
            },
            "required": ["thought"],
            "additionalProperties": False,
        },
    },
}

_SPEAK_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": ToolName.SPEAK_RESPONSE.value,
        "description": (
            "Сказать вслух (единственный голосовой канал). Кратко, без markdown. "
            "Пунктуация = ритм: … пауза, точки разделяют важное. «сэр»."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Текст речи."},
                "mood": {
                    "type": "string",
                    "enum": [m.value for m in SpeakMood],
                    "description": "professional / alert (срочно) / ironic.",
                },
                "speed": {
                    "type": "number",
                    "description": "темп: 1.0 норма, <1 медленнее, >1 быстрее (опц.).",
                },
                "pause": {
                    "type": "number",
                    "description": "пауза между предложениями, сек (опц.).",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

_SET_HUD_STATE_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": ToolName.SET_HUD_STATE.value,
        "description": "Визор Aegis: покажи состояние (работа/тревога/успех).",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "cyan/blue норма, amber внимание, red тревога, green успех, white нейтрально.",
                },
                "animation": {
                    "type": "string",
                    "enum": [a.value for a in HudAnimation],
                    "description": "idle покой / pulse работа / glitch тревога.",
                },
            },
            "required": ["color", "animation"],
            "additionalProperties": False,
        },
    },
}

_READ_TELEMETRY_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": ToolName.READ_TELEMETRY.value,
        "description": "Прочитать сенсор; результат вернётся тебе — не выдумывай цифры.",
        "parameters": {
            "type": "object",
            "properties": {
                "sensor": {
                    "type": "string",
                    "enum": [s.value for s in TelemetrySensor],
                    "description": "cpu / ram / network / pixel_phone.",
                }
            },
            "required": ["sensor"],
            "additionalProperties": False,
        },
    },
}

_EXECUTE_BASH_SCHEMA: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": ToolName.EXECUTE_BASH.value,
        "description": (
            "Выполнить bash в реальной системе. Сэндбокс и подтверждение "
            "разрушительных команд ядро делает само — не дублируй гейтинг."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Одна команда, без markdown и backticks.",
                },
                "requires_sudo": {
                    "type": "boolean",
                    "description": "true — нужен root (sudo -n).",
                },
                "background": {
                    "type": "boolean",
                    "description": "true — в фоне, не ждать.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

# Полный реестр.
TOOL_SCHEMAS: Final[tuple[dict[str, Any], ...]] = (
    _INTERNAL_MONOLOGUE_SCHEMA,
    _SPEAK_RESPONSE_SCHEMA,
    _SET_HUD_STATE_SCHEMA,
    _READ_TELEMETRY_SCHEMA,
    _EXECUTE_BASH_SCHEMA,
)

TOOLS_BY_NAME: Final[dict[str, dict[str, Any]]] = {
    s["function"]["name"]: s for s in TOOL_SCHEMAS
}


# ─────────────────────────── run_skill (Skill Registry) ───────────────────────
# Архитектурный переворот: модель НЕ пишет bash на лету. Она выбирает skill_id
# из заранее написанного каталога (см. src/skills) — grammar-enum физически не
# даёт выдумать несуществующий навык. reply встроен в тот же вызов → модель и
# говорит, и действует за ОДИН раунд (критично на N100, декод ~1 т/с). execute_bash
# остаётся как gated-fallback для длинного хвоста (того, чего нет в каталоге).
def build_run_skill_schema(skill_ids: list[str]) -> dict[str, Any]:
    """Собрать схему run_skill с enum по ПЕРЕДАННЫМ id навыков.

    enum — по ВСЕМ навыкам (а не по категории), поэтому схема инструмента
    идентична между action-категориями → llama-server переиспользует KV-префикс
    tool-блока. Семантику (какой id под какую фразу) даёт каталог в микро-промпте."""
    return {
        "type": "function",
        "function": {
            "name": ToolName.RUN_SKILL.value,
            "description": (
                "Выполнить ГОТОВЫЙ навык по skill_id (предпочтительный путь). "
                "Выбери skill_id из списка навыков в промпте; заполни reply — "
                "короткую фразу вслух оператору. args обычно пуст."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "enum": list(skill_ids),
                        "description": "Идентификатор навыка из каталога.",
                    },
                    "reply": {
                        "type": "string",
                        "description": "Короткая фраза вслух, без markdown. «сэр».",
                    },
                    "mood": {
                        "type": "string",
                        "enum": [m.value for m in SpeakMood],
                        "description": "professional / alert / ironic (опц.).",
                    },
                    "args": {
                        "type": "object",
                        "description": "Параметры навыка, если требуются (обычно пусто).",
                    },
                },
                "required": ["skill_id", "reply"],
                "additionalProperties": False,
            },
        },
    }


# ─────────────────────────── Tool-subset на категорию ─────────────────────────
# Семантический маршрутизатор отдаёт модели ТОЛЬКО релевантные инструменты.
#
# Action-категории (SYSTEM_OPS/UI_CONTROL/PENTEST_RECON) получают [run_skill,
# execute_bash]: сперва пытаемся попасть в готовый навык, иначе — bash под
# гейтом (ShadowExec + подтверждение). Набор ИДЕНТИЧЕН между ними (run_skill
# с глобальным enum + execute_bash) → стабильный KV-префикс tool-блока.
#
# CONVERSATION намеренно БЕЗ системного доступа — только речь и визор (это
# бесплатное свойство безопасности: чистый разговор не тронет систему).
#
# internal_monologue НЕ входит в боевые подмножества: на N100 отдельный
# tool-вызов «подумать» — лишний раунд диалога. Схема сохранена, в горячий путь
# не попадает.
def tools_for_category(category: str) -> list[dict[str, Any]]:
    """Вернуть подмножество JSON-схем для категории интента.

    Принимает строку или StrEnum (router.IntentCategory). CONVERSATION →
    речь+визор. Любая другая (включая неизвестную) → run_skill + execute_bash."""
    # Ленивый импорт реестра: tools импортируется очень рано (openai_client),
    # а skills тянет router/event_bus — держим зависимость на отложенном пути.
    from src.skills import skill_ids

    if str(category) == "CONVERSATION":
        return [_SPEAK_RESPONSE_SCHEMA, _SET_HUD_STATE_SCHEMA]
    return [build_run_skill_schema(skill_ids()), _EXECUTE_BASH_SCHEMA]


# ─────────────────────────── Типизированные аргументы ─────────────────────────
# llama-server отдаёт function.arguments как JSON-строку. Эти dataclass'ы
# валидируют/коэрсят её в строгие Python-типы — никаких KeyError в горячем пути.
def _coerce_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else (default if value is None else str(value))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _coerce_opt_float(value: Any) -> float | None:
    """Опциональное число (speed/pause). Невалидное/пустое → None (дефолт состояния)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_arguments(raw: Any) -> dict[str, Any]:
    """function.arguments может прийти строкой-JSON, dict'ом или мусором.

    Возвращаем dict при любом раскладе — пустой, если распарсить не удалось."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass(frozen=True, slots=True)
class MonologueArgs:
    thought: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MonologueArgs:
        return cls(thought=_coerce_str(d.get("thought")))


@dataclass(frozen=True, slots=True)
class SpeakArgs:
    text: str
    mood: SpeakMood = SpeakMood.PROFESSIONAL
    speed: float | None = None      # когнитивная просодия: темп (1.0 норма)
    pause: float | None = None      # пауза между предложениями, сек

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpeakArgs:
        raw_mood = _coerce_str(d.get("mood"), SpeakMood.PROFESSIONAL.value).lower()
        try:
            mood = SpeakMood(raw_mood)
        except ValueError:
            mood = SpeakMood.PROFESSIONAL
        return cls(
            text=_coerce_str(d.get("text")),
            mood=mood,
            speed=_coerce_opt_float(d.get("speed")),
            pause=_coerce_opt_float(d.get("pause")),
        )


@dataclass(frozen=True, slots=True)
class HudArgs:
    color: str
    animation: HudAnimation = HudAnimation.PULSE

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HudArgs:
        raw_anim = _coerce_str(d.get("animation"), HudAnimation.PULSE.value).lower()
        try:
            animation = HudAnimation(raw_anim)
        except ValueError:
            animation = HudAnimation.PULSE
        return cls(color=_coerce_str(d.get("color"), "cyan").lower(), animation=animation)


@dataclass(frozen=True, slots=True)
class TelemetryArgs:
    sensor: TelemetrySensor

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetryArgs:
        raw = _coerce_str(d.get("sensor"), TelemetrySensor.CPU.value).lower()
        try:
            sensor = TelemetrySensor(raw)
        except ValueError:
            sensor = TelemetrySensor.CPU
        return cls(sensor=sensor)


@dataclass(frozen=True, slots=True)
class ExecuteBashArgs:
    command: str
    requires_sudo: bool = False
    background: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecuteBashArgs:
        return cls(
            command=_coerce_str(d.get("command")).strip(),
            requires_sudo=_coerce_bool(d.get("requires_sudo")),
            background=_coerce_bool(d.get("background")),
        )


@dataclass(frozen=True, slots=True)
class RunSkillArgs:
    skill_id: str
    reply: str = ""
    mood: SpeakMood = SpeakMood.PROFESSIONAL
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunSkillArgs:
        raw_mood = _coerce_str(d.get("mood"), SpeakMood.PROFESSIONAL.value).lower()
        try:
            mood = SpeakMood(raw_mood)
        except ValueError:
            mood = SpeakMood.PROFESSIONAL
        inner = d.get("args")
        return cls(
            skill_id=_coerce_str(d.get("skill_id")).strip(),
            reply=_coerce_str(d.get("reply")).strip(),
            mood=mood,
            args=inner if isinstance(inner, dict) else {},
        )


__all__ = [
    "SpeakMood",
    "HudAnimation",
    "TelemetrySensor",
    "ToolName",
    "TOOL_SCHEMAS",
    "TOOLS_BY_NAME",
    "tools_for_category",
    "build_run_skill_schema",
    "parse_arguments",
    "MonologueArgs",
    "SpeakArgs",
    "HudArgs",
    "TelemetryArgs",
    "ExecuteBashArgs",
    "RunSkillArgs",
]
