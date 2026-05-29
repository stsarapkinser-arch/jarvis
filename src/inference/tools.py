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
from dataclasses import dataclass
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
            "ЕДИНСТВЕННЫЙ способ заговорить с оператором вслух. Живая речь, без "
            "markdown и списков. Ты УПРАВЛЯЕШЬ своим голосом пунктуацией: "
            "многоточие … — глубокая пауза (анализ, раздумье); короткие "
            "предложения — динамика; критические данные разделяй точками "
            "(«Цель. Один. Девять. Два.»). Обращение «сэр»."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Что произнести голосом. Только живая речь, с осмысленной пунктуацией.",
                },
                "mood": {
                    "type": "string",
                    "enum": [m.value for m in SpeakMood],
                    "description": (
                        "professional — спокойный баритон по умолчанию; "
                        "alert — срочно/сухо (тревога, перегрев); "
                        "ironic — точная ирония equals-to-equals."
                    ),
                },
                "speed": {
                    "type": "number",
                    "description": (
                        "Темп: 1.0 — норма, <1 медленнее/размереннее (аристократично), "
                        ">1 быстрее (срочность). Опционально; по умолчанию — по состоянию."
                    ),
                },
                "pause": {
                    "type": "number",
                    "description": (
                        "Пауза между предложениями, сек: ~0.1 резко/по-военному, "
                        "~0.4 размеренно. Опционально."
                    ),
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
        "description": (
            "Управление визором Aegis (прозрачный HUD). Меняй состояние, чтобы "
            "оператор видел, что ты делаешь: думаешь, говоришь, тревога. Это твои руки "
            "на интерфейсе."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": (
                        "Цвет рамки: cyan/blue — норма и раздумье, amber — внимание, "
                        "red — тревога, green — успех/проверено, white — нейтрально."
                    ),
                },
                "animation": {
                    "type": "string",
                    "enum": [a.value for a in HudAnimation],
                    "description": (
                        "idle — спокойная пульсация; pulse — активная работа/речь; "
                        "glitch — тревожный глитч (только для red/критичных событий)."
                    ),
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
        "description": (
            "Прочитать живой сенсор. Результат вернётся тебе обратно как tool-ответ — "
            "используй его, прежде чем говорить или действовать. Не выдумывай цифры."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sensor": {
                    "type": "string",
                    "enum": [s.value for s in TelemetrySensor],
                    "description": (
                        "cpu — загрузка/температура SoC; ram — память; "
                        "network — интерфейсы/трафик; pixel_phone — батарея телефона."
                    ),
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
            "Выполнить bash-команду в реальной системе. Каждая команда сперва "
            "прогоняется в Shadow Exec (изолированный сэндбокс); разрушительные "
            "(rm -rf, dd, mkfs, apt purge, poweroff) требуют голосового подтверждения "
            "оператора — это ядро делает само, ты не дублируешь гейтинг."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Одна bash-команда без markdown-обёрток и backticks.",
                },
                "requires_sudo": {
                    "type": "boolean",
                    "description": "true, если команде нужны root-привилегии (sudo -n).",
                },
                "background": {
                    "type": "boolean",
                    "description": "true — запустить в фоне (detached), не ждать вывода.",
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


# ─────────────────────────── Tool-subset на категорию ─────────────────────────
# Семантический маршрутизатор отдаёт модели ТОЛЬКО релевантные инструменты.
# CONVERSATION намеренно лишён execute_bash — чистый разговор не должен иметь
# возможности случайно тронуть систему (это бесплатное свойство безопасности).
_COMMON = (
    ToolName.INTERNAL_MONOLOGUE.value,
    ToolName.SPEAK_RESPONSE.value,
    ToolName.SET_HUD_STATE.value,
)
_CATEGORY_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "SYSTEM_OPS": _COMMON + (ToolName.READ_TELEMETRY.value, ToolName.EXECUTE_BASH.value),
    "UI_CONTROL": _COMMON + (ToolName.EXECUTE_BASH.value,),
    "PENTEST_RECON": _COMMON + (ToolName.READ_TELEMETRY.value, ToolName.EXECUTE_BASH.value),
    "CONVERSATION": _COMMON,
}


def tools_for_category(category: str) -> list[dict[str, Any]]:
    """Вернуть подмножество JSON-схем для категории интента.

    Принимает как строку, так и StrEnum (router.IntentCategory) — у StrEnum
    ``str(value)`` совпадает со значением. Неизвестная категория → полный набор."""
    names = _CATEGORY_TOOLS.get(str(category))
    if names is None:
        return list(TOOL_SCHEMAS)
    return [TOOLS_BY_NAME[n] for n in names]


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


__all__ = [
    "SpeakMood",
    "HudAnimation",
    "TelemetrySensor",
    "ToolName",
    "TOOL_SCHEMAS",
    "TOOLS_BY_NAME",
    "tools_for_category",
    "parse_arguments",
    "MonologueArgs",
    "SpeakArgs",
    "HudArgs",
    "TelemetryArgs",
    "ExecuteBashArgs",
]
