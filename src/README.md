# Jarvis Source Code Organization

Исходный код Jarvis организован по функциональным модулям:

## Структура директорий

```
src/
├── core/              # Основная орхестрация и точка входа
│   ├── orchestrator.py    # Главный движок Jarvis (обработка команд, LLM)
│   ├── entry_point.py     # Vosk слушатель и голосовой ввод
│   └── bootstrap.py       # Инициализация и запуск системы
│
├── services/          # Фоновые сервисы и демоны
│   ├── daemon_swarm.py    # Наблюдатели системных событий
│   ├── recon_daemon.py    # Разведка сети и системы
│   ├── sentinel.py        # Безопасность и мониторинг
│   └── watch_service.py   # Глубокое наблюдение за событиями
│
├── memory/            # Управление памятью и состоянием
│   ├── engine.py          # Хранилище дол Kronosгосрочной памяти
│   ├── storage.py         # Векторная база данных (Mnemosyne)
│   ├── ephemeral.py       # Временные данные и выполнение кода
│   └── snapshot.py        # Снимки состояния системы
│
├── ui/                # Пользовательский интерфейс
│   ├── hud.py             # HUD и визуализация (PyQt6)
│   ├── window_manager.py   # Управление окнами KWin
│   └── pixel_renderer.py   # Рендеринг пикселей для дисплея
│
├── audio/             # Обработка аудио
│   └── fft_analyzer.py    # FFT анализ и Piper TTS
│
├── network/           # Сетевые утилиты
│   └── scanner.py         # Сканирование сети (nmap)
│
├── security/          # Безопасность и выполнение
│   └── execution.py       # Shadow execution и sandboxing
│
└── common/            # Общие утилиты
    ├── event_bus.py       # Шина событий
    ├── parser.py          # Парсинг ответов и bash
    ├── singleton.py       # Singleton паттерн
    └── repair.py          # Быстрое исправление ошибок
```

## Соглашения по именованию

- **Модули**: snake_case (`orchestrator.py`, `window_manager.py`)
- **Классы**: PascalCase (`JarvisMain`, `DaemonSwarm`, `KWinOrchestrator`)
- **Функции**: snake_case (`compute_bands()`, `stream_nmap()`)
- **Константы**: UPPER_CASE (`LLAMA_MODEL_PATH`, `WEAVER_MEMORY_FACTS`)

## Импорты

После реоргнизации все импорты используют полные пути:

```python
# Правильно ✓
from src.core.orchestrator import Jarvis
from src.common.event_bus import EventBus, EventType
from src.memory.engine import ChronoMemory

# Неправильно ✗ (больше не работает)
from core import Jarvis
from event_bus import EventBus
```

## Как запустить

```bash
# Запуск основного приложения
python -m src.core.bootstrap

# Запуск тестов
pytest tests/ -v

# Линтинг
ruff check .
```

## Голос Джарвиса (TTS post-processing)

Синтез речи: **Piper** (`piper/ru_RU-dmitry-medium.onnx`) → raw PCM →
цепочка эффектов **sox** → `aplay`. Цепочки заданы в
`src/core/orchestrator.py::SOX_PROFILES`.

Цель DSP — приблизить нейтральный TTS к **кино-Джарвису** (Пол Беттани):
тёплый гладкий британский баритон, ровная подача, лёгкая «хай-фай»-комната.
Сознательно **без** фейзера и «шиммера» на верхах — они звучат робото-/сай-фай,
прочь от оригинала. Профили по состоянию:

| Профиль         | Когда                                  | Характер                          |
|-----------------|----------------------------------------|-----------------------------------|
| `normal`        | обычный ответ                          | тёплый баритон, тонкая комната    |
| `idle`          | IDLE state                             | глубже, медленнее, просторнее     |
| `alert`         | ALERT / destructive gate               | сухой, собранный, «впереди»       |
| `night_stealth` | темно (Pixel-сенсор) или ≥23ч / ≤6ч    | тёмный, тёплый, тихий             |

**Потолок метода (важно):** sox формирует тембр/стиль/пространство, но НЕ
переносит саму идентичность голоса актёра — спикер под капотом остаётся
русский мужской Piper. Для точного клонирования тембра Беттани нужен слой
**voice conversion** (RVC / so-vits-svc), обученный на сэмплах JARVIS, между
Piper и sox. Параметры цепочки можно валидировать на синтетическом тоне:

```bash
sox -n -t raw -r 22050 -e signed -b 16 -c 1 /tmp/t.raw synth 1 sine 130 sine 260
sox -t raw -r 22050 -e signed -b 16 -c 1 /tmp/t.raw -d <эффекты-профиля>
```

> Примечание: при частоте 22050 Гц частота Найквиста = 11025 Гц, поэтому
> shelf/eq выше ~11 кГц бессмысленны (теплота строится на низах и гладкости).

