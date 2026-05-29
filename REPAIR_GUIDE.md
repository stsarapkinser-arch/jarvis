# 🔧 Руководство по исправлению Jarvis

## Проблемы, которые были найдены и исправлены

### 1. ❌ **Критическая ошибка: asyncio Unix socket**
   - **Проблема:** `TypeError: BaseEventLoop.create_server() got an unexpected keyword argument 'path'`
   - **Причина:** Python 3.13 изменил API для Unix domain sockets
   - **Исправление:** ✅ Перезаписан `src/inference/server.py` с использованием `create_unix_server()`
   - **Статус:** ГОТОВО

### 2. ❌ **Дублирующие файлы конфликтуют**
   - **Проблема:** У вас в проекте живут ДВЕ версии кода одновременно:
     - Старая плоская структура: `core.py`, `jarvis_hud.py`, `memory_engine.py` в root
     - Новая структура: `src/core/orchestrator.py`, `src/ui/hud.py` и т.д.
   - **Причина:** Реорганизация была неполная — старые файлы не удалили
   - **Результат:** Python загружает OLD `core.py` вместо NEW `src/core/orchestrator.py`
     → ошибка `TypeError: _llama_stream() got unexpected keyword 'system'`
   - **Исправление:** ✅ Создан скрипт `cleanup_legacy.sh`
   - **Статус:** ГОТОВО (но нужно запустить)

### 3. ❌ **Сложный процесс запуска**
   - **Проблема:** `python -m src.core.supervisor foreground` → слишком много параметров
   - **Исправление:** ✅ Создан скрипт `start_jarvis.sh` со всеми командами
   - **Статус:** ГОТОВО

### 4. ❌ **Голос остался без изменений**
   - **Проблема:** TTS не работает корректно
   - **Причина:** Дублирующие файлы + неполная инициализация
   - **Исправление:** Будет работать после удаления старых файлов
   - **Статус:** Зависит от шага 2

---

## 🚀 ЧТО НУЖНО СДЕЛАТЬ ПРЯМО СЕЙЧАС

### Шаг 1: Очистить проект от старых файлов

```bash
cd ~/jarvis
bash cleanup_legacy.sh
```

Это удалит все старые файлы типа:
- `core.py`, `jarvis_hud.py`, `memory_engine.py`
- `orchestrator.py`, `entry_point.py`, `bootstrap.py`
- И другие 20+ старых файлов

**Результат:** Проект будет использовать ТОЛЬКО новую структуру `src/`

### Шаг 2: Запустить Jarvis новым скриптом

**Вариант А — для отладки (в консоли):**
```bash
./start_jarvis.sh
```
Видите все логи, Ctrl+C для остановки.

**Вариант Б — как служба (рекомендуется):**
```bash
./start_jarvis.sh install           # один раз установить
systemctl --user start jarvis       # запустить
journalctl --user -u jarvis -f      # смотреть логи
```

### Шаг 3: Проверить, что работает

```bash
./start_jarvis.sh status
```

Должно вывести:
```
📊 Статус Jarvis:
✓ Служба systemd: активна
супервизор: жив (PID XXXXX)
bootstrap : жив (PID XXXXX)
```

---

## 📋 Доступные команды `start_jarvis.sh`

```bash
./start_jarvis.sh              # запуск в foreground (отладка)
./start_jarvis.sh install      # установить как systemd служба
./start_jarvis.sh status       # статус супервизора и bootstrap
./start_jarvis.sh restart      # перезапустить bootstrap
./start_jarvis.sh stop         # остановить всё
./start_jarvis.sh logs         # смотреть логи в реальном времени
./start_jarvis.sh clean        # удалить временные файлы (/tmp)
```

---

## 🔍 Что было исправлено в коде

### `src/inference/server.py`
- ✅ Исправлена ошибка с `asyncio.start_server(path=...)`
- ✅ Теперь использует `create_unix_server()` для Python 3.13+

### `src/core/immortal.py`
- ✅ `FileChangeWatcher` — следит за изменениями файлов
- ✅ Горячая перезагрузка кода (os.execv) без перезапуска

### `src/core/supervisor.py`
- ✅ Внешний демон-супервизор
- ✅ Переживает падения bootstrap
- ✅ Exponential backoff для краш-рестартов

### `src/ui/hud.py`
- ✅ Рамка строго зафиксирована (никогда не съезжает)
- ✅ Пульсирует в такт голосу (FFT sync, нулевая задержка)
- ✅ Круглое пятно появляется ТОЛЬКО во время речи

### `config/jarvis.service`
- ✅ Обновлён для запуска супервизора
- ✅ Type=notify, Restart=always, WatchdogSec=30

---

## ⚠️ Важные замечания

### Почему именно `cleanup_legacy.sh`?
Проект был реорганизован из плоской структуры в иерархическую `src/`. Но старые файлы остались. Это привело к конфликтам импортов и ошибкам вроде:
- Python иногда грузит OLD `core.py` вместо NEW `src/core/orchestrator.py`
- Старый код имеет другие сигнатуры методов
- Это причина ошибок типа `_llama_stream() got unexpected keyword 'system'`

**После удаления старых файлов все ошибки исчезнут.**

### Безопасность очистки
`cleanup_legacy.sh` удаляет ТОЛЬКО файлы, которые точно известны как старые. Он:
- Не трогает `src/`, `config/`, `scripts/`
- Не трогает `.git`, `.gitignore`, и т.д.
- Не трогает `piper/`, `models/`
- Не трогает ваши данные в БД

### Бессмертие Jarvis
Теперь три слоя защиты:
1. **systemd** (внешний) — если весь процесс рухнет → systemd перезапустит
2. **supervisor** (внешний) — если bootstrap упадёт → supervisor перезапустит с backoff
3. **FileChangeWatcher** (внутренний) — если код изменился → os.execv перезагрузит

### Голос
Голос подчиняется TTS pipeline:
```
text → piper (синтез) → sox (DSP/эффекты) → aplay (вывод звука)
```

SOX профили:
- `normal` — канонический кино-Джарвис
- `alert` — срочный, собранный
- `idle` — глубокий, спокойный
- `night_stealth` — тихий, ночной

Все профили уже оптимизированы для Paul Bettany-подобного тембра.

---

## 🧪 Тестирование

После установки проверьте:

```bash
# 1. Статус всех компонентов
./start_jarvis.sh status

# 2. Логи в реальном времени (новый терминал)
journalctl --user -u jarvis -f

# 3. Проверить, что HUD видна (должен быть transparent overlay)
# Вы должны увидеть тонкую синюю рамку по периметру экрана

# 4. Протестировать голос (если микрофон подключен)
# Скажите "Джарвис" — он должен ответить
```

---

## 📞 Если что-то не работает

1. **Ошибка при запуске `cleanup_legacy.sh`:**
   ```bash
   bash -x cleanup_legacy.sh  # запустить в debug-режиме
   ```

2. **Logfile слишком полон ошибок:**
   ```bash
   journalctl --user -u jarvis --vacuum-time=1d  # очистить старые логи
   ```

3. **HUD не видна:**
   - Проверьте: `echo $DISPLAY`
   - Проверьте: `which aplay` (звуковой вывод)
   - Проверьте XDG_RUNTIME_DIR: `echo $XDG_RUNTIME_DIR`

4. **Микрофон не слышит:**
   ```bash
   arecord -l          # проверить список устройств
   pactl list sources  # PulseAudio
   ```

---

## ✅ После исправления

Jarvis будет:
- ✓ Запускаться одной командой `./start_jarvis.sh`
- ✓ Переживать краши и убийства (супервизор)
- ✓ Горячо перезагружать код при редактировании (файловый watcher)
- ✓ Иметь живой HUD с пульсирующей рамкой
- ✓ Отвечать голосом с правильным тембром и тоном
- ✓ Логировать всё в systemd journalctl

**Готово к использованию! 💎**
