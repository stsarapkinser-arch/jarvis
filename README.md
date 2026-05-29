# Jarvis

Автономный AI-ассистент для Kali Linux (KDE Plasma 6, Wayland, Intel N100).

## Архитектура

```
start_jarvis.py          — точка входа: QApplication + qasync event loop
├── core.py              — Jarvis: LLM-оркестровка, context weaver, Shadow Exec gate
├── jarvis_hud.py        — чистый QPainter HUD (1px рамка, 2D-сфера, WhisperLine)
├── event_bus.py         — EventBus singleton, SystemState, все EventType
├── sentinel.py          — системные ватчеры (thermal, iGPU, RAM/диск, suspend, dbus)
├── main.py              — JarvisMain: Vosk-слушатель голоса (daemon thread)
├── memory_engine.py     — ChronoMemory: ChromaDB двухуровневый (core + rolling TTL)
├── shadow_exec.py       — ShadowExec: sandbox (bwrap → podman → refuse)
├── daemon_swarm.py      — DaemonSwarm: journal, qdbus, clipboard наблюдатели
├── repair.py            — QuickPatcher: regex stderr → механический bash-фикс
├── ephemeral.py         — EphemeralRunner: исполнение Python-блоков из LLM
├── state_snapshot.py    — StateSnapshot: снапшот системы для LLM-контекста
├── mnemosyne.py         — Mnemosyne: автономный харвестер буфера/окон → ChromaDB
├── recon_daemon.py      — Wraith: WiFi/сетевая разведка
├── deep_watch.py        — DeepWatch: eBPF-наблюдатель процессов/сети
├── kwin.py              — KWinOrchestrator: управление окнами (qdbus/xdotool)
├── pixel.py             — KDE Connect / Android bridge
├── audio_fft.py         — PiperFFTPump: FFT PCM-потока Piper → пульс сферы
├── nmap_stream.py       — стриминг nmap-сканирования
├── commands.py          — CommandBook: реестр аудированных команд
├── parser.py            — parse_response: разбор LLM-ответов, sandbox-обёртки
└── singleton.py         — Singleton metaclass
```

## Стек

| Компонент | Реализация |
|---|---|
| LLM | llama-cpp-python in-process, `qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| iGPU offload | Intel N100 через `setup_igpu.sh` (OpenCL/Vulkan/CPU) |
| Embeddings | Ollama `all-minilm` (только ChromaDB) |
| Voice in | Vosk (kaldi model в `model/`) |
| Voice out | Piper (`piper/piper` + `ru_RU-dmitry-medium.onnx`) |
| GUI | PyQt6 + qasync, чистый QPainter |
| Memory | ChromaDB (core + rolling коллекции) |
| Sandbox | bubblewrap → podman → refuse |

> **Важно:** OpenGL/GLSL шейдеры удалены — на Intel N100 вызывали Core Dump
> (`vk::DeviceLostError`). HUD — только QPainter.

## Запуск

```bash
# 1. Установить зависимости
./setup_igpu.sh          # собрать llama-cpp-python с iGPU backend

pip install -r requirements.txt

# 2. Запустить
python start_jarvis.py

# Или как systemd user-сервис
./install_service.sh
systemctl --user start jarvis
```

## Тесты

```bash
python -m pytest tests/ -q
```

## Разработка

```bash
# Линт
pip install ruff
ruff check .

# Pre-commit хуки (один раз)
pip install pre-commit
pre-commit install
```

CI (`.github/workflows/ci.yml`) прогоняет `ruff` и `pytest` на каждый push/PR.

## Директории (не в git)

| Путь | Содержимое |
|---|---|
| `models/` | GGUF-модели (qwen2.5-coder-3b-instruct-q4_k_m.gguf, ~2 GB) |
| `model/` | Vosk kaldi-модель (~88 MB) |
| `piper/*.onnx` | Piper TTS модель (~61 MB) |
| `jarvis_memory/` | ChromaDB данные (персистентная память) |
