from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time

import qasync

try:
    import ollama as _ollama_mod
    _HAS_OLLAMA = True
except ImportError:
    _ollama_mod = None  # type: ignore[assignment]
    _HAS_OLLAMA = False

from PyQt6.QtWidgets import QApplication

from src.core.orchestrator import (
    LLM_ENDPOINT,
    LLAMA_MODEL_NAME,
    Jarvis,
)
from src.core.immortal import FileChangeWatcher
from src.services.daemon_swarm import DaemonSwarm
from src.services.watch_service import DeepWatch
from src.common.event_bus import EventBus, EventType
from src.ui.hud import JarvisHUD
from src.ui.window_manager import KWinOrchestrator
from src.core.entry_point import JarvisMain
from src.memory.engine import ChronoMemory
from src.memory.storage import Mnemosyne
from src.ui.pixel_renderer import PixelBridge
from src.services.recon_daemon import ReconDaemon
from src.services.sentinel import Sentinel

log = logging.getLogger("jarvis.boot")

HUD_REPOSITION_PERIOD_SEC = 6.0

# Эмбеддинги для ChronoMemory всё ещё на Ollama — миграция all-minilm на
# llama-cpp требует отдельной GGUF embedding-модели и переработки
# memory_engine.OllamaEmbedding (следующая фаза). LLM-мозг — уже llama-cpp.
REQUIRED_OLLAMA_EMBEDDING_MODELS: tuple[str, ...] = ("all-minilm",)


async def verify_inference_stack(jarvis: Jarvis) -> None:
    """Двухслойный preflight перед стартом подсистем:

    1. **llama-server (HTTP)** — главный мозг (Llama-3.2-3B-Instruct, Q4_K_M)
       в ОТДЕЛЬНОМ системном процессе. Если демон не отвечает по /health,
       Jarvis уходит в degraded-режим: голосовые команды слышим, но думать
       нечем — оператору внятно говорим, что запустить (./setup_server.sh).
    2. **Ollama + all-minilm** — эмбеддинги для ChronoMemory. Если ollama
       лежит, память деградирует до zero-recall, но это не fatal — Sentinel
       и Pixel-bridge всё равно живы.

    Никакая из проблем не валит boot — все диагностики идут через ``jarvis.say``,
    чтобы оператор услышал ровно один краткий брифинг."""
    issues: list[str] = []

    # — Layer 1: нативный llama-server (OpenAI REST). Веса грузит ОН, не Python.
    if not await jarvis._llm.health():
        issues.append(
            f"llama-server недоступен на {LLM_ENDPOINT} ({LLAMA_MODEL_NAME}) — "
            "запустите ./setup_server.sh или: systemctl --user start jarvis-llm"
        )

    # — Layer 2: Ollama embeddings (all-minilm).
    if not _HAS_OLLAMA:
        issues.append("ollama Python-пакет не установлен (нужен для эмбеддингов памяти): pip install ollama")
    else:
        try:
            client = _ollama_mod.AsyncClient()
            listing = await client.list()
        except Exception as exc:
            log.warning("ollama list() failed: %s", exc)
            issues.append(
                "Ollama не отвечает (нужна для эмбеддингов памяти): "
                "systemctl status ollama"
            )
            listing = None

        if listing is not None:
            raw_models = getattr(listing, "models", None)
            if raw_models is None and isinstance(listing, dict):
                raw_models = listing.get("models", [])
            raw_models = raw_models or []

            installed: set[str] = set()
            for m in raw_models:
                name = (
                    getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or (m.get("model") if isinstance(m, dict) else None)
                    or (m.get("name") if isinstance(m, dict) else None)
                    or ""
                )
                if name:
                    installed.add(str(name))
                    if ":" not in name:
                        installed.add(f"{name}:latest")

            def _have(model: str) -> bool:
                if model in installed:
                    return True
                return f"{model}:latest" in installed or any(
                    mi.startswith(f"{model}:") for mi in installed
                )

            missing = [m for m in REQUIRED_OLLAMA_EMBEDDING_MODELS if not _have(m)]
            if missing:
                cmds = ", ".join(f"ollama pull {m}" for m in missing)
                issues.append(f"эмбеддинги не скачаны — выполните: {cmds}")

    if issues:
        joined = " | ".join(issues)
        log.warning("preflight issues: %s", joined)
        jarvis.say(
            "Сэр, обнаружены проблемы при инициализации мозга. " + joined + ".",
            tone="alert",
        )
    else:
        log.info(
            "inference stack OK: llama-server (%s) @ %s, embeddings via ollama all-minilm",
            LLAMA_MODEL_NAME, LLM_ENDPOINT,
        )


def _sd_notify(msg: bytes) -> None:
    """Write a systemd sd_notify message if NOTIFY_SOCKET is set."""
    sock_path = os.environ.get("NOTIFY_SOCKET", "")
    if not sock_path:
        return
    try:
        addr = "\0" + sock_path[1:] if sock_path.startswith("@") else sock_path
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg)
    except Exception:
        pass


async def _sd_watchdog_loop() -> None:
    """Pings systemd watchdog every 10s (WatchdogSec=30 in the unit file).

    If the event loop hangs completely systemd restarts the service after 30s."""
    _sd_notify(b"READY=1")
    while True:
        _sd_notify(b"WATCHDOG=1")
        await asyncio.sleep(10)


async def _boot_greeting(jarvis: Jarvis) -> None:
    """Speaks a brief status 4s after all subsystems have started."""
    await asyncio.sleep(4.0)
    hour = time.localtime().tm_hour
    if 5 <= hour < 12:
        greeting = "Доброе утро"
    elif 12 <= hour < 18:
        greeting = "Добрый день"
    elif 18 <= hour < 23:
        greeting = "Добрый вечер"
    else:
        greeting = "Ночная смена"

    name_hint = ""
    try:
        import re
        hits = await asyncio.to_thread(jarvis.memory.recall, "user name operator")
        for h in hits:
            text = h.get("text", "") if isinstance(h, dict) else str(h)
            m = re.search(r"(?:меня зовут|operator[:\s]+|user[:\s]+)([A-Za-zА-Яа-яЁё]+)", text, re.I)
            if m:
                name_hint = f", {m.group(1)}"
                break
    except Exception:
        pass

    jarvis.say(
        f"{greeting}{name_hint}. Jarvis онлайн. Все системы инициализированы.",
        tone="normal",
    )


async def hud_layout_watcher(bus: EventBus, kwin: KWinOrchestrator) -> None:
    """Periodically pin the HUD window (keepAbove/skipTaskbar/skipPager/skipSwitcher).
    KWin под Wayland сбрасывает эти флаги при смене workspace / выходе из fullscreen
    приложений, поэтому heartbeat обязателен.

    NOTE: HUD position is strictly locked to native position (0.88, 0.86) — no repositioning."""
    log.info("HUD layout watcher started")
    while True:
        try:
            await kwin.pin_jarvis_hud()
        except Exception:
            log.exception("HUD pin failed")
        await asyncio.sleep(HUD_REPOSITION_PERIOD_SEC)


async def initial_hud_pin(kwin: KWinOrchestrator) -> None:
    """Первый pin HUD'а — отложен на 1 с, чтобы KWin успел зарегистрировать
    окно после ``hud.show_fullscreen()``. Без задержки workspace.windowList()
    его ещё не видит."""
    await asyncio.sleep(1.0)
    try:
        ok = await kwin.pin_jarvis_hud()
        log.info("initial HUD pin: %s", "OK" if ok else "skipped (qdbus unavailable)")
    except Exception:
        log.exception("initial HUD pin failed")


async def _prewarm_inference(jarvis: Jarvis) -> None:
    """Прогреть llama-server в фоне, пока Jarvis договаривает приветствие.

    Сам сервер — отдельный systemd-юнит (jarvis-llm.service), грузящий веса в
    iGPU независимо от нас. Здесь: (1) ждём готовности /health (до ~180с —
    холодная загрузка + компиляция Vulkan-шейдеров), затем (2) делаем ОДИН
    крошечный запрос (Jarvis.warmup) — он компилирует шейдеры и греет KV-префикс,
    чтобы ПЕРВАЯ реальная голосовая команда была мгновенной, а не ждала холодный
    старт. Любые ошибки глушим — реальный запрос всё равно переподнимет всё."""
    try:
        await asyncio.sleep(2.0)
        ready = False
        for _ in range(90):                 # 90 × 2с = 180с на холодный старт
            if await jarvis._llm.health():
                ready = True
                break
            await asyncio.sleep(2.0)
        if not ready:
            log.warning("llama-server не готов за 180с — прогрев пропущен")
            return
        if await jarvis.warmup():
            log.info("llama-server прогрет (Vulkan-шейдеры + KV-префикс)")
    except Exception:
        log.debug("inference prewarm skipped", exc_info=True)


async def amain() -> None:
    # Веса модели НЕ грузятся в этот процесс. Инференс — нативный llama-server
    # (jarvis-llm.service), который:
    #  (1) держит KV-кэш системного промпта в VRAM (--cache-reuse) → низкий TTFT;
    #  (2) изолирует нативные краши бэкенда (Vulkan/GGML_ASSERT) — падает демон,
    #      а не Jarvis; systemd (Restart=always) поднимает его за 2с;
    #  (3) переживает hot-reload Jarvis (os.execv), оставаясь тёплым.
    # Мы общаемся с ним по HTTP (LlamaServerClient, httpx) — неблокирующе, так
    # что HUD/шина/голос не замирают на раздумьях модели.

    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())

    hud = JarvisHUD()
    hud.show_fullscreen()

    jarvis = Jarvis()
    kwin = KWinOrchestrator()
    hud.subscribe_to_bus(bus)

    # Runtime-верификация LLM-стека (inference server + ollama embeddings).
    # Если чего-то нет — Джарвис скажет голосом и продолжит запуск без падения.
    await verify_inference_stack(jarvis)

    bus.subscribe(EventType.VOICE_INTENT, jarvis.on_voice_intent)
    bus.subscribe(EventType.DAEMON_ALERT, jarvis.on_daemon_alert)
    bus.subscribe(EventType.OS_EVENT, jarvis.on_os_event)
    bus.subscribe(EventType.PIXEL_EVENT, jarvis.on_pixel_event)
    bus.subscribe(EventType.RECON_ALERT, jarvis.on_recon_alert)
    bus.subscribe(EventType.DEEP_WATCH, jarvis.on_deep_watch)
    bus.subscribe(EventType.SYSTEM_WAKE, jarvis.on_system_wake)

    asyncio.create_task(bus.run(), name="event-bus")
    asyncio.create_task(_sd_watchdog_loop(), name="sd-watchdog")
    await DaemonSwarm().start_all()
    await Sentinel().start_all()
    await ReconDaemon().start_all()
    await PixelBridge().start()
    await DeepWatch().start()
    asyncio.create_task(hud_layout_watcher(bus, kwin), name="hud-layout")
    # KWin pin: первый — после небольшой задержки (окно должно зарегаться
    # в compositor), затем heartbeat внутри hud_layout_watcher.
    asyncio.create_task(initial_hud_pin(kwin), name="hud-pin-initial")

    # Mnemosyne: the chrono-brain harvester. Pulls the active window every
    # 10 seconds, watches clipboard for IP/CVE/code, autonomously RAGs old
    # findings and asks the LLM to think about what just got copied.
    memory = ChronoMemory()
    await memory.start_gc()  # 4-day TTL sweeper on the rolling tier

    # Mnemosyne берёт ШАРЕДНЫЙ llama-cpp инстанс из Jarvis через адаптер —
    # одна модель в RAM, ноль HTTP-походов в Ollama для clipboard-thinker.
    mnemo = Mnemosyne(
        bus=bus,
        memory=memory,
        kwin=kwin,
        llm_client=jarvis.llm_adapter,
    )
    mnemo.start()

    jarvis.start_proactive_loop()
    asyncio.create_task(_boot_greeting(jarvis), name="boot-greeting")
    asyncio.create_task(_prewarm_inference(jarvis), name="llm-prewarm")

    voice = JarvisMain()
    # Голосовой тред запускается через watchdog-обёртку: если run() завершится
    # (краш sounddevice, Vosk ошибка) — тред перезапустится через 5 секунд.
    def _voice_watchdog() -> None:
        import time as _time
        while True:
            try:
                voice.run(bus)
            except Exception:
                log.exception("voice watchdog: run() crashed")
            _time.sleep(5.0)
            log.info("voice watchdog: restarting voice loop")

    threading.Thread(target=_voice_watchdog, daemon=True, name="voice-watchdog").start()

    print("💎 JARVIS Protocol Fully Deployed.")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    # Порядок инициализации прозрачности под Wayland (ТЗ оператора):
    # 1) QSurfaceFormat с alpha-каналом ставится ДО QApplication, иначе
    #    дефолтный 0-bit alpha сделает HUD непрозрачным независимо от
    #    WA_TranslucentBackground.
    # 2) Затем QApplication.
    # 3) WA_TranslucentBackground / WA_AlwaysStackOnTop / Tool — внутри
    #    JarvisHUD.__init__, ДО создания GL-дочек.

    app = QApplication(sys.argv)
    # KWin script ищет HUD по caption ("JarvisHUD") и resourceClass
    # ("jarvis-hud") — ставим оба, чтобы pin_jarvis_hud сработал детерминированно.
    app.setApplicationName("jarvis-hud")
    app.setApplicationDisplayName("JarvisHUD")
    app.setQuitOnLastWindowClosed(False)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        # Бессмертие, слой 1: горячая перезагрузка кода. Watcher следит за
        # src/ и config/ — при правке файлов или git pull процесс делает
        # os.execv и поднимается со свежим кодом (тот же PID), без ручного
        # перезапуска. Слой 2 (переживание крашей/убийства) — внешний
        # супервизор: python -m src.core.supervisor.
        file_watcher = FileChangeWatcher()
        loop.create_task(file_watcher.start(), name="file-watcher")

        loop.create_task(amain(), name="amain")
        loop.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
