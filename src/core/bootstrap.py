from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time

import qasync

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
from src.ui.hud import JarvisHUD, is_wayland, layer_shell_available
from src.ui.window_manager import KWinOrchestrator
from src.core.entry_point import JarvisMain
from src.memory.engine import ChronoMemory
from src.memory.storage import Mnemosyne
from src.ui.pixel_renderer import PixelBridge
from src.services.recon_daemon import ReconDaemon
from src.services.sentinel import Sentinel
from src.inference.embeddings import (
    DEFAULT_EMBED_ENDPOINT,
    DEFAULT_EMBED_MODEL,
    EmbeddingClient,
)

log = logging.getLogger("jarvis.boot")

HUD_REPOSITION_PERIOD_SEC = 6.0


async def verify_inference_stack(jarvis: Jarvis) -> None:
    """Двухслойный preflight перед стартом подсистем:

    1. **llama-server (HTTP, :8080)** — главный мозг (Qwen2.5-3B-Instruct,
       Q4_K_M) в ОТДЕЛЬНОМ системном процессе. Если демон не отвечает по
       /health, Jarvis уходит в degraded-режим: голосовые команды слышим, но
       думать нечем — оператору внятно говорим, что запустить (./setup_server.sh).
    2. **embed-сервер (llama-server --embedding, :8090)** — эмбеддинги для
       ChronoMemory (вместо Ollama). Если лежит — память деградирует до
       zero-recall, но это не fatal: Sentinel и Pixel-bridge всё равно живы.

    Никакая из проблем не валит boot — все диагностики идут через ``jarvis.say``,
    чтобы оператор услышал ровно один краткий брифинг."""
    issues: list[str] = []

    # — Layer 1: нативный llama-server (OpenAI REST). Веса грузит ОН, не Python.
    if not await jarvis._llm.health():
        issues.append(
            f"llama-server недоступен на {LLM_ENDPOINT} ({LLAMA_MODEL_NAME}) — "
            "запустите ./setup_server.sh или: systemctl --user start jarvis-llm"
        )

    # — Layer 2: embed-сервер (llama-server --embedding). Sync-клиент гоняем в
    #   треде, чтобы не блокировать event-loop на время сетевой пробы.
    embed = EmbeddingClient()
    try:
        if not await asyncio.to_thread(embed.health):
            issues.append(
                f"embed-сервер недоступен на {DEFAULT_EMBED_ENDPOINT} "
                f"({DEFAULT_EMBED_MODEL}) — память без recall. Запустите "
                "./setup_embed_server.sh или: systemctl --user start jarvis-embed"
            )
        else:
            # Жив — но реально ли отдаёт вектор? Быстрая проба одним эмбеддингом.
            try:
                vec = await asyncio.to_thread(embed.embed, ["проверка связи"])
                if not vec or not vec[0]:
                    issues.append("embed-сервер ответил пустым вектором — проверьте модель")
            except Exception as exc:
                issues.append(f"embed-сервер не отдал вектор: {exc}")
    finally:
        embed.close()

    if issues:
        joined = " | ".join(issues)
        log.warning("preflight issues: %s", joined)
        jarvis.say(
            "Сэр, обнаружены проблемы при инициализации мозга. " + joined + ".",
            tone="alert",
        )
    else:
        log.info(
            "inference stack OK: llama-server (%s) @ %s, embeddings @ %s (%s)",
            LLAMA_MODEL_NAME, LLM_ENDPOINT, DEFAULT_EMBED_ENDPOINT, DEFAULT_EMBED_MODEL,
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
        if kwin._qdbus is None:
            log.info("initial HUD pin: skipped (нет qdbus6/qdbus)")
            return
        ok = await kwin.pin_jarvis_hud()
        # False при наличии qdbus — окно ещё не зарегистрировано или плагин
        # залип; периодический HUD layout watcher до-пинит позже. Не врём про qdbus.
        log.info("initial HUD pin: %s", "OK" if ok else "отложено (окно ещё не готово)")
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

    # Runtime-верификация LLM-стека (chat-сервер :8080 + embed-сервер :8090).
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
    jarvis.start_screen_watch_loop()  # proactive: заметить ошибку на экране → предложить помощь
    jarvis.start_reflection_loop()   # nightly memory consolidation → core facts
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

    # Wayland-родной HUD: если сессия Wayland И плагин layer-shell реально
    # установлен — включаем QtWayland shell-integration ДО QApplication. Проверка
    # наличия плагина обязательна: без неё Qt не найдёт интеграцию и HUD не
    # стартует. Не на Wayland / нет плагина → молча идём прежним путём.
    if (
        is_wayland()
        and layer_shell_available()
        and not os.environ.get("QT_WAYLAND_SHELL_INTEGRATION")
    ):
        os.environ["QT_WAYLAND_SHELL_INTEGRATION"] = "layer-shell"
        logging.getLogger("jarvis.bootstrap").info(
            "layer-shell integration enabled (Wayland-native HUD)"
        )

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
