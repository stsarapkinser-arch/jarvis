"""Бессмертие Jarvis, слой 2 из 2 — внешний супервизор-демон.

Запускает и стережёт ``python -m src.core.bootstrap`` как дочерний процесс.
Что бы с bootstrap ни случилось — краш, OOM-килл, ``kill``, Ctrl+C, плановая
перезагрузка кода — супервизор поднимает его заново. Сам супервизор при старте
делает double-fork и отвязывается от терминала (``setsid``), поэтому закрытие
терминала или Ctrl+C в нём НЕ убивают Джарвиса.

Именно этот слой выполняет требование оператора:
    «остановка python -m src.core.bootstrap не должна убивать Джарвиса,
     HUD или любой другой элемент».
Останавливаешь bootstrap — супервизор за ~секунду поднимает его обратно.

Команды:
    python -m src.core.supervisor            # демонизироваться и стеречь
    python -m src.core.supervisor foreground # стеречь в текущем терминале (для systemd)
    python -m src.core.supervisor status     # жив ли супервизор и bootstrap
    python -m src.core.supervisor stop        # корректно остановить всё
    python -m src.core.supervisor restart     # перезапустить bootstrap немедленно

Полностью остановить Джарвиса можно ТОЛЬКО через ``stop`` (или остановив
systemd-юнит) — это сознательно, чтобы случайный сигнал не «убил» ассистента.
"""
from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.core.immortal import RELOAD_EXIT_CODE

log = logging.getLogger("jarvis.supervisor")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Где храним PID'ы. /tmp переживает редактирование кода, чистится при ребуте.
_STATE_DIR = Path(os.environ.get("JARVIS_STATE_DIR", "/tmp"))
SUPERVISOR_PIDFILE = _STATE_DIR / "jarvis-supervisor.pid"
CHILD_PIDFILE = _STATE_DIR / "jarvis-bootstrap.pid"
SUPERVISOR_LOG = _STATE_DIR / "jarvis-supervisor.log"

# Экспоненциальный backoff для перезапуска после КРАШа (не reload).
CRASH_BACKOFF_SEC = [1, 2, 4, 8, 16, 30]
# Если ребёнок прожил дольше этого — считаем запуск удачным, сбрасываем backoff.
HEALTHY_UPTIME_SEC = 30.0

_BOOTSTRAP_CMD = [sys.executable, "-m", "src.core.bootstrap"]

# Флаг «нас просят остановиться» — выставляется обработчиком сигналов.
_stop_requested = False
# Флаг «перезапусти ребёнка немедленно» — по SIGHUP (restart-команда).
_restart_requested = False


# ───────────────────────── PID-файлы ─────────────────────────
def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _write_pid(path: Path, pid: int) -> None:
    try:
        path.write_text(str(pid))
    except OSError:
        log.warning("не смог записать pidfile %s", path)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # существует, но чужой — всё равно жив
    return True


# ───────────────────────── Демонизация ─────────────────────────
def _daemonize() -> None:
    """Классический double-fork: отвязываемся от терминала и сессии,
    чтобы закрытие терминала / Ctrl+C не доставали супервизор."""
    if os.fork() > 0:
        os._exit(0)            # первый родитель уходит
    os.setsid()                # новая сессия, нет управляющего терминала
    if os.fork() > 0:
        os._exit(0)            # второй родитель уходит — мы больше не лидер сессии
    os.chdir(str(_REPO_ROOT))
    os.umask(0)

    # Перенаправляем stdio в лог-файл — демону некуда больше писать.
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDONLY)
    logfd = os.open(str(SUPERVISOR_LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(logfd, sys.stdout.fileno())
    os.dup2(logfd, sys.stderr.fileno())
    os.close(devnull)
    os.close(logfd)


# ───────────────────────── Сигналы ─────────────────────────
def _install_signal_handlers() -> None:
    def _on_term(signum, _frame):
        global _stop_requested
        log.info("супервизор получил сигнал %s — останавливаемся", signum)
        _stop_requested = True

    def _on_hup(signum, _frame):
        global _restart_requested
        log.info("супервизор получил SIGHUP — перезапуск bootstrap")
        _restart_requested = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    signal.signal(signal.SIGHUP, _on_hup)


# ───────────────────────── Управление ребёнком ─────────────────────────
def _spawn_child() -> subprocess.Popen:
    env = os.environ.copy()
    env["JARVIS_SUPERVISED"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        _BOOTSTRAP_CMD,
        cwd=str(_REPO_ROOT),
        env=env,
        start_new_session=True,  # своя группа: сигналы терминала не долетают
    )
    _write_pid(CHILD_PIDFILE, proc.pid)
    log.info("bootstrap запущен (PID %d)", proc.pid)
    return proc


def _terminate_child(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        # Бьём всю группу процессов ребёнка (piper/sox/aplay тоже).
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        log.warning("bootstrap не завершился за 8с — SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


# ───────────────────────── Главный цикл ─────────────────────────
def _supervise() -> int:
    global _restart_requested
    _install_signal_handlers()
    _write_pid(SUPERVISOR_PIDFILE, os.getpid())
    log.info("супервизор Jarvis активен (PID %d)", os.getpid())

    backoff_idx = 0
    try:
        while not _stop_requested:
            started = time.monotonic()
            proc = _spawn_child()

            # Ждём ребёнка, периодически просыпаясь для проверки флагов.
            while True:
                if _stop_requested:
                    _terminate_child(proc)
                    return 0
                if _restart_requested:
                    _restart_requested = False
                    log.info("плановый перезапуск bootstrap по запросу")
                    _terminate_child(proc)
                    break
                try:
                    rc = proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue
                # Ребёнок сам завершился.
                uptime = time.monotonic() - started
                if rc == RELOAD_EXIT_CODE:
                    log.info("bootstrap ушёл на горячую перезагрузку — поднимаю сразу")
                    backoff_idx = 0
                elif uptime >= HEALTHY_UPTIME_SEC:
                    log.warning("bootstrap упал (rc=%s) после %.0fс аптайма — рестарт", rc, uptime)
                    backoff_idx = 0
                else:
                    delay = CRASH_BACKOFF_SEC[min(backoff_idx, len(CRASH_BACKOFF_SEC) - 1)]
                    log.warning("bootstrap упал (rc=%s) за %.1fс — рестарт через %dс",
                                rc, uptime, delay)
                    # Спим, но реагируем на stop.
                    for _ in range(delay):
                        if _stop_requested:
                            return 0
                        time.sleep(1)
                    backoff_idx += 1
                break
    finally:
        CHILD_PIDFILE.unlink(missing_ok=True)
        SUPERVISOR_PIDFILE.unlink(missing_ok=True)
        log.info("супервизор Jarvis остановлен")
    return 0


# ───────────────────────── CLI ─────────────────────────
def _cmd_status() -> int:
    sup = _read_pid(SUPERVISOR_PIDFILE)
    child = _read_pid(CHILD_PIDFILE)
    sup_ok = _pid_alive(sup)
    child_ok = _pid_alive(child)
    print(f"супервизор: {'жив' if sup_ok else 'не запущен'}"
          + (f" (PID {sup})" if sup_ok else ""))
    print(f"bootstrap : {'жив' if child_ok else 'не запущен'}"
          + (f" (PID {child})" if child_ok else ""))
    return 0 if sup_ok else 1


def _cmd_stop() -> int:
    sup = _read_pid(SUPERVISOR_PIDFILE)
    if not _pid_alive(sup):
        print("супервизор не запущен")
        SUPERVISOR_PIDFILE.unlink(missing_ok=True)
        return 1
    print(f"останавливаю супервизор (PID {sup})…")
    try:
        os.kill(sup, signal.SIGTERM)  # type: ignore[arg-type]
    except OSError as e:
        print(f"не удалось послать сигнал: {e}")
        return 1
    # Ждём, пока супервизор уберёт свой pidfile.
    for _ in range(20):
        if not _pid_alive(_read_pid(SUPERVISOR_PIDFILE)):
            print("остановлено")
            return 0
        time.sleep(0.5)
    print("супервизор не остановился вовремя")
    return 1


def _cmd_restart() -> int:
    sup = _read_pid(SUPERVISOR_PIDFILE)
    if not _pid_alive(sup):
        print("супервизор не запущен — нечего перезапускать")
        return 1
    os.kill(sup, signal.SIGHUP)  # type: ignore[arg-type]
    print("запрошен перезапуск bootstrap")
    return 0


def _already_running() -> bool:
    return _pid_alive(_read_pid(SUPERVISOR_PIDFILE))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    argv = argv if argv is not None else sys.argv[1:]
    cmd = (argv[0] if argv else "start").lower()

    if cmd == "status":
        return _cmd_status()
    if cmd == "stop":
        return _cmd_stop()
    if cmd == "restart":
        return _cmd_restart()

    if cmd in ("start", "foreground"):
        if _already_running():
            print("супервизор Jarvis уже запущен")
            return 0
        if cmd == "start":
            _daemonize()
        return _supervise()

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
