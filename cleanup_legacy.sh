#!/usr/bin/env bash
# Удаляет все СТАРЫЕ файлы из плоской структуры, которые конфликтуют с reorganized src/
# ВАЖНО: запустить один раз после git pull новой версии!

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "🧹 Очистка legacy-файлов (старая плоская структура)..."

# Список файлов из старой плоской структуры, которые нужно удалить:
OLD_FILES=(
    "core.py"
    "jarvis_hud.py"
    "entry_point.py"
    "orchestrator.py"
    "bootstrap.py"
    "memory_engine.py"
    "ephemeral.py"
    "event_bus.py"
    "parser.py"
    "repair.py"
    "singleton.py"
    "fft_analyzer.py"
    "shadow_exec.py"
    "execution.py"
    "scanner.py"
    "pixel.py"
    "kwin.py"
    "window_manager.py"
    "pixel_renderer.py"
    "daemon_swarm.py"
    "deep_watch.py"
    "recon_daemon.py"
    "sentinel.py"
    "watch_service.py"
    "mnemosyne.py"
    "snapshot.py"
    "storage.py"
    "nmap_stream.py"
    "disable_jarvis.sh"
    "enable_jarvis.sh"
    "main.py"
)

deleted_count=0
for file in "${OLD_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✗ Удаляю $file"
        rm -f "$file"
        ((deleted_count++))
    fi
done

# Удалить старые .pyc и __pycache__
find . -maxdepth 1 -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
find . -maxdepth 1 -name "*.pyc" -delete

# Очистить старые служебные файлы
rm -f .*.swp *.egg-info 2>/dev/null || true

echo "✓ Удалено $deleted_count файлов"
echo ""
echo "✓ Очистка завершена!"
echo "  Проект готов к работе с новой структурой src/"
echo ""
echo "Следующий шаг:"
echo "  ./start_jarvis.sh"
