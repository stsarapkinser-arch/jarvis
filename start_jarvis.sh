#!/usr/bin/env bash
# Джарвис: максимально удобный запуск в одной команде
# Использование:
#   ./start_jarvis.sh              - запуск в foreground (для отладки)
#   ./start_jarvis.sh install      - установить как systemd служба
#   ./start_jarvis.sh status       - проверить статус
#   ./start_jarvis.sh stop         - остановить

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Проверка Python и venv
if [[ ! -f "${SCRIPT_DIR}/.venv/bin/python" ]] && [[ ! -f "${SCRIPT_DIR}/venv/bin/python" ]]; then
    echo -e "${RED}❌ venv не найден. Создаю...${NC}"
    python3 -m venv "${SCRIPT_DIR}/.venv"
    "${SCRIPT_DIR}/.venv/bin/pip" install -q -r config/requirements.txt
fi

# Выбираем существующий venv; путь всегда абсолютный — systemd требует абсолютный ExecStart
if [[ -f "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PY="$(realpath "${SCRIPT_DIR}/.venv/bin/python")"
else
    PY="$(realpath "${SCRIPT_DIR}/venv/bin/python")"
fi
SUPERVISOR_CMD="$PY -m src.core.supervisor"

# Функция: вывести справку
show_help() {
    cat <<'EOF'
Джарвис — управление

ИСПОЛЬЗОВАНИЕ:
    ./start_jarvis.sh [КОМАНДА]

КОМАНДЫ:
    (пусто)          Запустить в foreground (Ctrl+C для остановки)
    install          Установить как systemd --user служба
    status           Показать статус супервизора и bootstrap'а
    restart          Перезапустить bootstrap немедленно
    stop             Корректно остановить всё
    clean            Удалить все временные файлы и логи
    logs             Показать логи в реальном времени (systemd)

ПРИМЕРЫ:
    ./start_jarvis.sh                    # запуск в консоли (отладка)
    ./start_jarvis.sh install            # установить как служба
    systemctl --user start jarvis        # запустить через systemd
    journalctl --user -u jarvis -f       # логи
EOF
}

# ────────────────────────────────────────────────────────────

cmd="${1:-}"

case "$cmd" in
    install)
        echo -e "${YELLOW}📦 Установка systemd юнита...${NC}"
        mkdir -p ~/.config/systemd/user

        # Генерируем юнит с текущими путями
        sed \
            -e "s|__JARVIS_HOME__|${SCRIPT_DIR}|g" \
            -e "s|__JARVIS_PYTHON__|${PY}|g" \
            "config/jarvis.service" > ~/.config/systemd/user/jarvis.service

        systemctl --user daemon-reload
        systemctl --user enable jarvis

        echo -e "${GREEN}✓ Установлено!${NC}"
        echo ""
        echo "Дальше:"
        echo "  systemctl --user start jarvis"
        echo "  journalctl --user -u jarvis -f"
        ;;

    status)
        echo -e "${YELLOW}📊 Статус Jarvis:${NC}"
        if systemctl --user is-active --quiet jarvis 2>/dev/null; then
            echo -e "${GREEN}✓ Служба systemd: активна${NC}"
        else
            echo -e "${RED}✗ Служба systemd: неактивна${NC}"
        fi

        # Проверяем супервизор
        $SUPERVISOR_CMD status 2>/dev/null || echo "Супервизор не запущен"
        ;;

    restart)
        echo -e "${YELLOW}🔄 Перезапуск bootstrap...${NC}"
        if ! systemctl --user is-active --quiet jarvis 2>/dev/null; then
            echo -e "${YELLOW}⚠ Служба systemd неактивна, запускаю супервизор вручную...${NC}"
            exec $SUPERVISOR_CMD restart
        else
            systemctl --user restart jarvis
            echo -e "${GREEN}✓ Перезагружено${NC}"
        fi
        ;;

    stop)
        echo -e "${YELLOW}⛔ Остановка Jarvis...${NC}"
        if systemctl --user is-active --quiet jarvis 2>/dev/null; then
            systemctl --user stop jarvis
            echo -e "${GREEN}✓ Остановлено через systemd${NC}"
        else
            $SUPERVISOR_CMD stop 2>/dev/null || echo "Супервизор не был запущен"
        fi
        ;;

    logs)
        echo -e "${YELLOW}📋 Логи (Ctrl+C для выхода):${NC}"
        journalctl --user -u jarvis -f
        ;;

    clean)
        echo -e "${YELLOW}🧹 Очистка временных файлов...${NC}"
        rm -f /tmp/jarvis-supervisor.pid /tmp/jarvis-bootstrap.pid /tmp/jarvis-supervisor.log
        rm -f /tmp/jarvis-llm.sock
        echo -e "${GREEN}✓ Очищено${NC}"
        ;;

    ""|--help|-h)
        show_help
        if [[ -z "$cmd" ]] || [[ "$cmd" == "--help" ]] || [[ "$cmd" == "-h" ]]; then
            exit 0
        fi
        ;;

    *)
        # Неизвестная команда — запускаем в foreground
        echo -e "${GREEN}🚀 Запуск Jarvis в foreground...${NC}"
        echo -e "${YELLOW}Ctrl+C для остановки${NC}"
        echo ""
        exec $SUPERVISOR_CMD foreground
        ;;
esac
