#!/usr/bin/env bash
# JARVIS — реактивация после аварийного выключения.
#
# Запускать после ./disable_jarvis.sh, когда оператор хочет вернуть
# Jarvis обратно в строй.
#
# Что делает:
#   1. Показывает почему он был отключён (если флаг есть).
#   2. Включает автозапуск.
#   3. Стартует службу сейчас.
#   4. Удаляет флаг.
set -euo pipefail

FLAG_FILE="/tmp/jarvis-disabled.flag"

echo "═══ JARVIS REACTIVATION ═══"

if [[ -f "${FLAG_FILE}" ]]; then
    echo "→ Был отключён со следующей причиной:"
    echo "─────"
    cat "${FLAG_FILE}"
    echo "─────"
    echo ""
fi

# Включить автозапуск (идемпотентно).
echo "→ Включаю автозапуск…"
systemctl --user enable jarvis.service

# Стартовать сейчас.
echo "→ Запускаю jarvis.service…"
systemctl --user start jarvis.service

# Убрать флаг.
if [[ -f "${FLAG_FILE}" ]]; then
    rm -f "${FLAG_FILE}"
    echo "→ Флаг ${FLAG_FILE} удалён."
fi

# Короткий статус.
echo ""
systemctl --user status jarvis.service --no-pager --lines=5 || true

cat <<EOF

✓ JARVIS активен и стартует с системой.
  Логи:  journalctl --user -u jarvis -f
  Стоп:  ./disable_jarvis.sh "причина"

EOF
