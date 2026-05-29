#!/usr/bin/env bash
# JARVIS — аварийный рычаг деактивации.
#
# По умолчанию Jarvis бессмертен и стартует с системой. Этот скрипт нужен
# только если что-то пошло НЕ ТАК и оператор хочет тишины:
#   - LLM зациклился и говорит без остановки
#   - Sentinel ложно срабатывает и заваливает голосом
#   - Нужно выполнить тяжёлую работу без фоновой нагрузки
#   - Бэкап/восстановление системы
#
# Что делает:
#   1. Останавливает службу СЕЙЧАС.
#   2. Отключает автозапуск (jarvis больше не стартует с системой).
#   3. Стопит все процессы piper/aplay, если они зависли.
#   4. Записывает причину в /tmp/jarvis-disabled.flag.
#
# Чтобы вернуть Jarvis обратно — запустить ./enable_jarvis.sh.
set -euo pipefail

REASON="${*:-no reason given}"
FLAG_FILE="/tmp/jarvis-disabled.flag"

echo "═══ JARVIS EMERGENCY SHUTDOWN ═══"
echo "Причина: ${REASON}"
echo ""

# 1) Остановить службу.
if systemctl --user is-active --quiet jarvis.service; then
    echo "→ Останавливаю jarvis.service…"
    systemctl --user stop jarvis.service || true
else
    echo "→ jarvis.service уже не запущен."
fi

# 2) Отключить автозапуск.
if systemctl --user is-enabled --quiet jarvis.service 2>/dev/null; then
    echo "→ Отключаю автозапуск…"
    systemctl --user disable jarvis.service
else
    echo "→ Автозапуск уже отключён."
fi

# 3) Подобрать висящие процессы озвучки/инференса.
echo "→ Зачищаю зависшие piper/aplay/llama-cpp…"
for proc in piper aplay; do
    pkill -u "$(id -u)" -x "${proc}" 2>/dev/null || true
done
# llama-cpp работает в python-процессе jarvis'а, отдельно убивать не нужно.

# 4) Записать флаг с причиной — enable_jarvis.sh его покажет при запуске.
{
    echo "Disabled at: $(date -Iseconds)"
    echo "Reason: ${REASON}"
    echo "By user: $(whoami)"
} > "${FLAG_FILE}"

cat <<EOF

✓ JARVIS деактивирован.
  Флаг: ${FLAG_FILE}

Чтобы вернуть обратно:
    ./enable_jarvis.sh

Статус:
    systemctl --user status jarvis.service

EOF
