#!/usr/bin/env bash
# JARVIS — установка/обновление systemd --user unit'а.
#
# Что делает:
#   1. Находит рабочий venv (тот, где импортируется PyQt6).
#   2. Подставляет реальные пути в jarvis.service.
#   3. Кладёт его в ~/.config/systemd/user/.
#   4. systemctl --user daemon-reload + enable.
#   5. Не стартует автоматически — финальный шаг за оператором.
#
# Идемпотентен: повторный запуск переустанавливает unit и перечитывает daemon.
set -euo pipefail

# Корень репозитория = родитель scripts/. Так пути для `python -m src.core.*`
# и WorkingDirectory указывают на репозиторий, а не на scripts/.
JARVIS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${JARVIS_HOME}/config/jarvis.service"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TARGET="${USER_UNIT_DIR}/jarvis.service"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "❌ Шаблон unit-файла не найден: ${TEMPLATE}" >&2
    exit 1
fi

# -- 1) Найти рабочий venv -----------------------------------------------------
# Кандидаты в порядке предпочтения. Первый, в котором импортируется PyQt6,
# побеждает. Так мы переживём миграцию .venv <-> venv без ручной правки.
declare -a CANDIDATES=(
    "${JARVIS_HOME}/.venv/bin/python"
    "${JARVIS_HOME}/venv/bin/python"
)

PY=""
for cand in "${CANDIDATES[@]}"; do
    if [[ -x "${cand}" ]] && "${cand}" -c "import PyQt6" >/dev/null 2>&1; then
        PY="$(realpath "${cand}")"
        break
    fi
done

if [[ -z "${PY}" ]]; then
    echo "❌ Не нашёл venv с установленным PyQt6. Проверьте:" >&2
    for cand in "${CANDIDATES[@]}"; do echo "   - ${cand}" >&2; done
    echo "   Запустите: ${JARVIS_HOME}/.venv/bin/pip install -r requirements.txt" >&2
    exit 2
fi
echo "✓ venv: ${PY}"

# -- 2) Подставить плейсхолдеры в unit ----------------------------------------
mkdir -p "${USER_UNIT_DIR}"
# Используем '|' как разделитель в sed, чтобы пути со слешами не ломали выражение.
sed \
    -e "s|__JARVIS_HOME__|${JARVIS_HOME}|g" \
    -e "s|__JARVIS_PYTHON__|${PY}|g" \
    "${TEMPLATE}" > "${TARGET}"
echo "✓ unit: ${TARGET}"

# -- 3) Перечитать конфиг + включить ------------------------------------------
systemctl --user daemon-reload
echo "✓ systemctl --user daemon-reload"

systemctl --user enable jarvis.service
echo "✓ systemctl --user enable jarvis.service"

# -- 4) Подсказки оператору ---------------------------------------------------
cat <<EOF

Установка завершена. Что дальше:

    Запуск сейчас:        systemctl --user start jarvis
    Статус:               systemctl --user status jarvis
    Логи в реальном вр.:  journalctl --user -u jarvis -f
    Остановить:           systemctl --user stop jarvis
    Снести unit:          systemctl --user disable jarvis && rm ${TARGET}

EOF
