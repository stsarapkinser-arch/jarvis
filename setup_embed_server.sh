#!/usr/bin/env bash
# setup_embed_server.sh — embed-сервер (llama-server --embedding) для JARVIS.
#
# СМЕРТЬ OLLAMA. Эмбеддинги памяти (ChronoMemory) больше не требуют демона
# ollama: их считает ВТОРОЙ лёгкий экземпляр llama-server в embedding-режиме с
# маленькой МУЛЬТИЯЗЫЧНОЙ моделью (русский у оператора). Тот же стек llama.cpp,
# ноль нового рантайма, out-of-process, ~120M в RAM на CPU.
#
# Предполагает, что бинарь llama-server уже собран через ./setup_server.sh
# (переиспользуем его, не собираем заново). Этот скрипт:
#   1) качает GGUF embedding-модели в models/embed/;
#   2) ставит systemd --user unit jarvis-embed.service (порт 8090, CPU);
#   3) проверяет /health и реальный эмбеддинг.
#
# Запуск (из корня репо):  ./setup_embed_server.sh
# Переменные окружения:
#   EMBED_MODEL_URL    — URL GGUF embedding-модели (по умолчанию multilingual-e5-small).
#   EMBED_MODEL_FILE   — имя файла модели в models/embed/ (выводится из URL).
#   EMBED_MODEL_ALIAS  — alias, под которым сервер отдаёт модель (= JARVIS_EMBED_MODEL).
#   EMBED_POOLING      — mean (e5/MiniLM, по умолчанию) | cls (bge) | last.
#   SERVER_BIN         — путь к llama-server (по умолчанию из сборки setup_server.sh).
#
# Идемпотентен: повторный запуск не качает модель заново.
#
# ВАЖНО про размерность: смена embedding-модели меняет размерность вектора и
# делает СТАРУЮ базу ./jarvis_memory несовместимой. При первой миграции с Ollama
# (или смене модели) очистите её:  rm -rf ./jarvis_memory
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BIN="${SERVER_BIN:-$REPO_ROOT/llama.cpp/build/bin/llama-server}"
EMBED_DIR="$REPO_ROOT/models/embed"

# Дефолт — multilingual-e5-small (384 dim, ~118M, сильный русский, mean pooling).
# Любую другую GGUF embedding-модель подставьте через EMBED_MODEL_URL.
EMBED_MODEL_URL="${EMBED_MODEL_URL:-https://huggingface.co/ChristianAzinn/multilingual-e5-small-gguf/resolve/main/multilingual-e5-small.fp16.gguf?download=true}"
# Имя файла: из URL (без query-строки) либо переопределяемо.
_DEFAULT_FILE="$(basename "${EMBED_MODEL_URL%%\?*}")"
EMBED_MODEL_FILE="${EMBED_MODEL_FILE:-$_DEFAULT_FILE}"
EMBED_MODEL_PATH="$EMBED_DIR/$EMBED_MODEL_FILE"
EMBED_MODEL_ALIAS="${EMBED_MODEL_ALIAS:-multilingual-e5-small}"
EMBED_POOLING="${EMBED_POOLING:-mean}"

UNIT_TEMPLATE="$REPO_ROOT/config/jarvis-embed.service"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_TARGET="$USER_UNIT_DIR/jarvis-embed.service"

say()  { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }

# ───────────── [1/3] Проверка бинаря + загрузка модели ─────────────
say "[1/3] Проверка llama-server и загрузка embedding-модели..."
if [ ! -x "$SERVER_BIN" ]; then
    warn "Бинарь llama-server не найден: $SERVER_BIN"
    echo "  Сначала соберите его:  ./setup_server.sh   (или укажите SERVER_BIN=...)"
    exit 1
fi
echo "  ✓ llama-server: $SERVER_BIN"

mkdir -p "$EMBED_DIR"
if [ -f "$EMBED_MODEL_PATH" ]; then
    echo "  ✓ модель уже есть: $EMBED_MODEL_PATH ($(du -h "$EMBED_MODEL_PATH" | cut -f1))"
else
    echo "  → качаю embedding-модель: $EMBED_MODEL_FILE"
    echo "    (переопределяемо через EMBED_MODEL_URL / EMBED_MODEL_FILE)"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 4 -s 4 --console-log-level=warn -d "$EMBED_DIR" -o "$EMBED_MODEL_FILE" "$EMBED_MODEL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$EMBED_MODEL_PATH.part" "$EMBED_MODEL_URL" && mv "$EMBED_MODEL_PATH.part" "$EMBED_MODEL_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "$EMBED_MODEL_PATH.part" "$EMBED_MODEL_URL" && mv "$EMBED_MODEL_PATH.part" "$EMBED_MODEL_PATH"
    else
        echo "✗ ни aria2c, ни curl, ни wget — нечем скачать." >&2
        exit 3
    fi
    echo "  ✓ загружено: $(du -h "$EMBED_MODEL_PATH" | cut -f1)"
fi

# ───────────── [2/3] systemd --user unit ─────────────
say "[2/3] Установка jarvis-embed.service..."
mkdir -p "$USER_UNIT_DIR"
sed \
    -e "s|__JARVIS_HOME__|${REPO_ROOT}|g" \
    -e "s|__LLAMA_SERVER_BIN__|${SERVER_BIN}|g" \
    -e "s|__EMBED_MODEL_FILE__|${EMBED_MODEL_FILE}|g" \
    -e "s|__EMBED_MODEL_ALIAS__|${EMBED_MODEL_ALIAS}|g" \
    -e "s|__EMBED_POOLING__|${EMBED_POOLING}|g" \
    "$UNIT_TEMPLATE" > "$UNIT_TARGET"

# Version-robustness: снять флаги, которых нет в этой сборке бинаря.
HELP="$("$SERVER_BIN" --help 2>&1 || true)"
for flag in --pooling --embedding; do
    if ! grep -q -- "$flag" <<<"$HELP"; then
        warn "флаг ${flag} не поддержан этой сборкой llama-server — соберите свежий llama.cpp"
    fi
done
echo "  ✓ unit: $UNIT_TARGET"
systemctl --user daemon-reload
systemctl --user enable jarvis-embed.service
echo "  ✓ enabled (автозапуск)"

# ───────────── [3/3] Запуск + health + проба эмбеддинга ─────────────
say "[3/3] Старт и проверка /health (порт 8090)..."
systemctl --user restart jarvis-embed.service || warn "не удалось стартовать — journalctl --user -u jarvis-embed"
printf "  ожидаю загрузки модели"
HEALTHY=0
for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8090/health 2>/dev/null | grep -q '"ok"'; then
        HEALTHY=1; break
    fi
    printf "."; sleep 1
done
echo
if [ "$HEALTHY" = "1" ]; then
    echo "  ✓ embed-сервер отвечает на http://127.0.0.1:8090"
    # Реальная проба: непустой вектор на русскую фразу.
    RESP="$(curl -fsS http://127.0.0.1:8090/v1/embeddings \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${EMBED_MODEL_ALIAS}\",\"input\":\"проверка связи\"}" 2>/dev/null || true)"
    if grep -q '"embedding"' <<<"$RESP"; then
        DIM="$(grep -o '"embedding":\[[^]]*' <<<"$RESP" | tr ',' '\n' | grep -c '.')"
        echo "  ✓ эмбеддинг получен (размерность ~${DIM})"
    else
        warn "сервер жив, но /v1/embeddings не вернул вектор — проверьте модель/флаг --embedding"
    fi
else
    warn "embed-сервер не готов за 60с. Логи: journalctl --user -u jarvis-embed -f"
fi

cat <<EOF

✓ Готово. Embed-сервер (замена Ollama):
    Статус:   systemctl --user status jarvis-embed
    Логи:     journalctl --user -u jarvis-embed -f
    Health:   curl http://127.0.0.1:8090/health
    Проба:    curl http://127.0.0.1:8090/v1/embeddings -d '{"input":"привет"}' -H 'Content-Type: application/json'

Если меняли embedding-модель — очистите старую базу:  rm -rf ./jarvis_memory
Переменные Jarvis (если порт/модель нестандартные):
    JARVIS_EMBED_ENDPOINT=http://127.0.0.1:8090
    JARVIS_EMBED_MODEL=${EMBED_MODEL_ALIAS}
EOF
