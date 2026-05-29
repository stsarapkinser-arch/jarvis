#!/usr/bin/env bash
# setup_server.sh — нативный llama-server (Vulkan) для JARVIS на Intel N100.
#
# Фаза 1 ТЗ «Отвязка инференса»: Python больше НЕ грузит веса. Этот скрипт:
#   1) собирает бинарь llama-server из llama.cpp с бэкендом Vulkan (iGPU);
#   2) качает Llama-3.2-3B-Instruct (GGUF Q4_K_M) — нативный tool-calling;
#   3) ставит systemd --user unit jarvis-llm.service (KV-cache reuse, ctx 8192);
#   4) проверяет /health.
#
# Запуск (из корня репо):  ./setup_server.sh
# Переменные окружения:
#   LLAMA_CPP_REF  — git-ref llama.cpp (по умолчанию master)
#   MODEL_URL      — переопределить URL GGUF
#   JOBS           — параллелизм сборки (по умолчанию nproc)
#   SKIP_BUILD=1   — пропустить компиляцию (только модель + unit)
#   JARVIS_LLM_EXTRA_FLAGS — доп. флаги llama-server для перфоманса, напр.
#                  "--flash-attn on" (ускоряет декод, вдвое режет KV) или
#                  "--cache-type-k q8_0". Зависят от версии Mesa/сборки.
#
# Идемпотентен: повторный запуск делает git pull + пересборку и не качает
# модель заново.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
SERVER_BIN="$BUILD_DIR/bin/llama-server"
MODELS_DIR="$REPO_ROOT/models"
MODEL_NAME="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

UNIT_TEMPLATE="$REPO_ROOT/config/jarvis-llm.service"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_TARGET="$USER_UNIT_DIR/jarvis-llm.service"

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }

# ───────────── [0/4] Проверка инструментов сборки ─────────────
say "[0/4] Проверка зависимостей сборки..."
MISSING=()
for tool in git cmake; do
    command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool")
done
# glslc (shaderc) нужен для компиляции Vulkan-шейдеров GGML.
command -v glslc >/dev/null 2>&1 || MISSING+=("glslc(shaderc)")
if ! command -v vulkaninfo >/dev/null 2>&1; then
    warn "vulkaninfo не найден — не могу подтвердить наличие Vulkan ICD для iGPU."
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
    warn "Отсутствуют: ${MISSING[*]}"
    echo "  Установите (Debian/Kali): sudo apt install -y \\"
    echo "    git cmake build-essential libvulkan-dev glslc vulkan-tools mesa-vulkan-drivers"
    [ "${SKIP_BUILD:-0}" = "1" ] || exit 1
fi

# ───────────── [1/4] Сборка llama-server (Vulkan) ─────────────
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    say "[1/4] SKIP_BUILD=1 — пропускаю компиляцию."
else
    say "[1/4] llama.cpp @ $LLAMA_CPP_REF (Vulkan backend)..."
    if [ -d "$LLAMA_DIR/.git" ]; then
        git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
        git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
    else
        git clone --depth 1 --branch "$LLAMA_CPP_REF" \
            https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" 2>/dev/null \
        || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi

    cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
        -DGGML_VULKAN=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --config Release --target llama-server -j "$JOBS"

    if [ ! -x "$SERVER_BIN" ]; then
        echo "✗ Сборка прошла, но бинарь не найден: $SERVER_BIN" >&2
        exit 2
    fi
    echo "  ✓ llama-server: $SERVER_BIN"
fi

# ───────────── [2/4] Загрузка GGUF Llama-3.2-3B ─────────────
say "[2/4] Модель $MODEL_NAME..."
mkdir -p "$MODELS_DIR"
if [ -f "$MODEL_PATH" ]; then
    echo "  ✓ уже есть: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
else
    echo "  → качаю из HuggingFace (можно переопределить MODEL_URL)..."
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 4 -s 4 --console-log-level=warn -d "$MODELS_DIR" -o "$MODEL_NAME" "$MODEL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$MODEL_PATH.part" "$MODEL_URL" && mv "$MODEL_PATH.part" "$MODEL_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "$MODEL_PATH.part" "$MODEL_URL" && mv "$MODEL_PATH.part" "$MODEL_PATH"
    else
        echo "✗ ни aria2c, ни curl, ни wget — нечем скачать." >&2
        exit 3
    fi
    echo "  ✓ загружено: $(du -h "$MODEL_PATH" | cut -f1)"
fi

# ───────────── [3/4] systemd --user unit ─────────────
say "[3/4] Установка jarvis-llm.service..."
if [ ! -x "$SERVER_BIN" ]; then
    warn "Бинарь $SERVER_BIN отсутствует (SKIP_BUILD?) — unit будет ссылаться на него; соберите перед стартом."
fi
mkdir -p "$USER_UNIT_DIR"
# Опциональные перф-флаги оператора (напр. JARVIS_LLM_EXTRA_FLAGS="--flash-attn on").
# Пусто → строку-плейсхолдер удаляем; иначе подставляем.
EXTRA="${JARVIS_LLM_EXTRA_FLAGS:-}"
if [ -n "$EXTRA" ]; then
    EXTRA_SED="s|__EXTRA_FLAGS__|${EXTRA}|"
    echo "  → доп. флаги сервера: ${EXTRA}"
else
    EXTRA_SED="/__EXTRA_FLAGS__/d"
fi
sed \
    -e "s|__JARVIS_HOME__|${REPO_ROOT}|g" \
    -e "s|__LLAMA_SERVER_BIN__|${SERVER_BIN}|g" \
    -e "$EXTRA_SED" \
    "$UNIT_TEMPLATE" > "$UNIT_TARGET"

# Version-robustness: снять из unit'а флаги, которых нет в этой сборке бинаря
# (каждый tuning-флаг — на своей continuation-строке, удаляем строку целиком).
if [ -x "$SERVER_BIN" ]; then
    HELP="$("$SERVER_BIN" --help 2>&1 || true)"
    for flag in --cache-ram --cache-reuse --jinja; do
        if ! grep -q -- "$flag" <<<"$HELP"; then
            sed -i "\| ${flag} |d" "$UNIT_TARGET"
            warn "флаг ${flag} не поддержан этой сборкой llama-server — убран из unit"
        fi
    done
fi
echo "  ✓ unit: $UNIT_TARGET"
systemctl --user daemon-reload
systemctl --user enable jarvis-llm.service
echo "  ✓ enabled (автозапуск)"

# ───────────── [4/4] Запуск + health + проверка iGPU ─────────────
say "[4/4] Старт и проверка /health..."
systemctl --user restart jarvis-llm.service || warn "не удалось стартовать через systemd — проверьте journalctl --user -u jarvis-llm"
printf "  ожидаю загрузки модели в iGPU"
HEALTHY=0
# До 180 с: холодный первый старт = компиляция Vulkan-шейдеров + загрузка весов
# в shared VRAM. На N100 это легко >60 с (старый лимит давал ложный «не готов»).
for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
        HEALTHY=1; break
    fi
    printf "."; sleep 1
done
echo
if [ "$HEALTHY" = "1" ]; then
    echo "  ✓ llama-server отвечает на http://127.0.0.1:8080"
else
    warn "сервер ещё не готов за 180с. Логи: journalctl --user -u jarvis-llm -f"
fi

# Проверка, что инференс реально на iGPU (Vulkan), а не свалился на CPU.
GPU_LOG="$(journalctl --user -u jarvis-llm --no-pager -n 400 2>/dev/null || true)"
if grep -qiE "offloaded [0-9]+/[0-9]+ layers to GPU|to the GPU|Vulkan0|ggml_vulkan|using Vulkan|Found .*Vulkan" <<<"$GPU_LOG"; then
    echo "  ✓ iGPU-оффлоад активен (Vulkan)"
    grep -iE "offloaded [0-9]+/[0-9]+ layers" <<<"$GPU_LOG" | tail -1 | sed 's/^/    /'
else
    warn "не вижу следов Vulkan/GPU-оффлоада — возможно, инференс на CPU (медленно)."
    echo "    Проверьте: journalctl --user -u jarvis-llm | grep -iE 'vulkan|offloaded|gpu'"
    echo "    Нужны пакеты: libvulkan1 mesa-vulkan-drivers; и сборка с -DGGML_VULKAN=ON."
fi

cat <<EOF

✓ Готово. Полезное:
    Статус:   systemctl --user status jarvis-llm
    Логи:     journalctl --user -u jarvis-llm -f
    Health:   curl http://127.0.0.1:8080/health
    Tools:    curl http://127.0.0.1:8080/v1/models

Дальше — установите основной сервис Jarvis: ./scripts/install_service.sh
EOF
