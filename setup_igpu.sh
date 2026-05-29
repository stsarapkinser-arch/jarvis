#!/usr/bin/env bash
# setup_igpu.sh — Intel N100 iGPU acceleration setup for Jarvis.
#
# Что делает:
#   1) Ставит llama-cpp-python с аппаратным backend'ом для интегрированного
#      графического ядра. Приоритет — OpenCL по ТЗ оператора. Если сборка
#      OpenCL не проходит на текущем стеке (Mesa клевер давно депрекейтнули
#      в апстриме llama.cpp), скрипт сам падает на Vulkan, потом — на чистый
#      CPU (OpenBLAS). Каждый шаг логируется, чтобы оператор мог понять,
#      какой бэкенд реально активен.
#   2) Скачивает qwen2.5-coder-3b-instruct-Q4_K_M.gguf в models/.
#   3) Импорт-проверка: from llama_cpp import Llama + наличие файла модели.
#
# Запуск (из корня репо):  ./setup_igpu.sh
# Переменные окружения:
#   VENV_DIR   — путь к venv (по умолчанию .venv)
#   MODEL_URL  — переопределить URL загрузки
#
# Идемпотентен — повторный запуск переустановит llama-cpp-python (нужно,
# когда оператор сменил Mesa-стек) и не будет качать модель повторно.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$REPO_ROOT/models"
MODEL_NAME="qwen2.5-coder-3b-instruct-q4_k_m.gguf"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf?download=true}"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

if [ ! -x "$PIP" ] || [ ! -x "$PY" ]; then
    echo "✗ venv не найден или сломан: $VENV_DIR" >&2
    echo "  создайте: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r requirements.txt" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

# ───────────── [1/3] llama-cpp-python с iGPU ─────────────
echo "[1/3] llama-cpp-python с аппаратным ускорением Intel iGPU..."

try_build() {
    local backend="$1"
    local cmake_args="$2"
    echo "  → пробую backend=$backend (CMAKE_ARGS='$cmake_args')"
    CMAKE_ARGS="$cmake_args -DCMAKE_BUILD_TYPE=Release" \
        FORCE_CMAKE=1 \
        "$PIP" install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
}

LLAMA_BACKEND="none"
# Порядок: OpenCL (ТЗ) → Vulkan (рекомендация Intel для Mesa NEO/ANV) → CPU BLAS.
if try_build "OpenCL" "-DGGML_OPENCL=ON" 2>&1 | tee /tmp/jarvis_llama_build.log; then
    LLAMA_BACKEND="OpenCL"
elif grep -qiE "opencl|cl\.h|Could NOT find" /tmp/jarvis_llama_build.log && \
     try_build "Vulkan" "-DGGML_VULKAN=ON"; then
    LLAMA_BACKEND="Vulkan"
elif try_build "CPU+BLAS" "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"; then
    LLAMA_BACKEND="CPU+BLAS"
else
    echo "✗ Все варианты сборки llama-cpp-python провалились." >&2
    echo "  Смотрите /tmp/jarvis_llama_build.log" >&2
    exit 2
fi
echo "  ✓ Активен backend: $LLAMA_BACKEND"

# ───────────── [2/3] Загрузка GGUF ─────────────
echo "[2/3] Qwen2.5-Coder 3B Instruct (Q4_K_M)..."
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  ✓ уже есть: $MODEL_PATH ($SIZE)"
else
    echo "  → качаю из HuggingFace..."
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 4 -s 4 --console-log-level=warn -d "$MODELS_DIR" -o "$MODEL_NAME" "$MODEL_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$MODEL_PATH.part" "$MODEL_URL"
        mv "$MODEL_PATH.part" "$MODEL_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "$MODEL_PATH.part" "$MODEL_URL"
        mv "$MODEL_PATH.part" "$MODEL_PATH"
    else
        echo "✗ ни aria2c, ни curl, ни wget — нечем скачать." >&2
        exit 3
    fi
    echo "  ✓ загружено: $(du -h "$MODEL_PATH" | cut -f1)"
fi

# ───────────── [3/3] Sanity check ─────────────
echo "[3/3] Проверка импорта и наличия модели..."
"$PY" - <<EOF
import os, sys
try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"✗ import llama_cpp failed: {e}", file=sys.stderr)
    sys.exit(4)
model_path = "$MODEL_PATH"
if not os.path.isfile(model_path):
    print(f"✗ model file missing: {model_path}", file=sys.stderr)
    sys.exit(5)
print(f"  ✓ llama_cpp.Llama импортируется")
print(f"  ✓ model: {model_path}")
print(f"  ✓ size:  {os.path.getsize(model_path)/(1024**3):.2f} GiB")
EOF

echo
echo "✓ iGPU-стек готов. Активный backend: $LLAMA_BACKEND"
echo "  Перезапустите Jarvis: systemctl --user restart jarvis"
echo "  Логи:                 journalctl --user -u jarvis -f"
