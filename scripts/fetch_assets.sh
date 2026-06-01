#!/usr/bin/env bash
# fetch_assets.sh — докачать офлайн-ассеты, которые НЕ хранятся в git:
#   * Piper (бинарь + либы + espeak-ng-data)   → piper/
#   * русский голос ru_RU-dmitry-medium         → piper/*.onnx(.json)
#   * Vosk small ru (акустическая модель слуха) → model/
#
# Раньше эти ~128 МБ бинарей лежали прямо в репозитории (клон тащил их каждый
# раз, история пухла). GGUF-веса мозга/эмбеддера уже качаются скриптами
# setup_server.sh / setup_embed_server.sh — этот скрипт делает то же для слуха
# и голоса. Идемпотентно: то, что уже на месте, не перекачивается.
#
# Переопределяемо через env: PIPER_URL / VOICE_ONNX_URL / VOICE_JSON_URL /
# VOSK_URL. Сеть нужна; в офлайне скрипт честно скажет, что и куда положить.
set -uo pipefail

SELF="${BASH_SOURCE[0]}"
while [ -h "$SELF" ]; do
    link="$(readlink "$SELF")"
    case "$link" in /*) SELF="$link";; *) SELF="$(dirname "$SELF")/$link";; esac
done
REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO_ROOT"

PIPER_DIR="$REPO_ROOT/piper"
MODEL_DIR="$REPO_ROOT/model"

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)  PIPER_ARCH="x86_64" ;;
    aarch64|arm64) PIPER_ARCH="aarch64" ;;
    armv7l)        PIPER_ARCH="armv7l" ;;
    *)             PIPER_ARCH="x86_64" ;;  # разумный дефолт под N100
esac

PIPER_URL="${PIPER_URL:-https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_${PIPER_ARCH}.tar.gz}"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitry/medium"
VOICE_ONNX_URL="${VOICE_ONNX_URL:-${VOICE_BASE}/ru_RU-dmitry-medium.onnx?download=true}"
VOICE_JSON_URL="${VOICE_JSON_URL:-${VOICE_BASE}/ru_RU-dmitry-medium.onnx.json?download=true}"
VOSK_URL="${VOSK_URL:-https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip}"

# ───────────────────────────── helpers ──────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }

fetch() {  # fetch <url> <out-path>
    local url="$1" out="$2"
    mkdir -p "$(dirname "$out")"
    echo "  ↓ $(basename "$out")"
    if have aria2c; then
        aria2c -q -x4 -s4 -o "$(basename "$out")" -d "$(dirname "$out")" "$url" && return 0
    elif have curl; then
        curl -fL --progress-bar -o "$out.part" "$url" && mv "$out.part" "$out" && return 0
    elif have wget; then
        wget -q --show-progress -O "$out.part" "$url" && mv "$out.part" "$out" && return 0
    else
        echo "  ✗ ни aria2c, ни curl, ни wget — нечем скачать." >&2
        return 1
    fi
    echo "  ✗ не удалось скачать: $url" >&2
    return 1
}

# ───────────────────────────── Piper (голос) ────────────────────────────────
ensure_piper() {
    if [ -x "$PIPER_DIR/piper" ]; then
        echo "✓ piper-бинарь уже на месте"
    else
        echo "→ Piper ($PIPER_ARCH)"
        local tmp; tmp="$(mktemp -d)"
        if fetch "$PIPER_URL" "$tmp/piper.tar.gz"; then
            # Тарбол распаковывается в каталог piper/ (бинарь + либы + espeak-ng-data).
            tar -xzf "$tmp/piper.tar.gz" -C "$REPO_ROOT" \
                && echo "✓ Piper распакован в $PIPER_DIR" \
                || echo "✗ распаковка Piper не удалась" >&2
        fi
        rm -rf "$tmp"
    fi

    # Русский голос (модель + конфиг). Конфиг сохраняем как *.json (без .onnx в
    # середине) — именно это имя ждёт код (VOICE_CONFIG в orchestrator.py).
    [ -f "$PIPER_DIR/ru_RU-dmitry-medium.onnx" ] \
        && echo "✓ голос ru_RU-dmitry на месте" \
        || fetch "$VOICE_ONNX_URL" "$PIPER_DIR/ru_RU-dmitry-medium.onnx"
    [ -f "$PIPER_DIR/ru_RU-dmitry-medium.json" ] \
        && echo "✓ конфиг голоса на месте" \
        || fetch "$VOICE_JSON_URL" "$PIPER_DIR/ru_RU-dmitry-medium.json"
}

# ───────────────────────────── Vosk (слух) ──────────────────────────────────
ensure_vosk() {
    if [ -f "$MODEL_DIR/am/final.mdl" ]; then
        echo "✓ Vosk-модель уже на месте"
        return 0
    fi
    echo "→ Vosk small ru"
    if ! have unzip; then
        echo "✗ нет unzip — поставь его (apt install unzip) и повтори" >&2
        return 1
    fi
    local tmp; tmp="$(mktemp -d)"
    if fetch "$VOSK_URL" "$tmp/vosk.zip"; then
        unzip -q "$tmp/vosk.zip" -d "$tmp" || { echo "✗ распаковка Vosk не удалась" >&2; rm -rf "$tmp"; return 1; }
        # Zip распаковывается в vosk-model-small-ru-0.22/ — переносим СОДЕРЖИМОЕ в model/.
        local src; src="$(find "$tmp" -maxdepth 1 -type d -name 'vosk-model-*' | head -1)"
        if [ -n "$src" ]; then
            mkdir -p "$MODEL_DIR"
            cp -a "$src"/. "$MODEL_DIR"/ && echo "✓ Vosk-модель в $MODEL_DIR" \
                || echo "✗ не удалось перенести Vosk-модель" >&2
        else
            echo "✗ в архиве Vosk не найден каталог модели" >&2
        fi
    fi
    rm -rf "$tmp"
}

echo "═══ JARVIS · докачка ассетов (слух + голос) ═══"
ensure_piper
ensure_vosk
echo "Готово. Проверка: jarvis test"
