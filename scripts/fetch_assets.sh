#!/usr/bin/env bash
# fetch_assets.sh — докачать офлайн-ассеты, которые НЕ хранятся в git:
#   * Piper (бинарь + либы + espeak-ng-data)   → piper/
#   * русский голос ru_RU-dmitri-medium         → piper/*.onnx(.json)
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
# Голос dmitri (НЕ «dmitry» — такого пути на HF нет, отсюда были 404). Файлы:
# ru_RU-dmitri-medium.onnx + ru_RU-dmitri-medium.onnx.json. Конфиг сохраняем
# локально как *.json (без .onnx в середине) — именно это имя ждёт код
# (VOICE_CONFIG в orchestrator.py передаёт его piper через --config).
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium"
VOICE_ONNX_URL="${VOICE_ONNX_URL:-${VOICE_BASE}/ru_RU-dmitri-medium.onnx?download=true}"
VOICE_JSON_URL="${VOICE_JSON_URL:-${VOICE_BASE}/ru_RU-dmitri-medium.onnx.json?download=true}"
VOSK_URL="${VOSK_URL:-https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip}"

# Браузерный User-Agent: часть CDN (в т.ч. HF/alphacephei за прокси) отдаёт
# curl/wget-дефолту 403/заглушку, а на «браузерный» UA — реальный файл. Именно
# поэтому «по ссылке из браузера качается, а скрипт — нет».
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

FAILED=0  # счётчик некритичных провалов — влияет на финальный код возврата

# ───────────────────────────── helpers ──────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }

# fetch <url> <out-path> [min-bytes]
# Качает url в out, перебирая ВСЕ доступные загрузчики (curl → wget → aria2c) —
# раньше скрипт сдавался после первого же инструмента. Каждый: ретраи, резюме,
# браузерный UA, следование редиректам. После скачивания проверяем размер: меньше
# min-bytes ⇒ это, скорее всего, HTML-страница ошибки (сервер отдал её с кодом
# 200), а не настоящий файл — такой «успех» считаем провалом.
fetch() {
    local url="$1" out="$2" min_bytes="${3:-1024}"
    mkdir -p "$(dirname "$out")"
    echo "  ↓ $(basename "$out")"
    local tmp="$out.part"
    rm -f "$tmp"

    local ok=1
    if have curl; then
        curl -fL --retry 5 --retry-delay 2 --retry-connrefused \
             -A "$UA" -o "$tmp" "$url" && ok=0
    fi
    if [ "$ok" -ne 0 ] && have wget; then
        rm -f "$tmp"
        wget -q --tries=5 --waitretry=2 --retry-connrefused \
             --user-agent="$UA" -O "$tmp" "$url" && ok=0
    fi
    if [ "$ok" -ne 0 ] && have aria2c; then
        rm -f "$tmp"
        aria2c -x4 -s4 --max-tries=5 --retry-wait=2 --allow-overwrite=true \
               --auto-file-renaming=false --user-agent="$UA" \
               -o "$(basename "$tmp")" -d "$(dirname "$tmp")" "$url" && ok=0
    fi
    if [ "$ok" -ne 0 ]; then
        if ! have curl && ! have wget && ! have aria2c; then
            echo "  ✗ ни curl, ни wget, ни aria2c — нечем скачать (apt install curl)" >&2
        else
            echo "  ✗ не удалось скачать: $url" >&2
        fi
        rm -f "$tmp"
        return 1
    fi

    local size; size="$(wc -c < "$tmp" 2>/dev/null || echo 0)"
    if [ "$size" -lt "$min_bytes" ]; then
        echo "  ✗ $(basename "$out"): получено лишь $size B (< $min_bytes) — похоже на страницу ошибки, а не файл" >&2
        rm -f "$tmp"
        return 1
    fi
    mv -f "$tmp" "$out"
    return 0
}

# fetch_hf <url> <out> [min-bytes] — как fetch, но при провале huggingface.co
# повторяет с зеркала hf-mirror.com (тот же приём, что в `jarvis` для GGUF-весов:
# в КНР/за фильтрами HF недоступен, а зеркало — да).
fetch_hf() {
    local url="$1" out="$2" min_bytes="${3:-1024}"
    if fetch "$url" "$out" "$min_bytes"; then
        return 0
    fi
    case "$url" in
        *huggingface.co*)
            echo "  ↻ huggingface.co не дал файл — пробую зеркало hf-mirror.com" >&2
            fetch "${url/huggingface.co/hf-mirror.com}" "$out" "$min_bytes" && return 0
            ;;
    esac
    return 1
}

# ───────────────────────────── Piper (голос) ────────────────────────────────
ensure_piper() {
    if [ -x "$PIPER_DIR/piper" ]; then
        echo "✓ piper-бинарь уже на месте"
    else
        echo "→ Piper ($PIPER_ARCH)"
        local tmp; tmp="$(mktemp -d)"
        if fetch "$PIPER_URL" "$tmp/piper.tar.gz" 1000000; then
            # Тарбол распаковывается в каталог piper/ (бинарь + либы + espeak-ng-data).
            if tar -xzf "$tmp/piper.tar.gz" -C "$REPO_ROOT" && [ -x "$PIPER_DIR/piper" ]; then
                echo "✓ Piper распакован в $PIPER_DIR"
            else
                echo "✗ распаковка Piper не удалась" >&2
                FAILED=1
            fi
        else
            FAILED=1
        fi
        rm -rf "$tmp"
    fi

    # Русский голос (модель + конфиг). Конфиг с HF называется *.onnx.json —
    # сохраняем его как *.json (код ждёт именно это имя, см. шапку).
    if [ -f "$PIPER_DIR/ru_RU-dmitri-medium.onnx" ]; then
        echo "✓ голос ru_RU-dmitri на месте"
    else
        fetch_hf "$VOICE_ONNX_URL" "$PIPER_DIR/ru_RU-dmitri-medium.onnx" 1000000 || FAILED=1
    fi
    if [ -f "$PIPER_DIR/ru_RU-dmitri-medium.json" ]; then
        echo "✓ конфиг голоса на месте"
    else
        fetch_hf "$VOICE_JSON_URL" "$PIPER_DIR/ru_RU-dmitri-medium.json" 200 || FAILED=1
    fi
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
        FAILED=1
        return 1
    fi
    local tmp; tmp="$(mktemp -d)"
    if fetch "$VOSK_URL" "$tmp/vosk.zip" 1000000; then
        if ! unzip -q "$tmp/vosk.zip" -d "$tmp"; then
            echo "✗ распаковка Vosk не удалась" >&2
            FAILED=1; rm -rf "$tmp"; return 1
        fi
        # Zip распаковывается в vosk-model-small-ru-0.22/ — переносим СОДЕРЖИМОЕ в model/.
        local src; src="$(find "$tmp" -maxdepth 1 -type d -name 'vosk-model-*' | head -1)"
        if [ -n "$src" ]; then
            mkdir -p "$MODEL_DIR"
            if cp -a "$src"/. "$MODEL_DIR"/ && [ -f "$MODEL_DIR/am/final.mdl" ]; then
                echo "✓ Vosk-модель в $MODEL_DIR"
            else
                echo "✗ не удалось перенести Vosk-модель" >&2
                FAILED=1
            fi
        else
            echo "✗ в архиве Vosk не найден каталог модели" >&2
            FAILED=1
        fi
    else
        FAILED=1
    fi
    rm -rf "$tmp"
}

echo "═══ JARVIS · докачка ассетов (слух + голос) ═══"
ensure_piper
ensure_vosk
if [ "$FAILED" -eq 0 ]; then
    echo "Готово. Проверка: jarvis test"
else
    echo "Завершено С ОШИБКАМИ — часть ассетов не скачалась (см. ✗ выше)." >&2
    echo "Повтори запуск (резюмируется) или подложи файлы вручную по путям из ссылок." >&2
fi
exit "$FAILED"
