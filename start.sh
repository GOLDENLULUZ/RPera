#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    printf '错误：未找到 uv，请先安装 uv。\n' >&2
    exit 1
fi

cd "$PROJECT_DIR"
printf '正在同步依赖……\n'
uv sync
BIND="$(uv run python -c 'from rpera.server_config import load_server_config; config = load_server_config(); print("0.0.0.0" if config.lan_access else "127.0.0.1", config.port)')"
read -r HOST PORT <<< "$BIND"
printf 'RPera 已启动：http://127.0.0.1:%s（监听 %s）\n' "$PORT" "$HOST"
printf '按 Ctrl+C 关闭。\n'
exec uv run uvicorn rpera.app:app --host "$HOST" --port "$PORT"
