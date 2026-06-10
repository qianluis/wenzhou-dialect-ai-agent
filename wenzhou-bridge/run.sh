#!/usr/bin/env bash
# 启动温州话桥(单进程版)。沿用习惯:启动脚本独立、destructive 操作不放这里。
set -euo pipefail
cd "$(dirname "$0")"

# 引擎可用环境变量切换(默认全 CPU/网络):
#   ASR_BACKEND=faster_whisper|mock   FASTER_WHISPER_MODEL=tiny|base|small
#   TTS_BACKEND=edge_tts|mock         EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
#   USE_NORM=true|false               NORM_BACKEND=rules|openai
export ASR_BACKEND="${ASR_BACKEND:-faster_whisper}"
export TTS_BACKEND="${TTS_BACKEND:-edge_tts}"

VENV="${VENV:-.venv}"
[ -d "$VENV" ] && source "$VENV/bin/activate"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "==> 温州话桥 启动于 http://$HOST:$PORT  (ASR=$ASR_BACKEND, TTS=$TTS_BACKEND)"
exec uvicorn server.main:app --host "$HOST" --port "$PORT"
