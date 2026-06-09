#!/usr/bin/env bash
# =============================================
#  🗣️ 温州话转普通话 AI Agent - 一键启动脚本
# =============================================
set -e

MODE="${1:-mock}"   # mock | firered | sensevoice

echo "=================================="
echo "  🗣️ 温州话 ↔ 普通话 双向翻译"
echo "  Mode: $MODE"
echo "=================================="

# 端口检测
PORT="${GRADIO_PORT:-7860}"
if lsof -i ":$PORT" >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用，尝试端口 $((PORT+1))"
    PORT=$((PORT+1))
fi

# 根据模式设置环境变量
export ASR_BACKEND="$MODE"
export ASR_WENZHOU_BACKEND="$MODE"
export TTS_BACKEND="edge_tts"

if [ "$MODE" = "mock" ]; then
    export ASR_MANDARIN_BACKEND="mock"
    export FUSION_MODE="single"
    export VAD_ENABLED="false"
    echo "  📦 模拟模式（无需模型，调试用）"
elif [ "$MODE" = "firered" ]; then
    export ASR_MANDARIN_BACKEND="mock"
    export FUSION_MODE="single"
    export VAD_ENABLED="false"
    echo "  🔥 FireRedASR2-AED 模式（温州话专用）"
    echo "  ⚠️  需要 ~2.4GB 可用内存"
elif [ "$MODE" = "sensevoice" ]; then
    export ASR_MANDARIN_BACKEND="sensevoice"
    export FUSION_MODE="single"
    export VAD_ENABLED="false"
    echo "  🎙️ SenseVoiceSmall 模式（通用多方言）"
    echo "  ⚠️  需要 ~1GB 可用内存"
fi

export GRADIO_PORT="$PORT"

echo "  🌐 浏览器打开: http://localhost:$PORT"
echo "=================================="

exec python frontend/gradio_app.py
