#!/usr/bin/env bash
# Wenzhou Dialect AI Agent - 一键启动脚本
# 无需GitHub，无需联网

set -e

echo "=================================="
echo "  🗣️ 温州话转普通话 AI Agent"
echo "  一键启动"
echo "=================================="

# 1. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 安装依赖
echo "📦 安装依赖..."
pip install -r requirements.txt -q

# 3. 启动(默认mock模式，无需下载模型)
echo "🚀 启动Gradio界面..."
echo "   浏览器打开: http://localhost:7860"
echo "=================================="
ASR_BACKEND=mock TTS_BACKEND=mock python frontend/gradio_app.py
