# 🗣️ 温州话转普通话 AI Agent

> **Wenzhou Dialect → Mandarin Speech Translation Agent**
> 实时温州话语音识别 + 普通话翻译 + 语音播报，目标准确率99/100

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-00a86b.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-5+-orange.svg)](https://gradio.app)
[![ClawHub](https://img.shields.io/badge/ClawHub-available-8A2BE2.svg)](https://clawhub.ai)
[![GitHub](https://img.shields.io/badge/GitHub-repo-181717?logo=github)](https://github.com/qianluis/wenzhou-dialect-ai-agent)

---

## ✨ 核心功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 🎙️ **ASR 语音识别** | ✅ | 支持 5 种 ASR 后端：Mock / Faster-Whisper / FunASR / FireRedASR / TeleSpeech-ASR |
| 🎯 **多模型融合** | ✅ | 多模型并行推理 + 投票融合，朝 99% 准确率目标 |
| ⚡ **实时流式翻译** | ✅ | WebSocket 分块流式识别，说话即翻译 |
| 🔊 **TTS 语音播报** | ✅ | Edge-TTS 自然语音合成，支持 mock 离线模式 |
| 🩺 **系统自检** | ✅ | 一键诊断 14 项组件健康状态，输出 HTML 报告 |
| 📝 **人工纠错** | ✅ | 错误修正自动沉淀为训练语料 |
| 🐳 **Docker 部署** | ✅ | 一键容器化部署 |
| ☁️ **HF Space** | ✅ | Hugging Face Spaces 一键部署 |
| 🧠 **ClawHub Skill** | ✅ | OpenClaw 技能市场可用 |

---

## 🏗️ 系统架构

```
┌─────────┐    ┌──────────────┐    ┌──────────────────┐
│  Audio   │───▶│ Preprocess   │───▶│   ASR Engine      │
│  Input   │    │ 16kHz mono   │    │   (5 backends)    │
└─────────┘    └──────────────┘    └────────┬─────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                      Faster-Whisper   FunASR    TeleSpeech-ASR
                              │             │             │
                              └─────────────┼─────────────┘
                                            ▼
                                  多模型融合投票
                                    (Majority Vote)
                                            │
                                            ▼
                              ┌──────────────────────┐
                              │ Mandarin Normalizer   │
                              │  (Rules / LLM)        │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   TTS Engine          │
                              │   (edge-tts / mock)   │
                              └──────────┬───────────┘
                                         ▼
                               ┌─────────────────┐
                               │  🔊 普通话语音     │
                               │  📝 普通话文本     │
                               └─────────────────┘
```

---

## 🚀 快速开始

### 前置要求

- Python 3.12+
- pip（推荐 venv 虚拟环境）
- ffmpeg（推荐，非 WAV 格式需要）

### 安装

```bash
# 克隆
git clone https://github.com/qianluis/wenzhou-dialect-ai-agent.git
cd wenzhou-dialect-ai-agent

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# 使用 mock 后端（无需下载模型，立即试用）
ASR_BACKEND=mock TTS_BACKEND=mock python frontend/gradio_app.py

# 或使用真实后端
# pip install edge-tts faster-whisper
# ASR_BACKEND=faster_whisper TTS_BACKEND=edge_tts python frontend/gradio_app.py
```

打开 http://localhost:7860 即可使用。

### API 服务

```bash
ASR_BACKEND=mock TTS_BACKEND=mock uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

- API 文档：http://localhost:8000/docs
- 系统诊断：http://localhost:8000/diagnostics_html

---

## 🔌 API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/` | 健康检查 |
| `POST` | `/transcribe` | 文件上传转写 |
| `POST` | `/transcribe_fusion` | 多模型融合转写（高精度模式） |
| `GET` | `/diagnostics` | JSON 格式系统自检 |
| `GET` | `/diagnostics_html` | HTML 格式系统自检报告 |
| `POST` | `/save_correction` | 保存人工修正数据 |
| `WS` | `/ws/transcribe` | WebSocket 实时流式转写 |

### 流式实时转写（WebSocket）

```python
import asyncio, websockets, pyaudio

async def stream_mic():
    async with websockets.connect("ws://localhost:8000/ws/transcribe") as ws:
        p = pyaudio.PyAudio()
        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16,
                        input=True, frames_per_buffer=5120)
        while True:
            data = stream.read(5120)
            await ws.send(data)      # 发送音频块
            msg = await ws.recv()    # 接收实时翻译结果
            print(msg)

asyncio.run(stream_mic())
```

---

## 🎯 5 种 ASR 后端对比

| 后端 | 状态 | 语言 | 速度 | 精度 | 安装 |
|------|------|------|------|------|------|
| **mock** | ✅ 内置 | 模拟温州话 | 极快 | 模拟 | 无需安装 |
| **Faster-Whisper** | ✅ 可选 | 多语言 | ⚡ 快 | 🟢 好 | `pip install faster-whisper` |
| **FunASR** | ✅ 可选 | 中文优先 | ⚡ 快 | 🟢 好 | `pip install funasr` |
| **FireRedASR2S** | ✅ 可选 | 中文 | 🐢 中等 | 🔵 更好 | `pip install git+https://github.com/FireRedTeam/FireRedASR.git` |
| **TeleSpeech-ASR** | ✅ 可选 | 中文方言 | 🐢 中等 | 🟣 **温州话最优** | `pip install transformers torch` |

### 🎯 多模型融合

设置 `FUSION_MODE` 环境变量：

- `single` — 单模型推理（最快）
- `cascading` — 级联回退（默认，平衡速度和精度）
- `ensemble` — 多模型并行 + 投票融合（最准确）

```bash
# 99% 精度模式
ASR_BACKEND=faster_whisper FUSION_MODE=ensemble TTS_BACKEND=edge_tts python frontend/gradio_app.py
```

---

## 🎯 准确率优化方案

目标是 **99/100** 准确率。实现路径：

1. **多模型融合** (§3.1) — 并行运行多个 ASR 模型，投票取多数
2. **级联回退** — 快速模型低置信度时回退更强大模型
3. **方言词表** — 构建温州话 ↔ 普通话对照词典（`data/lexicon/wenzhou_mandarin_glossary.csv`）
4. **人工纠错闭环** — 每次修正自动沉淀为训练数据
5. **置信度评估** — 启发式分析输出文本质量，低分自动触发重试

---

## 🔧 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_BACKEND` | `mock` | ASR后端：mock / faster_whisper / funasr / firered / telespeech |
| `TTS_BACKEND` | `mock` | TTS后端：mock / edge_tts |
| `LLM_BACKEND` | `rules` | 规范化引擎：rules / llm |
| `FUSION_MODE` | `cascading` | 融合策略：single / cascading / ensemble |
| `FUSION_ENABLED` | `false` | 默认是否启用融合 |
| `STREAM_CHUNK_MS` | `320` | 流式块大小(ms) |
| `FASTER_WHISPER_MODEL` | `small` | Whisper 模型大小 |
| `EDGE_TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | TTS 发音人 |
| `GRADIO_PORT` | `7860` | Gradio 端口 |

---

## 🐳 Docker 部署

```bash
docker compose up
```

或在 Hugging Face Spaces：

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
1. 创建 Space → Docker
2. 克隆本仓库
3. 自动构建运行

---

## 📂 项目结构

```
wenzhou2mandarin-agent/
├── backend/
│   ├── app.py               # FastAPI 主应用
│   ├── asr_engine.py         # ASR 引擎 (5 种后端)
│   ├── diagnostics.py        # 🩺 系统自检模块
│   ├── enhanced_pipeline.py  # 🎯 增强流水线 (融合+流式)
│   ├── streaming_asr.py      # ⚡ 实时流式 ASR 引擎
│   ├── pipeline.py           # 基础流水线
│   ├── config.py             # 配置管理
│   ├── tts_engine.py         # TTS 引擎
│   ├── llm_normalizer.py     # 普通话规范化
│   └── correction_store.py   # 纠错存储
├── frontend/
│   └── gradio_app.py         # Gradio Web 界面 (4 Tab)
├── scripts/
│   ├── preprocess_audio.py   # 音频预处理
│   ├── batch_transcribe.py   # 批量转写
│   └── evaluate_cer.py       # CER 评估
├── data/
│   ├── lexicon/              # 温州话↔普通话词典
│   ├── samples/              # 示例音频
│   └── corrections/          # 人工纠错数据
├── docs/
│   ├── research_report.md    # ASR 后端研究报告
│   ├── product_design.md     # 产品设计文档
│   └── dataset_plan.md       # 数据集建设计划
├── tests/
│   └── test_cer.py           # 10 项自动化测试
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── CLAW_TASK.md
└── README.md
```

---

## 📈 测试

```bash
pytest tests/ -v
# 10 passed ✅
```

---

## 🧠 ClawHub AI Skills 市场

本项目已上架 OpenClaw Skills 市场 (ClawHub.ai)：

```bash
# 安装技能
clawhub install wenzhou-dialect-ai-agent
```

---

## 🎯 路线图

- [x] 5 种 ASR 后端统一接口
- [x] 多模型融合投票机制
- [x] WebSocket 实时流式转写
- [x] 系统自检与诊断面板
- [x] 人工纠错数据收集
- [x] Docker 容器化部署
- [x] Hugging Face Spaces 支持
- [x] ClawHub Skills 市场上架
- [ ] 温州话专用微调数据集（>10万条）
- [ ] 端到端方言 ASR 模型微调
- [ ] 移动端 App 集成
- [ ] 多方言支持（闽南语、粤语、客家话）

---

## 📄 许可证

MIT License © 2026

---

## 🙏 致谢

- [OpenClaw](https://github.com/openclaw/openclaw) — Agent 框架
- [ClawHub](https://clawhub.ai) — Skills 市场
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — ASR 引擎
- [FunASR](https://github.com/modelscope/FunASR) — ASR 引擎
- [FireRedASR](https://github.com/FireRedTeam/FireRedASR) — ASR 引擎
- [Edge-TTS](https://github.com/rany2/edge-tts) — TTS 引擎

---

*Made with ❤️ by qianluis*
