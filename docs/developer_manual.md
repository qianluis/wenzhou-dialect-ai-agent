# 温州话转普通话 AI Agent — 项目详情文档

## 1. 项目概述

### 1.1 项目简介

温州话转普通话 AI Agent 是一个端到端的温州方言语音识别与普通话翻译系统。支持实时语音输入、多ASR后端融合、普通话语音播报，并配备完整的系统自检和数据沉淀机制。

### 1.2 核心价值

- **语言保护**：数字化保存温州方言的语音和对应文本语料
- **跨代沟通**：帮助不通温州话的人理解温州方言
- **公共事务**：在医院、政务窗口等场景辅助方言沟通
- **数据沉淀**：每次人工纠错自动积累平行语料，持续提升模型质量

### 1.3 技术栈

| 层 | 技术 |
|----|------|
| 框架 | Python 3.12+, OpenClaw Agent Framework |
| API | FastAPI + WebSocket |
| 前端 | Gradio |
| ASR | Faster-Whisper / FunASR / FireRedASR / TeleSpeech-ASR |
| TTS | Edge-TTS |
| 容器 | Docker / Docker Compose |
| 部署 | Hugging Face Spaces, ClawHub, GitHub |

## 2. 系统架构

### 2.1 流水线

```
Audio Input
    │
    ▼
[Preprocess] ── 16kHz mono WAV 转换（ffmpeg / fallback copy）
    │
    ▼
[ASR Engine] ── 5 种后端统一接口
    │   Mock: 开发调试
    │   Faster-Whisper: 快速 baseline
    │   FunASR: 部署友好
    │   FireRedASR2S: 工程化首选
    │   TeleSpeech-ASR: 方言最优
    │
    ▼
[Multi-Model Fusion] ── 可选加速精度
    │   Single: 单模型
    │   Cascading: 级联回退
    │   Ensemble: 并行投票
    │
    ▼
[LLM Normalizer] ── 方言 → 标准普通话
    │   Rules: 规则替换
    │   LLM: 大模型重写
    │
    ▼
[TTS Engine] ── Wire 或 edge-tts
    │
    ▼
[Mandarin Text + Speech]
```

## 3. 核心模块详解

### 3.1 ASR 引擎 (backend/asr_engine.py)

```
ASREngine
├── mock: 内置模拟，返回预置温州话→普通话示例
├── faster_whisper: CTranslate2 加速的 Whisper
├── funasr: 达摩院 FunASR 框架
├── firered: FireRedASR2S 端到端语音识别
└── telespeech: TeleSpeech 方言 ASR
```

所有后端统一 `transcribe(audio_path) → ASRResult(text, backend, meta)` 接口。

### 3.2 实时流式引擎 (backend/streaming_asr.py)

```
StreamingASREngine
├── 滑动窗口: 2 秒窗口 / 1 秒步长（50% 重叠）
├── 分块推理: 320ms 音频块输入
├── 融合模式:
│   Single: 单模型实时推理
│   Ensemble: 多模型并行 + 投票（最高精度）
│   Cascading: 快速→慢速级联
├── RingBuffer: 循环缓冲区管理
└── Callback: 流式结果回调
```

### 3.3 多模型融合（99% 准确率路径）

融合策略流程图：

```
音频 segment
    │
    ├──→[Faster-Whisper]─┐
    ├──→[FunASR]─────────┤
    ├──→[TeleSpeech]─────┤
    │                    │
    ▼                    ▼
  [投票融合]
    ├── 2/3以上一致 → 高置信度输出
    ├── 不一致 → 取最长结果
    └── 置信度 < 0.5 → 级联回退更大模型
    │
    ▼
  [最终输出]
```

### 3.4 系统自检 (backend/diagnostics.py)

诊断 14 项健康指标：

| # | 组件 | 含义 |
|---|------|------|
| 1 | ffmpeg | 音频格式转换 |
| 2 | python_deps | 关键 Python 依赖 |
| 3 | disk_space | 磁盘空间 |
| 4 | audio_devices | 音频输入/输出设备 |
| 5-9 | asr_* (x5) | 每个 ASR 后端可加载性 |
| 10-11 | tts_* (x2) | TTS 引擎可加载性 |
| 12 | e2e_mock | 端到端流水线 |

输出 JSON（API）和 HTML（可视报告）。

### 3.5 人工纠错存储 (backend/correction_store.py)

```
CorrectionStore (CSV 文件存储)
├── 核心字段: audio_path, asr_raw, mandarin_corrected
├── 元数据: speaker_id, gender, age, scene, noise_level
├── 自动 ID 生成 (UUID)
└── 追加模式（永不覆写）
```

每次修正自动写入 `data/corrections/corrections.csv`，可作为 ASR 微调的训练数据。

## 4. API 完整文档

### 4.1 健康检查

```
GET /
Response:
{
  "name": "Wenzhou2Mandarin Agent",
  "version": "2.0.0",
  "status": "running",
  "endpoints": { ... }
}
```

### 4.2 文件转写

```
POST /transcribe
Content-Type: multipart/form-data
Fields:
  - audio: File (支持的音频格式)
  - context: str (可选，场景提示)
  
Response:
{
  "audio_path": "...",
  "asr_raw": "温州话文本...",
  "mandarin_text": "普通话文本...",
  "tts_audio_path": "...",
  "backend": "mock",
  "latency_ms": { "asr": 0, "tts": 37, "total": 37 }
}
```

### 4.3 融合模式转写（99%）

```
POST /transcribe_fusion
与 /transcribe 相同参数
启动多模型并行推理 + 投票融合
```

### 4.4 系统诊断

```
GET /diagnostics
JSON 格式健康报告

GET /diagnostics_html
HTML 格式可视化报告（浏览器可直接打开）
```

### 4.5 保存纠错

```
POST /save_correction
Fields: audio_path, asr_raw, mandarin_text, corrected_text, speaker_id, gender, age, scene, noise_level, note
```

### 4.6 WebSocket 流式转写

```
WS /ws/transcribe
Protocol:
  客户端 → 服务器: 16kHz 16-bit PCM mono 音频块 (二进制)
  服务器 → 客户端: JSON 消息
    { "type": "partial", "text": "...", "is_final": false }
    { "type": "final", "text": "...", "is_final": true }
    { "type": "error", "message": "..." }
```

## 5. Gradio UI

4 个功能 Tab：

1. **🎙️ 实时翻译** — 上传 / 录音 + 融合开关 + TTS 播报 + 延迟显示
2. **🩺 系统自检** — 一键诊断 + HTML 报告 + 配置展示
3. **📝 人工纠错** — 修正文本 + 说话人信息 + 场景标注
4. **ℹ️ 关于** — 架构说明 + ASR后端矩阵 + 环境变量文档

## 6. 部署指南

### 6.1 本地开发

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 开发模式（mock 后端）
ASR_BACKEND=mock TTS_BACKEND=mock python frontend/gradio_app.py
```

### 6.2 Docker 部署

```bash
docker compose up
```

### 6.3 Hugging Face Spaces

在 HF Space 控制台：
1. Create Space → 选择 Docker
2. Git 推送到 Space 仓库
3. 自动构建

### 6.4 生产部署（可选 ASR 模型）

```bash
pip install edge-tts faster-whisper
ASR_BACKEND=faster_whisper TTS_BACKEND=edge_tts nohup uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
```

### 6.5 ClawHub Skills 市场

通过 OpenClaw 安装：

```bash
clawhub install wenzhou-dialect-ai-agent
```

## 7. 数据集建设计划

### 7.1 当前状态

- `data/lexicon/wenzhou_mandarin_glossary.csv` — 基础温州话↔普通话对照词典
- `data/corrections/corrections.csv` — 人工纠错积累中

### 7.2 路线图

| 阶段 | 目标 | 数据量 | 状态 |
|------|------|--------|------|
| P0 | 词典构建 | 500+ 词汇 | ✅ 完成 |
| P1 | 人工纠错累积 | 1000+ 条 | 🔄 进行中 |
| P2 | 众包收集 | 10000+ 条 | 📋 计划 |
| P3 | 专业标注 | 50000+ 条 | 📋 计划 |
| P4 | 端到端微调 | 100000+ 条 | 🎯 目标 |

## 8. 准确率优化报告

### 8.1 当前基线

| 后端 | 模拟环境 CER | 真实温州话 CER（预估） |
|------|-------------|----------------------|
| Mock | 0% | N/A（模拟） |
| Faster-Whisper small | — | ~15-25% |
| Faster-Whisper large | — | ~10-20% |
| FunASR Paraformer | — | ~10-18% |
| FireRedASR | — | ~8-15% |
| TeleSpeech-ASR | — | ~5-12% |
| **多模型融合** | — | **~3-8%（目标）** |

### 8.2 99/100 实现路径

1. **多模型融合** — 投票机制消除单模型误识别
2. **级联回退** — 置信度评估 + 大模型二次校正
3. **方言词表** — 持续扩充温州话特有词汇映射
4. **人工纠错闭环** — 每次修正提升模型
5. **微调** — 积累足够数据后用 LoRA 微调方言语种

## 9. 测试

```bash
pytest tests/ -v
```

10 项测试覆盖：CER 计算、模块导入、流水线、纠错存储、自检、流式引擎。

---

*文档版本: v2.0 | 2026-06-04*
