# 温州话桥 · wenzhou-bridge(第一阶段)

> 把温州话/普通话录音准实时转成普通话文字,并把普通话文字合成为普通话语音。
> 第一阶段聚焦 **A 方向**,全部自托管,链路单进程可跑。完整规格见 [CLAUDE.md](CLAUDE.md)。

## 这一版跑的是什么

CLAUDE.md 设计的第一阶段是 4 个 GPU 服务(Qwen3-ASR docker / vLLM Qwen3-1.7B / CosyVoice2 / 编排)。
本仓库这一版面向 **无 GPU 的单机**,把推理引擎内联进同一个 FastAPI 进程,用 CPU/网络引擎实现**同一条 A 方向链路与同样的验收标准**:

| 环节 | CLAUDE.md(GPU 版) | 本版(CPU 版) |
|------|---------------------|----------------|
| ASR | Qwen3-ASR docker `:8001` | **faster-whisper**(tiny, CTranslate2 CPU/int8) |
| 规范化 | vLLM Qwen3-1.7B `:8002` | **规则**:方言词典(~500)+ CSV 词条 + 清洗(可选切 `openai` 走网关) |
| TTS | CosyVoice2 `:8003` | **edge-tts**(微软在线普通话合成) |
| 编排+前端 | FastAPI `:8000` | FastAPI `:8000`(同) |

> 温州话识别仍是当中文/吴语**泛化识别**,准确率先记录基线——和 CLAUDE.md 的预期一致。
> 真正高准确率的温州话 ASR、温州话 TTS 属于第二阶段(M9 上微调),不在本仓库。
> 换回 GPU 栈时,接口字段(`type=final/raw/mandarin`、`/tts`)保持不变,改 `server/*.py` 的引擎实现即可。

## 架构

```
浏览器(麦克风 16k PCM, WebSocket 推流)
        │
        ▼
FastAPI 编排层 (server/main.py)
   ├─ webrtcvad 断句 → 整句音频(装不上自动回退能量断句)
   ├─ ASR     : faster-whisper  (server/asr.py)
   ├─ 规范化  : 规则/词典        (server/normalize.py)
   └─ TTS     : edge-tts 流式     (server/tts.py)→ 音频回浏览器
```

## 快速开始

```bash
cd wenzhou-bridge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh                      # → http://localhost:8000
```

首次录音时才下载 faster-whisper tiny 模型(~75MB,走 HuggingFace)。

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASR_BACKEND` | `faster_whisper` | `faster_whisper` / `mock` |
| `FASTER_WHISPER_MODEL` | `tiny` | `tiny` / `base` / `small`(越大越准越慢) |
| `TTS_BACKEND` | `edge_tts` | `edge_tts` / `mock` |
| `EDGE_TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | edge-tts 发音人 |
| `USE_NORM` | `true` | 是否做普通话规范化 |
| `NORM_BACKEND` | `rules` | `rules` / `openai`(走 `OPENAI_BASE_URL`/`OPENAI_API_KEY`) |

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 引擎状态自检 |
| `WS` | `/ws/asr` | 推 16k/mono/PCM16 → 断句识别 → `{type:final, raw, mandarin}` |
| `POST` | `/tts` | `{text, voice?}` → 流式音频 |

## 第一阶段验收(对照 CLAUDE.md)

- [ ] 对麦克风说普通话,短时内出正确文字
- [ ] 说温州话出文字(记录基线 WER,**不期待很高**)
- [ ] 文本框打普通话 → 合成 → 听到自然普通话语音
- [ ] 单句端到端延迟(本机 CPU 上以记录实测为准)

## 目录

```
wenzhou-bridge/
├── CLAUDE.md                  # 第一阶段完整规格(GPU 版设计)
├── server/
│   ├── main.py                # FastAPI:/ws/asr、/tts、/health
│   ├── config.py              # 引擎选择 + 兜底配置
│   ├── asr.py                 # faster-whisper 封装
│   ├── normalize.py           # 规则规范化编排
│   ├── tts.py                 # edge-tts 流式封装
│   ├── wenzhou_translator.py  # 内置方言↔普通话词典(复用)
│   └── llm_normalizer.py      # 清洗 + CSV 词典(复用)
├── web/index.html             # 浏览器录音前端
├── data/lexicon/              # 温州话↔普通话 CSV 词条
├── requirements.txt
└── run.sh
```
