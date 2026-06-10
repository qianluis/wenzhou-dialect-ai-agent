"""
温州话桥 配置。

设计沿用 CLAUDE.md 的"多 provider 兜底"思路:每个引擎都可用环境变量切换后端,
本地引擎挂了时可改走 API。第一阶段(本机无 GPU)默认全部用 CPU/网络引擎:
  - ASR  : faster-whisper(tiny, CPU/int8)   —— 替代 CLAUDE.md 里的 Qwen3-ASR docker
  - 规范化: rules(词典 + 清洗)              —— 替代 vLLM Qwen3-1.7B(可选切 openai 走网关)
  - TTS  : edge-tts(走网络)                 —— 替代 CosyVoice2
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
LEXICON_PATH = ROOT / "data" / "lexicon" / "wenzhou_mandarin_glossary.csv"

# 采样率(浏览器重采样到 16k 推流;webrtcvad 也按 16k 取帧)
SR = 16000

# ── ASR ──
ASR_BACKEND = os.getenv("ASR_BACKEND", "faster_whisper")      # faster_whisper | mock
FW_MODEL = os.getenv("FASTER_WHISPER_MODEL", "base")         # tiny(快) | base(默认,均衡) | small(准但慢)
FW_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
FW_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
FW_BEAM = int(os.getenv("WHISPER_BEAM", "1"))                 # CPU 上 1 最快
# 偏置成简体普通话(whisper 默认易出繁体);装了 opencc 还会再繁转简
ASR_INITIAL_PROMPT = os.getenv("ASR_INITIAL_PROMPT", "以下是普通话的句子。")

# ── 规范化(方言/口语 → 标准普通话) ──
USE_NORM = os.getenv("USE_NORM", "true").lower() == "true"
NORM_BACKEND = os.getenv("NORM_BACKEND", "rules")            # rules | openai
# openai 兜底(走用户现有网关,如 MiMo/百炼;仅当 NORM_BACKEND=openai 时用)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── TTS ──
TTS_BACKEND = os.getenv("TTS_BACKEND", "edge_tts")          # edge_tts | mock
EDGE_VOICE = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# ── VAD(句级断句) ──
VAD_AGGR = int(os.getenv("VAD_AGGR", "2"))                  # 0-3,越大越激进
FRAME_MS = 30                                               # webrtcvad 只接受 10/20/30ms
SIL_LIMIT = int(os.getenv("SIL_LIMIT", "16"))              # 连续静音帧数判句末(16*30≈480ms)
MIN_VOICED_MS = int(os.getenv("MIN_VOICED_MS", "300"))     # 太短的语音段丢弃,避免噪声触发
