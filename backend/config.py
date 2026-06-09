from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CORRECTION_DIR = PROJECT_ROOT / "data" / "corrections"
LEXICON_PATH = PROJECT_ROOT / "data" / "lexicon" / "wenzhou_mandarin_glossary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORRECTION_DIR.mkdir(parents=True, exist_ok=True)

# ── ASR Backend ──
# Primary: FireRedASR2-AED (best for Wenzhou/Wu dialects)
# Secondary cascade: SenseVoiceSmall (fast, multi-dialect)
ASR_BACKEND = os.getenv("ASR_BACKEND", "firered")
ASR_WENZHOU_BACKEND = os.getenv("ASR_WENZHOU_BACKEND", "firered")
ASR_MANDARIN_BACKEND = os.getenv("ASR_MANDARIN_BACKEND", "sensevoice")

TTS_BACKEND = os.getenv("TTS_BACKEND", "edge_tts")
LLM_BACKEND = os.getenv("LLM_BACKEND", "rules")

FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "tiny")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# ── Streaming / Accuracy ──
# Available modes:
#   "single"     (default, memory-safe) — one ASR model at a time. ~2.3GB for FireRed
#   "cascading"  SenseVoice fast pass → FireRed fallback. Requires ~3.2GB RAM
#   "ensemble"   run all backends and majority vote. Heaviest
# Note: 5.5GB RAM environment should use "single" to avoid OOM
FUSION_MODE = os.getenv("FUSION_MODE", "single")
# Stream chunk duration in ms (160ms = fast response for real-time mic)
STREAM_CHUNK_MS = int(os.getenv("STREAM_CHUNK_MS", "160"))
# Whether to use multi-model fusion by default on file uploads
FUSION_ENABLED = os.getenv("FUSION_ENABLED", "true").lower() == "true"

# ── VAD ──
VAD_ENABLED = os.getenv("VAD_ENABLED", "true").lower() == "true"

# ── Data collection ──
SAVE_CORRECTIONS = os.getenv("SAVE_CORRECTIONS", "true").lower() == "true"

# ── UI defaults ──
DEFAULT_ACCURACY_TARGET = int(os.getenv("DEFAULT_ACCURACY_TARGET", "99"))
