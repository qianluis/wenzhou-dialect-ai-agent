from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CORRECTION_DIR = PROJECT_ROOT / "data" / "corrections"
LEXICON_PATH = PROJECT_ROOT / "data" / "lexicon" / "wenzhou_mandarin_glossary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORRECTION_DIR.mkdir(parents=True, exist_ok=True)

# --- Core backends ---
ASR_BACKEND = os.getenv("ASR_BACKEND", "mock")
TTS_BACKEND = os.getenv("TTS_BACKEND", "mock")
LLM_BACKEND = os.getenv("LLM_BACKEND", "rules")

FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "small")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# --- Streaming / Accuracy ---
# "single" | "ensemble" | "cascading"
FUSION_MODE = os.getenv("FUSION_MODE", "cascading")
# Stream chunk duration in ms (320ms = 20ms*16, good for real-time mic)
STREAM_CHUNK_MS = int(os.getenv("STREAM_CHUNK_MS", "320"))
# Whether to use multi-model fusion by default on file uploads
FUSION_ENABLED = os.getenv("FUSION_ENABLED", "false").lower() == "true"

# --- Data collection ---
SAVE_CORRECTIONS = os.getenv("SAVE_CORRECTIONS", "true").lower() == "true"

# --- UI defaults ---
DEFAULT_ACCURACY_TARGET = int(os.getenv("DEFAULT_ACCURACY_TARGET", "99"))
