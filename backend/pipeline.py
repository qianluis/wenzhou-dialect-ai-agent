from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import shutil
import uuid

from backend.config import (
    ASR_BACKEND, TTS_BACKEND, LLM_BACKEND, FASTER_WHISPER_MODEL,
    EDGE_TTS_VOICE, OUTPUT_DIR, LEXICON_PATH, CORRECTION_DIR
)
from backend.asr_engine import ASREngine
from backend.llm_normalizer import MandarinNormalizer
from backend.tts_engine import TTSEngine
from backend.correction_store import CorrectionStore


class Wenzhou2MandarinPipeline:
    def __init__(self):
        self.asr = ASREngine(backend=ASR_BACKEND, model_name=FASTER_WHISPER_MODEL)
        self.normalizer = MandarinNormalizer(lexicon_path=LEXICON_PATH, backend=LLM_BACKEND)
        self.tts = TTSEngine(backend=TTS_BACKEND, voice=EDGE_TTS_VOICE, output_dir=OUTPUT_DIR)
        self.store = CorrectionStore(CORRECTION_DIR / "corrections.csv")

    def run(self, audio_path: str | Path, context: str = "") -> Dict[str, Any]:
        local_audio = self._copy_audio(audio_path)
        asr_result = self.asr.transcribe(local_audio)
        mandarin = self.normalizer.normalize(asr_result.text, context=context)
        tts_path = self.tts.synthesize(mandarin)
        return {
            "audio_path": str(local_audio),
            "asr_raw": asr_result.text,
            "mandarin_text": mandarin,
            "tts_audio_path": str(tts_path),
            "backend": asr_result.backend,
            "meta": asr_result.meta or {},
        }

    def save_correction(self, result: Dict[str, Any], corrected_text: str, **meta) -> str:
        return self.store.append(
            audio_path=result.get("audio_path", ""),
            asr_raw=result.get("asr_raw", ""),
            mandarin_corrected=corrected_text,
            **meta
        )

    def _copy_audio(self, audio_path: str | Path) -> Path:
        src = Path(audio_path)
        ext = src.suffix or ".wav"
        dst = OUTPUT_DIR / f"input_{uuid.uuid4().hex[:10]}{ext}"
        shutil.copyfile(src, dst)
        return dst
