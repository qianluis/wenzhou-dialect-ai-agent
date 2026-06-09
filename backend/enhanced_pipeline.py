"""
Enhanced Pipeline (Optimized v2)
================================
Extends the base bidirectional pipeline with:
  1. VAD-based silence removal
  2. Cascading ASR (SenseVoice → FireRed)
  3. System diagnostics
  4. Correction store

Supports both file-based and streaming inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, Callable
import numpy as np

from backend.config import (
    ASR_BACKEND, TTS_BACKEND, LLM_BACKEND, FASTER_WHISPER_MODEL,
    EDGE_TTS_VOICE, OUTPUT_DIR, LEXICON_PATH, CORRECTION_DIR,
    FUSION_MODE, STREAM_CHUNK_MS, VAD_ENABLED,
)
from backend.asr_engine import ASREngine
from backend.streaming_asr import StreamingASREngine
from backend.llm_normalizer import MandarinNormalizer
from backend.tts_engine import TTSEngine
from backend.correction_store import CorrectionStore
from backend.vad_engine import VADEngine
from backend.bidirectional_pipeline import BidirectionalPipeline
from backend.diagnostics import run_all, print_report


_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EnhancedPipeline()
    return _pipeline_instance


class EnhancedPipeline:
    """
    Enhanced pipeline with streaming, cascading, and diagnostics.

    Uses BidirectionalPipeline for core logic + Streaming for real-time.
    """

    def __init__(self):
        self._base = BidirectionalPipeline()
        self.store = CorrectionStore(CORRECTION_DIR / "corrections.csv")
        self._streaming_engine = None

    @property
    def asr(self):
        return self._base.asr_wenzhou

    @property
    def normalizer(self):
        return self._base.normalizer

    @property
    def tts(self):
        return self._base.tts_mandarin

    # ------------------------------------------------------------------
    # File-based inference (with VAD + cascading)
    # ------------------------------------------------------------------
    def run(self, audio_path: str | Path, context: str = "", fusion: bool = False) -> Dict[str, Any]:
        if fusion:
            return self._base.process(audio_path, context=context, tts_enabled=True)
        else:
            return self._base.process(audio_path, context=context, tts_enabled=True)

    # ------------------------------------------------------------------
    # Streaming inference (real-time)
    # ------------------------------------------------------------------
    def get_streaming_engine(self) -> StreamingASREngine:
        if self._streaming_engine is None:
            self._streaming_engine = StreamingASREngine(
                backend=ASR_BACKEND,
                model_name=FASTER_WHISPER_MODEL,
                chunk_duration_ms=STREAM_CHUNK_MS,
                fusion_mode=FUSION_MODE,
            )
        return self._streaming_engine

    async def stream_transcribe(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming transcription from async audio chunk generator.
        Yields partial results, then final result.
        """
        engine = self.get_streaming_engine()
        engine.reset()

        async for chunk_bytes in audio_chunks:
            if len(chunk_bytes) < 2:
                continue
            samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            partial = engine.push_chunk(samples)
            if partial:
                normalized = self.normalizer.normalize(partial)
                if callback:
                    callback(normalized)
                yield {
                    "type": "partial",
                    "text": normalized,
                    "raw": partial,
                    "is_final": False,
                }

        final_raw = engine.flush()
        if final_raw:
            normalized = self.normalizer.normalize(final_raw)
            if callback:
                callback(normalized)
            yield {
                "type": "final",
                "text": normalized,
                "raw": final_raw,
                "is_final": True,
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def run_diagnostics(self) -> Dict[str, Any]:
        return run_all()

    def print_diagnostics(self):
        report = run_all()
        print_report(report)
        return report

    # ------------------------------------------------------------------
    # Correction store
    # ------------------------------------------------------------------
    def save_correction(self, result: Dict[str, Any], corrected_text: str, **meta) -> str:
        return self.store.append(
            audio_path=result.get("audio_path", ""),
            asr_raw=result.get("asr_raw", ""),
            mandarin_corrected=corrected_text,
            **meta
        )
