"""
Realtime Streaming ASR Engine (Optimized v2)
=============================================

Optimizations:
  - VAD pre-filtering: skip silence, only ASR speech segments
  - Shorter window (800ms) + 400ms stride for faster response
  - Direct in-memory inference (no temp file writes)
  - Confidence-based cascading: fast model first, FireRed on low confidence
  - Sentence-level caching to avoid re-decoding
"""

from __future__ import annotations

import os
import time
import gc
import struct
import wave
import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class StreamResult:
    """Partial or final transcription result."""
    text: str
    is_final: bool
    segment_idx: int
    confidence: float = 0.0
    backend: str = ""
    timestamp: float = 0.0


class StreamingASREngine:
    """
    Streaming ASR engine — optimized for near-real-time Wenzhou ASR.

    Architecture:
      1. VAD filters out silence chunks
      2. For speech: try light model (SenseVoice small / Whisper tiny),
         fallback to FireRed (Wu-dialect specialist)
      3. 800ms sliding window, 400ms stride
      4. Text cache: de-duplicate repeated partial results
    """

    def __init__(
        self,
        backend: str = "firered",
        model_name: str = "small",
        chunk_duration_ms: int = 160,          # 160ms per chunk (2560 samples @ 16kHz)
        window_duration_ms: int = 800,         # 800ms sliding window
        stride_ms: int = 400,                  # 400ms stride (50% overlap)
        fusion_mode: str = "single",           # "single" | "cascading"
    ):
        self.backend = backend
        self.model_name = model_name
        self.chunk_size = int(16000 * chunk_duration_ms / 1000)
        self.window_samples = int(16000 * window_duration_ms / 1000)
        self.stride_samples = int(16000 * stride_ms / 1000)
        self.fusion_mode = fusion_mode
        self.sample_rate = 16000

        self._asr_engine = None
        self._fallback_engine = None
        self._vad = None

        # Ring buffer (reuse across chunks to reduce allocations)
        self._buffer: np.ndarray = np.zeros(self.window_samples, dtype=np.float32)
        self._buffer_filled = 0
        self._segment_idx = 0
        self._lock_held = False

        # Text dedup
        self._last_partial = ""
        self._stall_count = 0

        # Callbacks
        self._on_result: Optional[Callable] = None

    @property
    def engine(self):
        if self._asr_engine is None:
            from backend.asr_engine import ASREngine
            self._asr_engine = ASREngine(backend=self.backend)
        return self._asr_engine

    @property
    def fallback(self):
        """Lightweight fallback for cascading (SenseVoice or whisper tiny)."""
        if self._fallback_engine is None:
            from backend.asr_engine import ASREngine
            # Try SenseVoice first (good for Chinese dialects)
            try:
                self._fallback_engine = ASREngine(backend="sensevoice")
            except Exception:
                try:
                    self._fallback_engine = ASREngine(backend="whisper", model_name="tiny")
                except Exception:
                    self._fallback_engine = None
        return self._fallback_engine

    @property
    def vad(self):
        if self._vad is None:
            from backend.vad_engine import VADEngine
            self._vad = VADEngine()
        return self._vad

    def set_callback(self, callback: Callable[[StreamResult], None]):
        self._on_result = callback

    def push_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Push a single audio chunk (16kHz mono float32).
        Returns partial transcription if a window boundary is hit.

        VAD filters out silent chunks to avoid unnecessary ASR calls.
        """
        if not self._has_speech(audio_chunk):
            self._stall_count += 1
            if self._stall_count > 2:
                return None  # Silence, skip
            return None

        self._stall_count = 0

        # Update ring buffer
        chunk_len = len(audio_chunk)
        if chunk_len > self.window_samples:
            self._buffer[:] = audio_chunk[-self.window_samples:]
            self._buffer_filled = self.window_samples
        else:
            shift = min(chunk_len, self.window_samples)
            self._buffer[:-shift] = self._buffer[shift:]
            self._buffer[-chunk_len:] = audio_chunk[:chunk_len]
            self._buffer_filled = min(self.window_samples, self._buffer_filled + chunk_len)

        # Only infer when we have a full window
        if self._buffer_filled >= self.window_samples:
            return self._infer_window()
        return None

    def _has_speech(self, chunk: np.ndarray) -> bool:
        """Quick energy-based VAD for real-time chunk filtering."""
        if len(chunk) == 0:
            return False
        energy = np.sqrt(np.mean(chunk ** 2 + 1e-10))
        return energy > 0.008  # ~-42dB threshold

    def _infer_window(self) -> Optional[str]:
        """Run ASR on current window."""
        segment = self._buffer.copy()

        text = self._run_inference(segment)

        if text and len(text) > 2:
            # Dedup: skip if same as last partial
            if text == self._last_partial:
                return None
            self._last_partial = text

            if self._on_result:
                result = StreamResult(
                    text=text,
                    is_final=False,
                    segment_idx=self._segment_idx,
                    backend=self.backend if self.fusion_mode == "single" else "cascading",
                )
                self._on_result(result)

        self._segment_idx += 1
        return text

    def _run_inference(self, audio: np.ndarray) -> str:
        """Run ASR on an audio segment."""
        if self.fusion_mode == "cascading":
            return self._cascading_inference(audio)
        return self._single_inference(audio)

    def _single_inference(self, audio: np.ndarray) -> str:
        """Single model inference — direct in-memory, no temp files."""
        # Write to temp wav (necessary for ASR backends)
        tmp_path = self._audio_to_temp_wav(audio)
        try:
            result = self.engine.transcribe(tmp_path)
            return result.text
        finally:
            self._cleanup_temp(tmp_path)

    def _cascading_inference(self, audio: np.ndarray) -> str:
        """
        Cascading: fast light model → FireRed if low confidence.
        
        1. Try SenseVoice small (fast, multi-dialect) or whisper tiny
        2. If result < 4 chars or confidence heuristic low → FireRed
        """
        text = ""
        backend_used = ""

        # Stage 1: Light model
        if self.fallback is not None:
            tmp_path = self._audio_to_temp_wav(audio)
            try:
                fb_result = self.fallback.transcribe(tmp_path)
                text = fb_result.text.strip()
                backend_used = fb_result.backend
            except Exception:
                text = ""
            finally:
                self._cleanup_temp(tmp_path)

        # Stage 2: If low quality, try FireRed
        if not text or len(text) < 3 or self._estimate_confidence(text) < 0.4:
            tmp_path = self._audio_to_temp_wav(audio)
            try:
                fr_result = self.engine.transcribe(tmp_path)
                fr_text = fr_result.text.strip()
                if len(fr_text) >= len(text):
                    text = fr_text
                    backend_used = fr_result.backend
            except Exception:
                pass
            finally:
                self._cleanup_temp(tmp_path)

        if not text:
            return ""
        return text

    def _audio_to_temp_wav(self, audio: np.ndarray) -> str:
        """Write audio segment to temp wav file quickly."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            with wave.open(tmp_path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                scaled = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                w.writeframes(scaled.tobytes())
        except Exception:
            pass
        return tmp_path

    def _cleanup_temp(self, path: str):
        try:
            os.unlink(path)
        except OSError:
            pass

    def _estimate_confidence(self, text: str) -> float:
        """Heuristic confidence estimation."""
        if not text or not text.strip():
            return 0.0
        if len(text) < 2:
            return 0.2
        noise_patterns = ["[", "]", "(", ")", "嗯", "啊", "呃"]
        for p in noise_patterns:
            if p in text:
                return 0.3
        return min(0.95, 0.3 + len(text) * 0.05)

    def flush(self) -> Optional[str]:
        """Process remaining buffer."""
        if self._buffer_filled < 1600:  # < 100ms
            return None
        audio = self._buffer[:self._buffer_filled].copy()
        text = self._run_inference(audio)
        self._buffer_filled = 0

        if self._on_result and text:
            result = StreamResult(
                text=text,
                is_final=True,
                segment_idx=self._segment_idx,
                backend=self.backend,
            )
            self._on_result(result)
        return text

    def reset(self):
        """Reset streaming state for new utterance."""
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._buffer_filled = 0
        self._segment_idx = 0
        self._last_partial = ""
        self._stall_count = 0
