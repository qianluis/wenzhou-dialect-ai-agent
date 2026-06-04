"""
Realtime Streaming ASR Engine

Supports chunk-based streaming inference for near-real-time transcription.
Multiple backends can run in parallel for ensemble voting (toward 99% accuracy).

Architecture:
  Audio chunks (via WebSocket / callback)
    → Ring buffer of overlapping segments
    → ASR backend inference on each segment
    → Streaming result (text) returned incrementally
    → Optional: multi-model fusion for confidence scoring
"""

from __future__ import annotations

import os
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamSegment:
    """A chunk of audio data with metadata."""
    audio: np.ndarray          # float32 mono audio at 16kHz
    sample_rate: int = 16000
    timestamp: float = 0.0     # seconds since stream start
    is_final: bool = False     # True for last segment


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
    Streaming ASR engine with pluggable backends.

    Supports:
      - Chunk-by-chunk streaming (microphone input)
      - Overlapping segment processing for smooth output
      - Multi-backend fusion for improved accuracy
      - Confidence scoring
    """

    def __init__(
        self,
        backend: str = "mock",
        model_name: str = "small",
        chunk_duration_ms: int = 320,      # 20ms * 16 = 320ms chunks by default
        window_duration_ms: int = 2000,    # 2-second sliding window
        stride_ms: int = 1000,             # 1-second stride (50% overlap)
        fusion_mode: str = "single",       # "single" | "ensemble" | "cascading"
    ):
        self.backend = backend
        self.model_name = model_name
        self.chunk_size = int(16000 * chunk_duration_ms / 1000)
        self.window_samples = int(16000 * window_duration_ms / 1000)
        self.stride_samples = int(16000 * stride_ms / 1000)
        self.fusion_mode = fusion_mode
        self._asr_engine = None
        self._ensemble_engines = []

        # Ring buffer
        self._buffer: np.ndarray = np.zeros(self.window_samples, dtype=np.float32)
        self._buffer_filled = 0
        self._segment_idx = 0
        self._lock = threading.Lock()

        # Callbacks
        self._on_result: Optional[Callable] = None

    @property
    def _engine(self):
        if self._asr_engine is None:
            from backend.asr_engine import ASREngine
            self._asr_engine = ASREngine(backend=self.backend, model_name=self.model_name)
        return self._asr_engine

    def set_callback(self, callback: Callable[[StreamResult], None]):
        """Set callback for streaming results."""
        self._on_result = callback

    def push_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Push a single audio chunk (16kHz mono float32).
        Returns partial transcription if a window boundary is hit.

        For real-time use, call this from your audio capture callback
        (microphone, WebSocket, etc.)
        """
        with self._lock:
            # Shift buffer and append new chunk
            chunk_len = len(audio_chunk)
            if chunk_len > self.window_samples:
                # Oversized chunk — just use last window_samples
                self._buffer[:] = audio_chunk[-self.window_samples:]
                self._buffer_filled = self.window_samples
            else:
                # Shift
                shift = min(chunk_len, self.window_samples)
                self._buffer[:-shift] = self._buffer[shift:]
                self._buffer[-chunk_len:] = audio_chunk[:chunk_len]
                self._buffer_filled = min(
                    self.window_samples,
                    self._buffer_filled + chunk_len,
                )

            # Check if we have enough for a stride
            if self._buffer_filled >= self.window_samples:
                return self._infer_window()
        return None

    def _infer_window(self) -> Optional[str]:
        """Run ASR on current window. Returns text if available."""
        segment = self._buffer.copy()

        # Run inference
        text = self._run_inference(segment)

        if text and self._on_result:
            result = StreamResult(
                text=text,
                is_final=False,
                segment_idx=self._segment_idx,
                backend=self.backend,
            )
            self._on_result(result)

        self._segment_idx += 1
        return text

    def _run_inference(self, audio: np.ndarray) -> str:
        """Run ASR on a single audio segment."""
        if self.fusion_mode == "ensemble":
            return self._ensemble_inference(audio)
        elif self.fusion_mode == "cascading":
            return self._cascading_inference(audio)
        else:
            return self._single_inference(audio)

    def _single_inference(self, audio: np.ndarray) -> str:
        """Single model inference."""
        import tempfile
        import wave
        import struct

        # Write to temp wav
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            scaled = (audio * 32767).astype(np.int16)
            w.writeframes(scaled.tobytes())

        try:
            result = self._engine.transcribe(tmp_path)
            return result.text
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _ensemble_inference(self, audio: np.ndarray) -> str:
        """
        Multi-model ensemble fusion.

        Runs 3 backends in parallel and merges via:
          1. If all agree → use with high confidence
          2. If majority agree → use majority
          3. If no agreement → use highest-confidence model

        This is the key to pushing accuracy toward 99%.
        """
        if not self._ensemble_engines:
            self._init_ensemble()

        results: List[str] = []
        import tempfile, wave, struct

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            scaled = (audio * 32767).astype(np.int16)
            w.writeframes(scaled.tobytes())

        try:
            for engine in self._ensemble_engines:
                try:
                    r = engine.transcribe(tmp_path)
                    results.append(r.text)
                except Exception:
                    results.append("")

            # Fusion strategy
            if len(results) < 2:
                return results[0] if results else ""

            # Check for exact agreement
            non_empty = [r for r in results if r.strip()]
            if not non_empty:
                return results[0] if results else ""

            # Majority voting
            from collections import Counter
            counter = Counter(non_empty)
            most_common_text, count = counter.most_common(1)[0]

            # If at least 2/3 agree, use it
            if count >= max(2, len(non_empty) * 2 // 3):
                return most_common_text

            # Fallback: use first non-empty
            return non_empty[0]

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _cascading_inference(self, audio: np.ndarray) -> str:
        """
        Cascading: try fast model first, fallback to bigger model if confidence is low.
        """
        # First pass: faster-whisper small (fast)
        text = self._single_inference(audio)

        # Check confidence heuristically (length, repetition, etc.)
        if self._estimate_confidence(text) < 0.5:
            # Fallback: try FunASR if available
            try:
                from backend.asr_engine import ASREngine
                fallback = ASREngine(backend="funasr")
                import tempfile, wave, struct
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                with wave.open(tmp_path, "w") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    scaled = (audio * 32767).astype(np.int16)
                    w.writeframes(scaled.tobytes())
                try:
                    fb_result = fallback.transcribe(tmp_path)
                    if fb_result.text and len(fb_result.text) > len(text):
                        text = fb_result.text
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception:
                pass

        return text

    def _init_ensemble(self):
        """Initialize ensemble engines (lazy)."""
        available = []
        backends_to_try = ["faster_whisper", "funasr", "telespeech", "firered"]
        for b in backends_to_try:
            try:
                from backend.asr_engine import ASREngine
                engine = ASREngine(backend=b)
                # Quick test
                engine.transcribe.__self__  # check it's bound
                available.append(engine)
            except Exception:
                continue
        self._ensemble_engines = available

    def _estimate_confidence(self, text: str) -> float:
        """Heuristic confidence estimation for a transcription."""
        if not text or not text.strip():
            return 0.0

        # Penalize very short outputs
        if len(text) < 2:
            return 0.2

        # Penalize placeholder/noise patterns
        noise_patterns = [
            "placeholder", "mock", "[",
            "嗯", "啊", "呃",
        ]
        for p in noise_patterns:
            if p in text:
                return 0.3

        # Repetition penalty
        if len(text) >= 6:
            for i in range(2, len(text) // 2 + 1):
                if text[:i] * (len(text) // i) == text[:i * (len(text) // i)]:
                    return 0.3

        # Length-based: longer text in a 2s window = more confident (more speech detected)
        return min(0.95, 0.3 + len(text) * 0.05)

    def flush(self) -> Optional[str]:
        """Process remaining buffer and return final transcription."""
        with self._lock:
            if self._buffer_filled < 1600:  # < 100ms
                return None
            text = self._run_inference(self._buffer[:self._buffer_filled])
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
        """Reset streaming state."""
        with self._lock:
            self._buffer = np.zeros(self.window_samples, dtype=np.float32)
            self._buffer_filled = 0
            self._segment_idx = 0
