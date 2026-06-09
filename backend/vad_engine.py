"""
VAD Engine — Voice Activity Detection for Wenzhou ASR
======================================================
Lightweight VAD to filter out silence before ASR inference.

Reduces latency by only sending speech segments to the ASR backend.
Uses Silero VAD (tiny, fast, no GPU needed) when available, with
a simple energy-based fallback.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


class VADEngine:
    """
    Voice Activity Detection.

    Primary: Silero VAD (if torch + silero_vad installed).
    Fallback: simple energy-based.
    """

    def __init__(self):
        self._silero = None
        self._use_silero = False
        # Try to load silero-vad package (pip install silero-vad)
        try:
            import silero_vad
            self._silero = silero_vad
            self._use_silero = True
        except ImportError:
            pass
        # Skip torch.hub (hangs on trust prompt)
        # Pure energy-based VAD will be used as fallback

    def get_speech_segments(
        self,
        audio_path: str | Path,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        """Return list of (start_sec, end_sec) speech segments."""
        if self._use_silero and self._silero is not None:
            return self._vad_silero(audio_path, sample_rate)
        else:
            return self._vad_energy(audio_path, sample_rate)

    def _vad_silero(
        self,
        audio_path: str | Path,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        try:
            import torch
            wav = self._read_wav(audio_path, sample_rate)
            if wav is None or len(wav) < sample_rate * 0.1:  # < 100ms
                return [(0.0, 0.0)]
            wav_tensor = torch.from_numpy(wav).float()

            if hasattr(self._silero, 'get_speech_timestamps'):
                # Using silero_vad package
                segments = self._silero.get_speech_timestamps(
                    wav_tensor,
                    self._silero_model if hasattr(self, '_silero_model') else None,
                    sampling_rate=sample_rate,
                    threshold=0.5,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=100,
                )
                return [
                    (seg['start'] / sample_rate, seg['end'] / sample_rate)
                    for seg in segments
                ]
            else:
                return [(0.0, len(wav) / sample_rate)]
        except Exception:
            return [(0.0, float(self._get_duration(audio_path)))]

    def _vad_energy(
        self,
        audio_path: str | Path,
        sample_rate: int = 16000,
        threshold_db: float = -35.0,
        min_energy_duration_ms: int = 150,
    ) -> List[Tuple[float, float]]:
        """Simple energy-based VAD (works well for clean recordings)."""
        wav = self._read_wav(audio_path, sample_rate)
        if wav is None or len(wav) == 0:
            return [(0.0, 0.0)]

        frame_len = int(sample_rate * 0.03)  # 30ms frames
        hop_len = int(sample_rate * 0.01)    # 10ms hop

        threshold = 10 ** (threshold_db / 20.0)

        is_speech = []
        for start in range(0, len(wav) - frame_len + 1, hop_len):
            frame = wav[start:start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            is_speech.append(rms > threshold)

        # Merge contiguous speech frames
        segments = []
        in_speech = False
        speech_start = 0
        min_frames = min_energy_duration_ms // 10

        for i, speaking in enumerate(is_speech):
            if speaking and not in_speech:
                speech_start = i * hop_len / sample_rate
                in_speech = True
            elif not speaking and in_speech:
                if (i * hop_len / sample_rate - speech_start) >= min_energy_duration_ms * 0.001:
                    segments.append((speech_start, i * hop_len / sample_rate))
                in_speech = False

        if in_speech:
            segments.append((speech_start, len(wav) / sample_rate))

        if not segments:
            segments = [(0.0, len(wav) / sample_rate)]

        return segments

    def _read_wav(self, audio_path: str | Path, target_sr: int = 16000) -> Optional[np.ndarray]:
        """Read wav file and return float32 mono array at target_sr."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_path))
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                import scipy.signal
                new_len = int(len(data) * target_sr / sr)
                data = scipy.signal.resample(data, new_len)
            return data.astype(np.float32)
        except Exception:
            pass

        try:
            with wave.open(str(audio_path), 'rb') as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if sr != target_sr:
                    import scipy.signal
                    new_len = int(n * target_sr / sr)
                    data = scipy.signal.resample(data, new_len)
                return data
        except Exception:
            return None

    def _get_duration(self, audio_path: str | Path) -> float:
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """Quick check: is this chunk likely speech?"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > 10 ** (-35 / 20.0)
