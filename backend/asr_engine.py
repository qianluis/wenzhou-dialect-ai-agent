from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import os


@dataclass
class ASRResult:
    text: str
    backend: str
    language: str = "zh"
    meta: Optional[Dict[str, Any]] = None


class ASREngine:
    """
    Unified ASR interface.

    Available backends:
    - mock: deterministic demo output
    - faster_whisper: optional baseline if faster-whisper is installed
    - funasr: placeholder for production integration
    - firered: placeholder for FireRedASR/FireRedASR2S
    - telespeech: placeholder for TeleSpeech-ASR
    """

    def __init__(self, backend: str = "mock", model_name: str = "small"):
        self.backend = backend
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.backend == "mock":
            return self._mock_transcribe(audio_path)
        if self.backend == "faster_whisper":
            return self._faster_whisper_transcribe(audio_path)
        if self.backend == "funasr":
            return self._not_ready_backend("FunASR", audio_path)
        if self.backend == "firered":
            return self._not_ready_backend("FireRedASR/FireRedASR2S", audio_path)
        if self.backend == "telespeech":
            return self._not_ready_backend("TeleSpeech-ASR", audio_path)

        raise ValueError(f"Unsupported ASR backend: {self.backend}")

    def _mock_transcribe(self, audio_path: Path) -> ASRResult:
        # Keep this output intentionally imperfect so the normalizer has work to do.
        return ASRResult(
            text="我 mock 识别到一段温州话，意思大概是今天去医院办事情，医生让我明天再来。",
            backend="mock",
            meta={"audio": str(audio_path), "note": "demo only"}
        )

    def _faster_whisper_transcribe(self, audio_path: Path) -> ASRResult:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it or set ASR_BACKEND=mock."
            ) from exc

        if self._model is None:
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

        segments, info = self._model.transcribe(
            str(audio_path),
            language="zh",
            vad_filter=True,
            beam_size=5
        )
        text = "".join(seg.text for seg in segments).strip()
        return ASRResult(
            text=text,
            backend="faster_whisper",
            language=getattr(info, "language", "zh"),
            meta={"duration": getattr(info, "duration", None)}
        )

    def _not_ready_backend(self, name: str, audio_path: Path) -> ASRResult:
        """
        Placeholder keeps the project runnable. Replace this with real inference code.

        Recommended integration pattern:
        1. Put model loading in __init__ or a lazy loader.
        2. Convert audio to 16kHz mono wav before inference.
        3. Return ASRResult(text=..., backend=..., meta=...).
        """
        return ASRResult(
            text=f"[{name} placeholder] 请在 backend/asr_engine.py 中接入真实模型。当前音频：{audio_path.name}",
            backend=self.backend,
            meta={"status": "placeholder"}
        )
