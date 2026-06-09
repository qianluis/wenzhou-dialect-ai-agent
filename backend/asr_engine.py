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
    - faster_whisper: baseline using faster-whisper (if installed + HF access)
    - sensevoice: SenseVoiceSmall via funasr (~893MB, supports Chinese dialects)
    - whisper: OpenAI Whisper small via openai-whisper (~461MB, 99 languages)
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
        if self.backend == "sensevoice":
            return self._sensevoice_transcribe(audio_path)
        if self.backend == "whisper":
            return self._whisper_transcribe(audio_path)
        if self.backend == "funasr":
            return self._not_ready_backend("FunASR", audio_path)
        if self.backend == "firered":
            return self._firered_transcribe(audio_path)
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

    def _sensevoice_transcribe(self, audio_path: Path) -> ASRResult:
        try:
            from backend.adapters.sensevoice_adapter import SenseVoiceASREngine
        except Exception as exc:
            raise RuntimeError(
                "SenseVoice adapter not available. Run: pip install funasr torchaudio"
            ) from exc

        if self._model is None:
            self._model = SenseVoiceASREngine(
                model_name="iic/SenseVoiceSmall",
                device=os.getenv("ASR_DEVICE", "cpu"),
                vad=True,
            )

        result = self._model.transcribe(str(audio_path), language="auto", use_itn=True)
        return ASRResult(
            text=result["text"],
            backend="sensevoice",
            language=result["language"],
            meta={
                "model": result.get("model", "SenseVoiceSmall"),
                "time_seconds": result.get("time_seconds"),
                "segments": result.get("segments", []),
            }
        )

    def _whisper_transcribe(self, audio_path: Path) -> ASRResult:
        try:
            from backend.adapters.whisper_adapter import WhisperASREngine
        except Exception as exc:
            raise RuntimeError(
                "Whisper adapter not available. Run: pip install openai-whisper soundfile"
            ) from exc

        if self._model is None:
            self._model = WhisperASREngine(
                model_size=os.getenv("WHISPER_MODEL", "small"),
                device=os.getenv("ASR_DEVICE", "cpu"),
            )

        result = self._model.transcribe(
            str(audio_path),
            language=os.getenv("WHISPER_LANG", None),  # None = auto detect
            beam_size=int(os.getenv("WHISPER_BEAM", "5")),
        )
        return ASRResult(
            text=result["text"],
            backend="whisper",
            language=result["language"],
            meta={
                "model": result.get("model", "whisper-small"),
                "time_seconds": result.get("time_seconds"),
                "segments": result.get("segments", []),
            }
        )

    def _firered_transcribe(self, audio_path: Path) -> ASRResult:
        try:
            from backend.adapters.firered_adapter import FireredASREngine
        except Exception as exc:
            raise RuntimeError(
                "FireRed adapter not available. Check backend/adapters/firered_adapter.py"
            ) from exc

        if self._model is None:
            self._model = FireredASREngine()

        result = self._model.transcribe(str(audio_path))
        return ASRResult(
            text=result.text,
            backend="firered",
            language="zh",
            meta={
                "model": "FireRedASR2-AED",
                "inference_time_s": result.inference_time_s,
                "duration_s": result.duration_s,
            }
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
