"""
Whisper (OpenAI) ASR Adapter for 温州话转普通话项目.

Model: whisper-small (~461MB)
- Supports 99 languages (including Chinese dialects)
- CPU friendly, uses FP32 on CPU
- Load model: whisper.load_model("small", device="cpu")
- Audio input: float32 numpy array at 16kHz

Usage:
    engine = WhisperASREngine(model_size="small")
    result = engine.transcribe("path/to/audio.wav")
    # => ASRResult(text="...", language="zh", ...)
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Model size -> local cache path (if pre-downloaded)
MODEL_CACHE = {
    "small": "/home/sandbox/.openclaw/workspace/repo/whisper_models/small.pt",
}

import soundfile as sf


class WhisperASREngine:
    """OpenAI Whisper ASR backend."""

    def __init__(self, model_size: str = "small",
                 device: str = "cpu",
                 model_path: Optional[str] = None):
        self.model_size = model_size
        self.device = device
        self._model = None

        # Allow override model path (for offline/cached models)
        self._model_path = model_path or MODEL_CACHE.get(model_size)

    def _ensure_model(self):
        if self._model is not None:
            return
        import whisper

        if self._model_path and os.path.isfile(self._model_path):
            logger.info(f"Loading Whisper {self.model_size} from local: {self._model_path}")
            t0 = time.time()
            self._model = whisper.load_model(self._model_path, device=self.device)
        else:
            logger.info(f"Loading Whisper {self.model_size} (download if needed)...")
            t0 = time.time()
            self._model = whisper.load_model(self.model_size, device=self.device)

        elapsed = time.time() - t0
        logger.info(f"Whisper {self.model_size} loaded in {elapsed:.1f}s")

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   beam_size: int = 5, **kwargs) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (wav/mp3/etc.)
            language: Language code ("zh", "en", "yue", etc.) or None for auto
            beam_size: Beam size for decoding

        Returns:
            dict with keys:
                - text: transcribed text
                - language: detected language
                - confidence: estimated confidence
                - segments: list of {start, end, text}
                - raw: full Whisper result
        """
        self._ensure_model()

        audio_path = str(audio_path)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing {audio_path} (lang={language})...")
        t0 = time.time()

        # Load audio with soundfile (works for wav; for other formats whisper
        # uses its own decoder which may need ffmpeg)
        try:
            audio_np, sr = sf.read(audio_path)
            audio_np = audio_np.astype('float32')
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
        except Exception as e:
            logger.warning(f"soundfile failed, falling back to whisper loader: {e}")
            import whisper
            audio_np = whisper.load_audio(audio_path)

        result = self._model.transcribe(
            audio_np,
            language=language,
            beam_size=beam_size,
            **kwargs,
        )

        elapsed = time.time() - t0

        text = result.get("text", "").strip()
        detected_lang = result.get("language", language or "zh")

        # Convert segments
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            })

        logger.info(f"Transcribed in {elapsed:.2f}s: {text[:80]}")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": 0.85,  # rough estimate
            "segments": segments,
            "raw": result,
            "model": f"whisper-{self.model_size}",
            "time_seconds": round(elapsed, 3),
        }

    def unload(self):
        """Free model memory."""
        self._model = None
        import gc
        gc.collect()
        logger.info(f"Whisper {self.model_size} unloaded.")
