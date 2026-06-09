"""
SenseVoiceSmall ASR Adapter for 温州话转普通话项目.

Model: iic/SenseVoiceSmall (通义实验室)
- Model size: ~893MB (model.pt)
- Supports: 中文(含方言)、英语、日语、韩语、粤语
- CPU/GPU: CPU 可运行，~5.5GB RAM OK
- Install: pip install funasr torchaudio
- Model: AutoModel(model='iic/SenseVoiceSmall', device='cpu')

Usage:
    engine = SenseVoiceASREngine()
    result = engine.transcribe("path/to/audio.wav")
    # => ASRResult(text="...", language="zh", confidence=0.95, ...)
"""

import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SenseVoiceASREngine:
    """SenseVoiceSmall ASR backend via funasr AutoModel."""

    def __init__(self, model_name: str = "iic/SenseVoiceSmall",
                 device: str = "cpu", vad: bool = True):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._vad_enabled = vad

    def _ensure_model(self):
        if self._model is not None:
            return
        from funasr import AutoModel

        kwargs = dict(
            model=self.model_name,
            device=self.device,
            disable_update=True,
        )
        if self._vad_enabled:
            kwargs["vad_model"] = "fsmn-vad"
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

        logger.info(f"Loading SenseVoiceSmall model (device={self.device})...")
        t0 = time.time()
        self._model = AutoModel(**kwargs)
        elapsed = time.time() - t0
        logger.info(f"SenseVoiceSmall loaded in {elapsed:.1f}s")

    def transcribe(self, audio_path: str, language: str = "auto",
                   use_itn: bool = True) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (16kHz mono wav recommended)
            language: Language hint ("auto", "zh", "yue", "en", etc.)
            use_itn: Apply inverse text normalization

        Returns:
            dict with keys:
                - text: transcribed text
                - language: detected language
                - confidence: estimated confidence (0-1)
                - segments: list of {start, end, text} or empty
                - raw: raw model output
        """
        self._ensure_model()

        audio_path = str(audio_path)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing {audio_path} (lang={language})...")
        t0 = time.time()

        result = self._model.generate(
            input=audio_path,
            cache={},
            language=language,
            use_itn=use_itn,
        )

        elapsed = time.time() - t0

        # Parse result
        raw_text = ""
        if result and isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get("text", "")

        # SenseVoice output format: <|lang|><|EMO_xxx|><|xxx|>text
        detected_lang = "zh"
        text = raw_text
        import re
        lang_match = re.search(r"<\|(\w+)\|>", text)
        if lang_match:
            detected_lang = lang_match.group(1)

        # Strip tags
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        logger.info(f"Transcribed in {elapsed:.2f}s: {text[:80]}")

        return {
            "text": text,
            "language": detected_lang,
            "confidence": 0.85,  # rough estimate
            "segments": [{"start": 0, "end": 0, "text": text}] if text else [],
            "raw": result,
            "model": "SenseVoiceSmall",
            "time_seconds": round(elapsed, 3),
        }

    def unload(self):
        """Free model memory."""
        self._model = None
        import gc
        gc.collect()
        logger.info("SenseVoiceSmall model unloaded.")
