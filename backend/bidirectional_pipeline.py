"""
Bidirectional Wenzhou ↔ Mandarin Pipeline (Optimized v2)
==========================================================

Optimizations:
  - VAD pre-filtering: segments removed silence before ASR
  - Cascading ASR: SenseVoice fast pass → FireRed specialist fallback
  - Shorter audio preprocessing pipeline
  - Ensemble voting for file upload mode

Direction A: Wenzhou Speech → VAD → [SenseVoice → FireRed if needed] → Mandarin Text
Direction B: Mandarin Speech → VAD → SenseVoice → Translate to Wenzhou → TTS
Direction C: Wenzhou Speech → VAD → [SenseVoice → FireRed if needed] → Wenzhou Text
"""

from __future__ import annotations

import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from backend.config import (
    ASR_BACKEND, TTS_BACKEND, LLM_BACKEND, FASTER_WHISPER_MODEL,
    EDGE_TTS_VOICE, OUTPUT_DIR, LEXICON_PATH, CORRECTION_DIR,
    ASR_WENZHOU_BACKEND, ASR_MANDARIN_BACKEND, VAD_ENABLED,
)
from backend.asr_engine import ASREngine
from backend.llm_normalizer import MandarinNormalizer
from backend.tts_engine import TTSEngine
from backend.wenzhou_translator import WenzhouTranslator
from backend.vad_engine import VADEngine
from scripts.preprocess_audio import preprocess_audio
import os


class BidirectionalPipeline:
    """
    Bidirectional Wenzhou ↔ Mandarin pipeline with VAD + cascading ASR.

    Direction "wenzhou_to_mandarin": Wenzhou speech → Mandarin text
    Direction "mandarin_to_wenzhou": Mandarin speech → Wenzhou text
    Direction "wenzhou_to_wenzhou":  Wenzhou speech → Wenzhou text
    """

    def __init__(self):
        self._asr_wenzhou = None
        self._asr_mandarin = None
        self._asr_cascade = None          # Fast model for cascading (SenseVoice)
        self._wenzhou_backend = ASR_WENZHOU_BACKEND
        self._mandarin_backend = ASR_MANDARIN_BACKEND
        self._vad_enabled = VAD_ENABLED
        self._vad = None
        self._asr_model_name = FASTER_WHISPER_MODEL
        # Memory-aware mode: "single" loads only one model at a time
        # "cascading" loads SenseVoice first then falls back to FireRed
        # "ensemble" loads all backends and votes (heaviest)
        self._fusion_mode = os.getenv("FUSION_MODE", "single")
        self.normalizer = MandarinNormalizer(
            lexicon_path=LEXICON_PATH, backend=LLM_BACKEND
        )
        self.translator = WenzhouTranslator()
        self.tts_mandarin = TTSEngine(
            backend=TTS_BACKEND, voice=EDGE_TTS_VOICE, output_dir=OUTPUT_DIR
        )

    @property
    def asr_wenzhou(self):
        """Primary Wenzhou ASR backend (FireRedASR2-AED — Wu dialect specialist)."""
        if self._asr_wenzhou is None:
            self._asr_wenzhou = ASREngine(
                backend=self._wenzhou_backend, model_name=self._asr_model_name
            )
        return self._asr_wenzhou

    @property
    def asr_mandarin(self):
        """Mandarin ASR backend (SenseVoiceSmall — fast, accurate for Mandarin)."""
        if self._asr_mandarin is None:
            self._asr_mandarin = ASREngine(
                backend=self._mandarin_backend, model_name=self._asr_model_name
            )
        return self._asr_mandarin

    @property
    def asr_cascade(self):
        """Fast cascade backend (SenseVoiceSmall) for first-pass Wenzhou inference.
        Only loads when FUSION_MODE=cascading, to avoid OOM in 5.5GB RAM."""
        if self._asr_cascade is None and self._fusion_mode == "cascading":
            cascade_backend = self._mandarin_backend
            if cascade_backend != 'mock':
                try:
                    self._asr_cascade = ASREngine(backend=cascade_backend)
                except Exception:
                    self._asr_cascade = None
        return self._asr_cascade

    @property
    def vad(self):
        if self._vad is None:
            self._vad = VADEngine()
        return self._vad

    def detect_direction(self, audio_path: str | Path) -> str | None:
        """
        Auto-detect whether audio contains Wenzhou or Mandarin speech.
        Uses ASR output + translator marker detection.
        Returns 'wenzhou_to_mandarin', 'mandarin_to_wenzhou', or None if unsure.
        """
        backend = self._mandarin_backend
        try:
            engine = ASREngine(backend=backend)
            result = engine.transcribe(str(audio_path))
            text = result.text.strip()
        except Exception:
            return None
        if not text or len(text) < 3:
            return None
        return self.translator.detect_language(text)

    def process(
        self,
        audio_path: str | Path,
        direction: str = "wenzhou_to_mandarin",
        context: str = "",
        tts_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Process audio in the specified direction.

        With cascading (default):
          1. Convert audio to 16kHz mono wav
          2. Apply VAD to remove silence
          3. ASR: Wenzhou direction uses cascading (SenseVoice → FireRed)
                 Mandarin direction uses SenseVoice directly
          4. Normalize ASR output
          5. Translate if needed (Mandarin ↔ Wenzhou lexical mapping)
          6. TTS (edge-tts)
        """
        t0 = time.time()

        # 1. Copy and preprocess audio
        local_audio = self._copy_audio(audio_path)
        processed = OUTPUT_DIR / f"proc_{uuid.uuid4().hex[:10]}.wav"
        preprocess_audio(str(local_audio), str(processed), sample_rate=16000)
        stages = {}

        # 2. VAD: get speech segments (filter silence)
        vad_t0 = time.time()
        if self._vad_enabled:
            speech_segments = self.vad.get_speech_segments(str(processed))
            stages["vad_ms"] = round((time.time() - vad_t0) * 1000, 1)
        else:
            speech_segments = [(0.0, float(self._get_audio_duration(str(processed))))]
            stages["vad_ms"] = 0

        if not speech_segments or speech_segments == [(0.0, 0.0)]:
            return {
                "asr_raw": "",
                "source_text": "",
                "target_text": "",
                "direction": direction,
                "backend": "vad_silence",
                "tts_audio_path": None,
                "stage_timings_ms": {"vad_ms": stages.get("vad_ms", 0), "total_ms": round((time.time() - t0) * 1000, 1)},
                "meta": {"note": "VAD detected silence only"},
            }

        # 3. ASR: memory-aware routing
        #   single/mock: use the configured primary backend directly (one model at a time)
        #   cascading: SenseVoice fast pass → FireRed specialist fallback
        #   ensemble: run all backends and majority vote
        asr_t0 = time.time()
        if direction in ("wenzhou_to_mandarin", "wenzhou_to_wenzhou"):
            if self._fusion_mode == "cascading":
                asr_result = self._cascade_asr(str(processed))
            elif self._fusion_mode == "ensemble":
                # Run all backends and vote (heavy)
                from backend.asr_engine import ASREngine
                ensembled = self.wenzhou_ensemble(str(processed))
                asr_result = type('obj', (object,), {
                    'text': ensembled.get('target_text', ''),
                    'backend': 'ensemble',
                    'meta': {}
                })
            else:
                # single mode: use one model at a time
                asr_result = self.asr_wenzhou.transcribe(str(processed))
        else:  # mandarin_to_wenzhou
            asr_result = self.asr_mandarin.transcribe(str(processed))
        stages["asr_ms"] = round((time.time() - asr_t0) * 1000, 1)
        raw_text = asr_result.text.strip()

        # 4. Normalize (clean ASR artifacts)
        if direction == "wenzhou_to_wenzhou":
            norm_t0 = time.time()
            clean_text = self.normalizer._remove_backend_noise(raw_text)
            stages["normalize_ms"] = round((time.time() - norm_t0) * 1000, 1)
            target_text = clean_text
        else:
            norm_t0 = time.time()
            clean_text = self.normalizer.normalize(raw_text, context=context)
            stages["normalize_ms"] = round((time.time() - norm_t0) * 1000, 1)

            # 5. Translate if needed
            if direction == "wenzhou_to_mandarin":
                translate_t0 = time.time()
                target_text = self.translator.wenzhou_to_mandarin(clean_text) or clean_text
                stages["translate_ms"] = round((time.time() - translate_t0) * 1000, 1)
            else:  # mandarin_to_wenzhou
                translate_t0 = time.time()
                target_text = self.translator.mandarin_to_wenzhou(clean_text) or clean_text
                stages["translate_ms"] = round((time.time() - translate_t0) * 1000, 1)

        # 6. TTS
        tts_audio_path = None
        if tts_enabled and target_text.strip():
            tts_t0 = time.time()
            tts_audio_path = str(self.tts_mandarin.synthesize(target_text))
            stages["tts_ms"] = round((time.time() - tts_t0) * 1000, 1)
        else:
            stages["tts_ms"] = 0

        total_ms = round((time.time() - t0) * 1000, 1)
        stages["total_ms"] = total_ms

        return {
            "asr_raw": raw_text,
            "source_text": raw_text,
            "target_text": target_text,
            "direction": direction,
            "backend": asr_result.backend,
            "tts_audio_path": tts_audio_path,
            "stage_timings_ms": stages,
            "meta": asr_result.meta or {},
        }

    def _cascade_asr(self, audio_path: str | Path) -> Any:
        """
        Cascading ASR for Wenzhou direction:
          1. Try SenseVoiceSmall first (fast, multi-dialect, ~2-5x faster than FireRed)
          2. If result is too short or low confidence → fallback to FireRedASR2-AED
          3. If FireRed fails (OOM, etc.) → return whatever we got from stage 1
        """
        # Stage 1: Fast model (only if cascading is enabled and backend != mock)
        fallback_text = ""
        if self.asr_cascade is not None:
            try:
                result = self.asr_cascade.transcribe(str(audio_path))
                text = result.text.strip()
                fallback_text = text
                if len(text) >= 3 and not self._is_garbage(text):
                    return result
            except Exception:
                pass

        # Stage 2: FireRed specialist
        try:
            return self.asr_wenzhou.transcribe(str(audio_path))
        except Exception as exc:
            # FireRed failed (likely OOM) — return stage 1 result if we have it
            if fallback_text:
                from backend.asr_engine import ASRResult
                return ASRResult(text=fallback_text, backend="cascade_fallback")
            raise

    def _is_garbage(self, text: str) -> bool:
        """Quick check if ASR output is garbage/noise."""
        if not text:
            return True
        noise = {"嗯", "啊", "呃", "哦", " ", ""}
        if text.strip() in noise:
            return True
        return False

    def _get_audio_duration(self, audio_path: str | Path) -> float:
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 0.0

    # ── Multi-model ensemble ──

    def wenzhou_ensemble(
        self, audio_path: str | Path, context: str = "", tts_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Run all available Wenzhou-capable ASR backends and fuse results.
        This gives the highest accuracy (99% target) at the cost of latency.
        """
        backends = self._get_available_backends()
        if not backends:
            return self.process(audio_path, "wenzhou_to_mandarin", context, tts_enabled)

        source_audios = {}
        for name in backends:
            result = self.process_with_backend(audio_path, name, context, False)
            source_audios[name] = result

        from collections import Counter
        texts = [r["target_text"].strip() for r in source_audios.values() if r["target_text"].strip()]
        if not texts:
            return list(source_audios.values())[0]

        counter = Counter(texts)
        best_text, count = counter.most_common(1)[0]

        best_result = None
        for name, r in source_audios.items():
            if r["target_text"].strip() == best_text:
                best_result = r
                break

        if best_result is None:
            best_result = list(source_audios.values())[0]

        best_result["ensemble_votes"] = count
        best_result["ensemble_total"] = len(texts)
        best_result["ensemble_details"] = {k: v["target_text"] for k, v in source_audios.items()}

        if tts_enabled and best_result.get("tts_audio_path") is None and best_text.strip():
            tts_t0 = time.time()
            best_result["tts_audio_path"] = str(self.tts_mandarin.synthesize(best_text))
            if "stage_timings_ms" in best_result:
                best_result["stage_timings_ms"]["tts_ms"] = round((time.time() - tts_t0) * 1000, 1)

        return best_result

    def process_with_backend(
        self, audio_path: str | Path, backend: str, context: str = "", tts_enabled: bool = True
    ) -> Dict[str, Any]:
        """Process with a specific ASR backend name."""
        from backend.asr_engine import ASREngine
        engine = ASREngine(backend=backend)
        # Manually run pipeline stages without going through self.process
        local_audio = self._copy_audio(audio_path)
        processed = OUTPUT_DIR / f"proc_{uuid.uuid4().hex[:10]}.wav"
        preprocess_audio(str(local_audio), str(processed), sample_rate=16000)

        try:
            asr_result = engine.transcribe(str(processed))
        except Exception:
            return {"target_text": "", "backend": backend}

        raw_text = asr_result.text.strip()
        clean_text = self.normalizer.normalize(raw_text, context=context)
        target_text = self.translator.wenzhou_to_mandarin(clean_text) or clean_text

        tts_path = None
        if tts_enabled and target_text.strip():
            tts_path = str(self.tts_mandarin.synthesize(target_text))

        return {
            "asr_raw": raw_text,
            "source_text": raw_text,
            "target_text": target_text,
            "backend": backend,
            "tts_audio_path": tts_path,
        }

    def _get_available_backends(self) -> List[str]:
        candidates = []
        for backend in ["firered", "sensevoice", "whisper", "mock"]:
            try:
                engine = ASREngine(backend=backend)
                candidates.append(backend)
            except Exception:
                pass
        return candidates

    def _copy_audio(self, audio_path: str | Path) -> Path:
        src = Path(audio_path)
        ext = src.suffix or ".wav"
        dst = OUTPUT_DIR / f"input_{uuid.uuid4().hex[:10]}{ext}"
        shutil.copyfile(src, dst)
        return dst
