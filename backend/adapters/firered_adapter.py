"""
FireRedASR2-AED Adapter (Optimized v2)
=======================================
Wrapper for the 1.18B-parameter FireRedASR2-AED Wenzhou/Wu-dialect ASR model.
Half-precision (FP16) to fit in ~5.5GB RAM.

Optimizations:
  - Model args loaded from separate JSON, no full checkpoint scan needed
  - Single half-precision checkpoint loaded directly (no FP32 intermediate)
  - Lazy load: model only initialized on first transcribe call
  - Minimal memory footprint: no redundant copies
"""

import os
import time
import gc
import json
import torch
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ASRResult:
    text: str
    confidence: float
    language: Optional[str] = None
    duration_s: Optional[float] = None
    inference_time_s: Optional[float] = None
    model: str = "fireredasr2-aed"


# ---- lazy imports (heavy deps) ----
_model = None
_feat_extractor = None
_tokenizer = None

MODEL_DIR = None
ARGS_PATH = None
CHECKPOINT_PATH = None
HALF_CHECKPOINT_PATH = None


def _init_paths():
    global MODEL_DIR, ARGS_PATH, CHECKPOINT_PATH, HALF_CHECKPOINT_PATH
    adapter_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(adapter_dir))
    MODEL_DIR = os.path.join(
        project_root,
        "pretrained_models",
        "xukaituo",
        "FireRedASR2-AED",
    )
    ARGS_PATH = os.path.join(MODEL_DIR, "model_args.json")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model.pth.tar")
    HALF_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model_half.pth.tar")


def _load_model():
    """Lazy-load the FireRedASR2-AED model (half-precision)."""
    global _model, _feat_extractor, _tokenizer
    if _model is not None:
        return

    _init_paths()
    import sys
    adapter_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(adapter_dir))
    fireredasr_ui = os.path.join(os.path.dirname(project_root), "fireredasr-ui")
    sys.path.insert(0, fireredasr_ui)
    sys.path.insert(0, os.path.join(fireredasr_ui, "fireredasr"))

    from fireredasr.models.fireredasr_aed import FireRedAsrAed
    from fireredasr.data.asr_feat import ASRFeatExtractor
    from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

    torch.set_grad_enabled(False)
    # Set FP16 default BEFORE model creation to keep ~2.4GB resident
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    print("[FireRedAdapter v2] Loading model args from JSON...")
    with open(ARGS_PATH) as f:
        args_dict = json.load(f)

    # Reconstruct a namespace-like object for from_args
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    print("[FireRedAdapter v2] Creating model architecture (FP16)...")
    model = FireRedAsrAed.from_args(args)
    gc.collect()

    print("[FireRedAdapter v2] Loading half-precision checkpoint directly...")
    # The half checkpoint may contain model_state_dict + args;
    # weights_only=False is safe here as we control this file
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    sd = torch.load(HALF_CHECKPOINT_PATH, map_location="cpu", mmap=True, weights_only=False)
    # If the checkpoint has a top-level dict wrapping, unwrap
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    gc.collect()

    print("[FireRedAdapter v2] Injecting weights (in-place, no copy)...")
    msd = model.state_dict()
    for k in sd:
        if k in msd:
            msd[k].copy_(sd[k])
    del sd, msd
    gc.collect()

    model.eval()
    torch.set_default_dtype(old_dtype)

    print("[FireRedAdapter v2] Loading feature extractor & tokenizer...")
    feat = ASRFeatExtractor(os.path.join(MODEL_DIR, "cmvn.ark"))
    tok = ChineseCharEnglishSpmTokenizer(
        os.path.join(MODEL_DIR, "dict.txt"),
        os.path.join(MODEL_DIR, "train_bpe1000.model"),
    )

    _model = model
    _feat_extractor = feat
    _tokenizer = tok
    print("[FireRedAdapter v2] ✅ Model ready (FP16, ~2.4GB resident)!")


class FireredASREngine:
    """
    FireRedASR2-AED backend for Wu dialect (Wenzhouhua) ASR.
    Uses half-precision to fit in ~5.5GB RAM.
    Optimized for low-latency inference.
    """

    def __init__(self, **kwargs):
        _load_model()

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe a single audio file (16kHz mono wav)."""
        t0 = time.time()

        with torch.inference_mode():
            feats, lengths, durs = _feat_extractor([audio_path])
            feats = feats.half()

            hyps = _model.transcribe(
                feats, lengths,
                beam_size=1, nbest=1, decode_max_len=0,
                softmax_smoothing=1.25, length_penalty=0.6, eos_penalty=1.0,
            )

        elapsed = time.time() - t0

        text = ""
        for hyp in hyps:
            hyp0 = hyp[0]
            ids = [int(id) for id in hyp0["yseq"].cpu()]
            text = _tokenizer.detokenize(ids)
            break

        # Check if result looks like garbage/silence
        clean_text = text.strip()
        if clean_text in ("<sil>", "<sil>", "<blank>", "<unk>", ""):
            clean_text = ""

        return ASRResult(
            text=clean_text,
            confidence=1.0 if clean_text else 0.0,
            duration_s=durs[0] if durs else None,
            inference_time_s=round(elapsed, 2),
            model="fireredasr2-aed",
        )


if __name__ == "__main__":
    import sys
    audio = sys.argv[1] if len(sys.argv) > 1 else "test_sine.wav"
    engine = FireredASREngine()
    result = engine.transcribe(audio)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
