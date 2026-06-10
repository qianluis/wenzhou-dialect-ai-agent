"""
ASR 引擎:把整句 16k/mono/PCM16 音频转成中文文本。

第一阶段用 faster-whisper(CTranslate2, CPU/int8),替代 CLAUDE.md 里的 Qwen3-ASR docker。
温州话当中文泛化识别(language=zh),准确率先记录基线——这正是 CLAUDE.md 的预期。
模型懒加载:首句到达时才加载,避免启动即占内存。
"""
from __future__ import annotations
import numpy as np

from . import config

_model = None
_load_error: str | None = None

# 可选繁转简(装了 opencc 才启用,否则原样返回)
try:
    import opencc
    _t2s = opencc.OpenCC("t2s")
except Exception:  # noqa: BLE001
    _t2s = None


def _to_simplified(text: str) -> str:
    return _t2s.convert(text) if _t2s else text


def _get_model():
    global _model, _load_error
    if _model is None and _load_error is None:
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(
                config.FW_MODEL, device=config.FW_DEVICE, compute_type=config.FW_COMPUTE
            )
        except Exception as exc:  # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
    return _model


def status() -> dict:
    """供 /health 用:不触发下载,只报当前后端与是否已加载。"""
    return {
        "backend": config.ASR_BACKEND,
        "model": config.FW_MODEL,
        "device": config.FW_DEVICE,
        "compute_type": config.FW_COMPUTE,
        "loaded": _model is not None,
        "load_error": _load_error,
    }


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_pcm(pcm: bytes) -> str:
    """整句 PCM16(16k mono) → 文本。空串表示没识别到内容。"""
    if config.ASR_BACKEND == "mock":
        # 故意留点口语/方言味,让规范化器有活干
        return "该日去医院,医生话我明朝再来"

    model = _get_model()
    if model is None:
        raise RuntimeError(f"faster-whisper 不可用: {_load_error}")

    audio = _pcm16_to_float32(pcm)
    if audio.size < config.SR * 0.2:  # < 0.2s 直接跳过
        return ""
    segments, _info = model.transcribe(
        audio,
        language="zh",
        beam_size=config.FW_BEAM,
        vad_filter=False,      # 上游已用 webrtcvad 断句,这里不再二次 VAD
        condition_on_previous_text=False,
        initial_prompt=config.ASR_INITIAL_PROMPT,  # 偏置简体普通话
    )
    text = "".join(seg.text for seg in segments).strip()
    return _to_simplified(text)
