"""
TTS 引擎:普通话文本 → 语音(流式)。

第一阶段用 edge-tts(微软在线 TTS,走网络,无需 GPU),替代 CLAUDE.md 里的 CosyVoice2。
返回 mp3 字节流;前端 <audio> 直接播放 blob。
edge-tts 不可用时回退到 mock(一段正弦 beep 的 wav),保证 UI 不挂。
"""
from __future__ import annotations
from typing import AsyncGenerator
import io
import math
import struct
import wave

from . import config


async def synth_stream(text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]:
    voice = voice or config.EDGE_VOICE
    if config.TTS_BACKEND == "edge_tts":
        async for chunk in _edge_stream(text, voice):
            yield chunk
    else:
        yield _mock_wav()


async def _edge_stream(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    import edge_tts
    communicate = edge_tts.Communicate(text=text, voice=voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


def media_type() -> str:
    return "audio/mpeg" if config.TTS_BACKEND == "edge_tts" else "audio/wav"


def _mock_wav(seconds: float = 0.5, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    n = int(seconds * sr)
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            value = int(3000 * math.sin(2 * math.pi * 440 * i / sr))
            wf.writeframes(struct.pack("<h", value))
    return buf.getvalue()


def status() -> dict:
    return {"backend": config.TTS_BACKEND, "voice": config.EDGE_VOICE}
