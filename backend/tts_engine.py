from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import asyncio
import wave
import struct
import math


class TTSEngine:
    """
    Text-to-speech engine.

    - mock: generate a short silent wav so the UI stays runnable.
    - edge_tts: generate Mandarin speech through edge-tts if installed.
    """

    def __init__(self, backend: str = "mock", voice: str = "zh-CN-XiaoxiaoNeural", output_dir: str | Path = "outputs"):
        self.backend = backend
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str) -> Path:
        stem = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if self.backend == "edge_tts":
            out = self.output_dir / f"tts_{stem}.mp3"
            asyncio.run(self._edge_tts(text, out))
            return out

        out = self.output_dir / f"tts_{stem}.wav"
        self._mock_wav(out)
        return out

    async def _edge_tts(self, text: str, out_path: Path):
        try:
            import edge_tts
        except Exception as exc:
            raise RuntimeError("edge-tts is not installed. Install it or set TTS_BACKEND=mock.") from exc
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(str(out_path))

    def _mock_wav(self, out_path: Path, seconds: float = 0.4, sample_rate: int = 16000):
        # Low-volume sine beep instead of pure silence, so players can detect a valid audio stream.
        n = int(seconds * sample_rate)
        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n):
                value = int(800 * math.sin(2 * math.pi * 440 * i / sample_rate))
                wf.writeframes(struct.pack("<h", value))
