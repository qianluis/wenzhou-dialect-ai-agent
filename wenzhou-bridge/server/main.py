"""
温州话桥 编排层(单进程版)。

CLAUDE.md 的第一阶段架构是 4 个服务(ASR/规范化/TTS/编排 各占一端口),那套要 GPU。
本机无 GPU,这里把推理引擎内联进同一个 FastAPI 进程,用 CPU/网络引擎实现同一条 A 方向链路:

  浏览器(麦克风 16k PCM, WS 推流)
        │
        ▼
  /ws/asr : webrtcvad 断句 → faster-whisper 识别 → 规则规范化 → 回文字
  /tts    : 普通话文本 → edge-tts → 流式音频回浏览器

接口/字段与 CLAUDE.md 给的骨架保持一致(type=final, raw, mandarin),便于以后换回 GPU 服务。
"""
from __future__ import annotations
import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import config, asr, tts, normalize

app = FastAPI(title="温州话桥 wenzhou-bridge", version="0.1.0")

FRAME_BYTES = int(config.SR * config.FRAME_MS / 1000) * 2     # 16k/30ms/PCM16 = 960
MIN_VOICED_BYTES = int(config.SR * config.MIN_VOICED_MS / 1000) * 2

# webrtcvad 优先;装不上时回退到能量阈值断句,保证链路不依赖编译环境
try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except Exception:  # noqa: BLE001
    _HAS_WEBRTCVAD = False


class _EnergyVad:
    """webrtcvad 缺失时的兜底:基于帧能量(RMS)判断有无语音。"""
    def __init__(self, threshold: int = 500):
        self.threshold = threshold

    def is_speech(self, frame: bytes, sr: int) -> bool:
        import numpy as np
        a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
        return rms > self.threshold


def _make_vad():
    if _HAS_WEBRTCVAD:
        return webrtcvad.Vad(config.VAD_AGGR)
    return _EnergyVad()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr": asr.status(),
        "tts": tts.status(),
        "normalize": {
            "enabled": config.USE_NORM,
            "backend": config.NORM_BACKEND,
            "lexicon_count": normalize.lexicon_count(),
        },
        "vad": "webrtcvad" if _HAS_WEBRTCVAD else "energy_fallback",
    }


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    vad = _make_vad()
    voiced = bytearray()
    silence = 0
    leftover = b""

    async def flush_sentence():
        nonlocal voiced
        if len(voiced) < MIN_VOICED_BYTES:
            voiced = bytearray()
            return
        pcm = bytes(voiced)
        voiced = bytearray()
        await ws.send_text(json.dumps({"type": "recognizing"}, ensure_ascii=False))
        try:
            raw = await asyncio.to_thread(asr.transcribe_pcm, pcm)
        except Exception as exc:  # noqa: BLE001
            await ws.send_text(json.dumps({"type": "error", "msg": str(exc)}, ensure_ascii=False))
            return
        if not raw:
            return
        mand = await asyncio.to_thread(normalize.normalize, raw)
        await ws.send_text(json.dumps(
            {"type": "final", "raw": raw, "mandarin": mand}, ensure_ascii=False))

    try:
        while True:
            chunk = await ws.receive_bytes()
            buf = leftover + chunk
            while len(buf) >= FRAME_BYTES:
                frame, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                if vad.is_speech(frame, config.SR):
                    voiced += frame
                    silence = 0
                elif voiced:
                    silence += 1
                    if silence >= config.SIL_LIMIT:
                        await flush_sentence()
                        silence = 0
            leftover = buf
    except WebSocketDisconnect:
        # 收尾:把残余语音也识别掉(连接已断,仅尽力)
        if len(voiced) >= MIN_VOICED_BYTES:
            try:
                raw = await asyncio.to_thread(asr.transcribe_pcm, bytes(voiced))
                if raw:
                    print("[tail]", raw)
            except Exception:  # noqa: BLE001
                pass


class TTSReq(BaseModel):
    text: str
    voice: str | None = None


@app.post("/tts")
async def tts_endpoint(req: TTSReq):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    async def gen():
        async for b in tts.synth_stream(text, req.voice):
            yield b

    return StreamingResponse(gen(), media_type=tts.media_type())


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (config.WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
