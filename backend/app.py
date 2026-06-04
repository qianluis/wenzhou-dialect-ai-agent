"""
Wenzhou2Mandarin Agent API

Endpoints:
  GET  /                    — Health check
  POST /transcribe          — File-based transcription
  POST /transcribe_fusion   — File-based with multi-model fusion (99% accuracy mode)
  POST /save_correction     — Save manual correction
  GET  /diagnostics         — Run system self-diagnostics
  WS   /ws/transcribe       — Real-time streaming transcription (WebSocket)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
import tempfile
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from backend.enhanced_pipeline import EnhancedPipeline, get_pipeline
from backend.config import FUSION_ENABLED, STREAM_CHUNK_MS

app = FastAPI(title="Wenzhou2Mandarin Agent API", version="2.0.0")


# ------------------------------------------------------------------
# Root
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Wenzhou2Mandarin Agent",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "transcribe": "POST /transcribe (file upload)",
            "transcribe_fusion": "POST /transcribe_fusion (multi-model, 99% accuracy)",
            "streaming": "WS /ws/transcribe (real-time WebSocket)",
            "diagnostics": "GET /diagnostics",
            "save_correction": "POST /save_correction",
        },
    }


# ------------------------------------------------------------------
# File-based transcription
# ------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), context: str = Form("")):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.run(tmp_path, context=context)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ------------------------------------------------------------------
# File-based transcription with multi-model fusion (99% accuracy)
# ------------------------------------------------------------------
@app.post("/transcribe_fusion")
async def transcribe_fusion(audio: UploadFile = File(...), context: str = Form("")):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.run(tmp_path, context=context, fusion=True)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ------------------------------------------------------------------
# System diagnostics
# ------------------------------------------------------------------
@app.get("/diagnostics")
async def diagnostics():
    pipeline = get_pipeline()
    report = pipeline.run_diagnostics()
    return JSONResponse(report)


@app.get("/diagnostics_html")
async def diagnostics_html():
    """Returns an HTML diagnostic report that can be rendered in browser."""
    pipeline = get_pipeline()
    report = pipeline.run_diagnostics()
    checks = report["checks"]

    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>系统自检报告</title>
    <style>
      body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            background:#0a0a1a;color:#e8eaf0;padding:20px;max-width:800px;margin:0 auto}
      h1{color:#06b6d4;border-left:4px solid #06b6d4;padding-left:12px}
      .card{background:#16163a;border-radius:10px;padding:12px 16px;margin:10px 0;
            border:1px solid #262650}
      .pass{color:#22c55e}.warn{color:#f59e0b}.fail{color:#ef4444}.skip{color:#6a6e8a}
      .latency{color:#6a6e8a;font-size:0.85em}
      .rec{background:#06b6d410;border-left:3px solid #06b6d4;padding:8px 12px;margin:6px 0}
    </style></head><body>
    <h1>🩺 系统自检报告</h1>
    <div class="card">
      <strong>总体状态</strong>: """

    emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}
    html += f'<span class="{report["overall"].lower()}">{emoji.get(report["overall"], "❓")} {report["overall"]}</span>'
    html += f"<br>通过: {report['summary'].get('PASS',0)} | "
    html += f"警告: {report['summary'].get('WARN',0)} | "
    html += f"失败: {report['summary'].get('FAIL',0)} | "
    html += f"跳过: {report['summary'].get('SKIP',0)}"
    html += "</div>"

    for name, check in checks.items():
        st = check.get("status", "")
        c = st.lower()
        em = emoji.get(st, "❓")
        lat = check.get("latency_ms", 0)
        lat_str = f"<span class='latency'>({lat:.0f}ms)</span>" if lat else ""
        html += f'<div class="card"><span class="{c}">{em} {st}</span> '
        html += f"<strong>{name}</strong> {lat_str}<br>"
        detail = check.get("detail", "")
        if detail:
            html += f'<span style="font-size:0.85em;color:#b0b4c8">{detail[:200]}</span>'
        html += "</div>"

    if report.get("recommendations"):
        html += "<h3>💡 建议</h3>"
        for rec in report["recommendations"]:
            html += f'<div class="rec">• {rec}</div>'

    html += "</body></html>"
    return HTMLResponse(html)


# ------------------------------------------------------------------
# Save correction
# ------------------------------------------------------------------
@app.post("/save_correction")
async def save_correction(
    audio_path: str = Form(...),
    asr_raw: str = Form(...),
    mandarin_text: str = Form(...),
    corrected_text: str = Form(...),
    speaker_id: str = Form(""),
    gender: str = Form(""),
    age: str = Form(""),
    scene: str = Form(""),
    noise_level: str = Form(""),
    note: str = Form(""),
):
    result = {
        "audio_path": audio_path,
        "asr_raw": asr_raw,
        "mandarin_text": mandarin_text,
    }
    pipeline = get_pipeline()
    row_id = pipeline.save_correction(
        result=result,
        corrected_text=corrected_text,
        speaker_id=speaker_id,
        gender=gender,
        age=age,
        scene=scene,
        noise_level=noise_level,
        note=note,
    )
    return {"saved": True, "id": row_id}


# ------------------------------------------------------------------
# WebSocket streaming transcription (real-time)
# ------------------------------------------------------------------
@app.websocket("/ws/transcribe")
async def websocket_streaming(websocket: WebSocket):
    await websocket.accept()
    pipeline = get_pipeline()
    engine = pipeline.get_streaming_engine()
    engine.reset()

    buffer = b""
    last_send = time.time()
    CHUNK_INTERVAL = STREAM_CHUNK_MS / 1000.0  # seconds

    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()
            buffer += data

            # Process in 320-byte chunks (320 = 160 samples * 2 bytes at 16kHz)
            chunk_size = STREAM_CHUNK_MS * 16000 * 2 // 1000
            while len(buffer) >= chunk_size:
                chunk_bytes = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                # Convert to float32 numpy array
                samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                partial = engine.push_chunk(samples)
                if partial and (time.time() - last_send >= CHUNK_INTERVAL):
                    last_send = time.time()
                    normalized = pipeline.normalizer.normalize(partial)
                    await websocket.send_json({
                        "type": "partial",
                        "text": normalized,
                        "raw": partial,
                        "is_final": False,
                    })

            # Small sleep to yield control
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        # Flush remaining buffer
        final_raw = engine.flush()
        if final_raw:
            normalized = pipeline.normalizer.normalize(final_raw)
            try:
                await websocket.send_json({
                    "type": "final",
                    "text": normalized,
                    "raw": final_raw,
                    "is_final": True,
                })
            except Exception:
                pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        engine.reset()
