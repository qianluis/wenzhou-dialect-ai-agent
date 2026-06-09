"""
Lightweight FastAPI server — ultra memory safe.
Models loaded on *first request*, not at startup.
"""
from __future__ import annotations
import os, sys, json, time, tempfile, shutil
from pathlib import Path
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ⚠️ DO NOT import torch/funasr/backend here — delay all heavy imports
_pipeline = None
_pipeline_lock = None  # created on demand

def _get_pipeline():
    global _pipeline, _pipeline_lock
    if _pipeline is None:
        # Lazily create lock and import
        import threading
        if _pipeline_lock is None:
            _pipeline_lock = threading.Lock()
        with _pipeline_lock:
            if _pipeline is None:
                # Import here, not at module level
                from backend.bidirectional_pipeline import BidirectionalPipeline
                _pipeline = BidirectionalPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup: nothing heavy
    # shutdown
    global _pipeline, _pipeline_lock
    _pipeline = None
    _pipeline_lock = None
    import gc
    gc.collect()


app = FastAPI(title="Wenzhou⇄Mandarin", version="4.0", lifespan=lifespan)

# ── Static ──
app.mount("/static", StaticFiles(directory=str(ROOT / "frontend" / "static")), name="static")


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "frontend" / "static" / "index.html").read_text())


@app.post("/api/detect")
async def api_detect(audio: UploadFile = File(...)):
    """Auto-detect: Wenzhou speech or Mandarin speech?"""
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            audio_path = tmp.name
        from backend.bidirectional_pipeline import BidirectionalPipeline
        p = _get_pipeline()
        direction = p.detect_direction(audio_path)
        return {"direction": direction, "note": "auto-detected from ASR output"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.unlink(audio_path)
            except: pass


@app.get("/api/lexicon/count")
async def lexicon_count():
    """Return lexicon stats for frontend display."""
    from backend.wenzhou_translator import _MANDARIN_TO_WENZHOU, _WENZHOU_TO_MANDARIN
    return {
        "mandarin_to_wenzhou": len(_MANDARIN_TO_WENZHOU),
        "wenzhou_to_mandarin": len(_WENZHOU_TO_MANDARIN),
    }


@app.post("/api/translate")
async def api_translate(
    audio: UploadFile = File(...),
    direction: str = Form("wenzhou_to_mandarin"),
    context: str = Form(""),
):
    t_start = time.time()
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            audio_path = tmp.name

        p = _get_pipeline()
        result = p.process(audio_path, direction=direction, context=context or "",
                           tts_enabled=True)

        timings = result.get("stage_timings_ms", {})
        timings["total_ms"] = int((time.time() - t_start) * 1000)

        tts_url = None
        tts_path = result.get("tts_audio_path")
        if tts_path and os.path.exists(tts_path):
            tts_url = f"/api/tts/{os.path.basename(tts_path)}"

        return {
            "asr_raw": result.get("asr_raw", ""),
            "target_text": result.get("target_text", ""),
            "backend": result.get("backend", "?"),
            "stage_timings": timings,
            "tts_url": tts_url,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


@app.get("/api/tts/{filename}")
async def tts_audio(filename: str):
    path = ROOT / "outputs" / filename
    if path.exists():
        return FileResponse(str(path), media_type="audio/wav")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/results/raw")
async def raw_result():
    try:
        return HTMLResponse(open("/tmp/wenzhou_transcription.txt").read()[:5000])
    except:
        return HTMLResponse("(无数据)")

@app.get("/api/results/translated")
async def translated_result():
    try:
        return HTMLResponse(open("/tmp/wenzhou_mandarin_translated.txt").read()[:5000])
    except:
        return HTMLResponse("(无数据)")


@app.get("/api/diagnostics")
async def diagnostics():
    import psutil
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "memory_gb": {"total": round(mem.total/1024**3,1), "available": round(mem.available/1024**3,1), "percent": mem.percent},
        "disk_gb": {"free": round(disk.free/1024**3,1)},
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n{'='*45}")
    print(f"  🌐 温州话 ↔ 普通话 翻译服务")
    print(f"  {'='*45}")
    print(f"  本地: http://localhost:{port}")
    print(f"  {'='*45}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
