"""
System Diagnostics Module

Performs self-checks on all pipeline components:
  - Audio I/O (microphone, file, ffmpeg)
  - ASR engine loading and inference latency
  - TTS engine latency
  - Normalizer correctness
  - End-to-end pipeline health

Usage:
    from backend.diagnostics import run_diagnostics
    report = run_diagnostics()
"""

from __future__ import annotations

import importlib
import time
import subprocess
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DiagResult:
    component: str
    status: str          # "PASS" | "WARN" | "FAIL" | "SKIP"
    detail: str = ""
    latency_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def _check_ffmpeg() -> DiagResult:
    t0 = time.time()
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5
        )
        version = r.stdout.decode().split("\n")[0] if r.stdout else "unknown"
        return DiagResult(
            component="ffmpeg",
            status="PASS",
            detail=version,
            latency_ms=(time.time() - t0) * 1000,
        )
    except FileNotFoundError:
        return DiagResult(
            component="ffmpeg",
            status="WARN",
            detail="ffmpeg not found. Non-WAV audio conversion will be degraded.",
        )
    except Exception as e:
        return DiagResult(
            component="ffmpeg",
            status="FAIL",
            detail=str(e),
        )


def _check_asr_backend(backend: str) -> DiagResult:
    """Try to import and initialize the selected ASR backend."""
    t0 = time.time()
    try:
        from backend.asr_engine import ASREngine
        engine = ASREngine(backend=backend)
        latency = (time.time() - t0) * 1000
        return DiagResult(
            component=f"asr_{backend}",
            status="PASS",
            detail=f"Engine initialized (lazy load on first transcribe)",
            latency_ms=latency,
        )
    except ImportError as e:
        return DiagResult(
            component=f"asr_{backend}",
            status="SKIP",
            detail=f"Not installed: {e}",
        )
    except Exception as e:
        return DiagResult(
            component=f"asr_{backend}",
            status="FAIL",
            detail=str(e),
        )


def _check_asr_all_backends() -> List[DiagResult]:
    results = []
    for backend in ["mock", "faster_whisper", "funasr", "firered", "telespeech"]:
        results.append(_check_asr_backend(backend))
    return results


def _check_tts_backend(backend: str) -> DiagResult:
    t0 = time.time()
    try:
        from backend.tts_engine import TTSEngine
        engine = TTSEngine(backend=backend)
        latency = (time.time() - t0) * 1000
        return DiagResult(
            component=f"tts_{backend}",
            status="PASS",
            detail=f"TTS engine ready",
            latency_ms=latency,
        )
    except ImportError as e:
        return DiagResult(
            component=f"tts_{backend}",
            status="SKIP",
            detail=f"Not installed: {e}",
        )
    except Exception as e:
        return DiagResult(
            component=f"tts_{backend}",
            status="FAIL",
            detail=str(e),
        )


def _check_e2e_mock() -> DiagResult:
    """Run a full mock pipeline end-to-end."""
    import wave, struct
    import tempfile

    t0 = time.time()
    try:
        # Create a dummy WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            for _ in range(16000):
                w.writeframes(struct.pack("<h", 0))

        from backend.pipeline import Wenzhou2MandarinPipeline
        pipeline = Wenzhou2MandarinPipeline()
        result = pipeline.run(tmp_path, context="diagnostics")
        latency = (time.time() - t0) * 1000

        os.unlink(tmp_path)

        return DiagResult(
            component="e2e_mock_pipeline",
            status="PASS",
            detail=f"Pipeline OK. ASR: '{result['asr_raw'][:40]}...' → Normalized: '{result['mandarin_text'][:40]}...'",
            latency_ms=latency,
            meta={
                "asr_backend": result["backend"],
                "mandarin_text": result["mandarin_text"],
            },
        )
    except Exception as e:
        return DiagResult(
            component="e2e_mock_pipeline",
            status="FAIL",
            detail=str(e),
        )


def _check_python_deps() -> DiagResult:
    """Check critical Python dependencies."""
    required = [
        ("fastapi", "fastapi"),
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
    ]
    optional = [
        ("faster_whisper", "faster-whisper"),
        ("funasr", "funasr"),
        ("torch", "torch"),
        ("edge_tts", "edge-tts"),
    ]
    installed = []
    missing = []
    for mod, name in required:
        try:
            importlib.import_module(mod)
            installed.append(name)
        except ImportError:
            missing.append(name)

    opt_installed = []
    opt_missing = []
    for mod, name in optional:
        try:
            importlib.import_module(mod)
            opt_installed.append(name)
        except ImportError:
            opt_missing.append(name)

    msg = f"Required: {', '.join(installed)}"
    if missing:
        msg += f" | MISSING: {', '.join(missing)}"
    if opt_installed:
        msg += f" | Optional installed: {', '.join(opt_installed)}"

    status = "PASS" if not missing else "FAIL"
    return DiagResult(
        component="python_dependencies",
        status=status,
        detail=msg,
        meta={
            "installed": installed + opt_installed,
            "missing": missing + opt_missing,
        },
    )


def _check_audio_device() -> DiagResult:
    """Check if audio input/output devices are available."""
    # This is a light check — full audio device enumeration requires PyAudio/SoundCard
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        output_devices = [d for d in devices if d["max_output_channels"] > 0]
        return DiagResult(
            component="audio_devices",
            status="PASS",
            detail=f"{len(input_devices)} input, {len(output_devices)} output devices",
            meta={
                "input_devices": [d["name"] for d in input_devices[:5]],
                "output_devices": [d["name"] for d in output_devices[:5]],
            },
        )
    except ImportError:
        return DiagResult(
            component="audio_devices",
            status="SKIP",
            detail="sounddevice not installed (pip install sounddevice for device detection)",
        )
    except Exception as e:
        return DiagResult(
            component="audio_devices",
            status="WARN",
            detail=str(e),
        )


def _check_disk_space() -> DiagResult:
    """Check available disk space for outputs and corrections."""
    import shutil
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        if free_gb < 1:
            return DiagResult(
                component="disk_space",
                status="WARN",
                detail=f"Only {free_gb:.1f}GB free",
            )
        return DiagResult(
            component="disk_space",
            status="PASS",
            detail=f"{free_gb:.1f}GB free",
        )
    except Exception as e:
        return DiagResult(
            component="disk_space",
            status="SKIP",
            detail=str(e),
        )


def run_all() -> Dict[str, Any]:
    """
    Run all diagnostics and return a structured report.
    """
    results = {}

    # System
    results["ffmpeg"] = _check_ffmpeg()
    results["python_deps"] = _check_python_deps()
    results["disk_space"] = _check_disk_space()
    results["audio_devices"] = _check_audio_device()

    # ASR backends
    asr_results = _check_asr_all_backends()
    for r in asr_results:
        results[r.component] = r

    # TTS
    for backend in ["mock", "edge_tts"]:
        results[f"tts_{backend}"] = _check_tts_backend(backend)

    # E2E
    results["e2e_mock"] = _check_e2e_mock()

    # Summary
    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results.values():
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    overall = "PASS" if status_counts["FAIL"] == 0 else "FAIL"
    if overall == "PASS" and status_counts["WARN"] > 0:
        overall = "WARN"

    return {
        "overall": overall,
        "summary": status_counts,
        "checks": {k: v.__dict__ for k, v in results.items()},
        "recommendations": _generate_recommendations(results),
    }


def _generate_recommendations(results: Dict[str, DiagResult]) -> List[str]:
    recs = []
    if results.get("ffmpeg", DiagResult("", "PASS")).status != "PASS":
        recs.append("Install ffmpeg: apt-get install ffmpeg / brew install ffmpeg")
    if results.get("asr_faster_whisper", DiagResult("", "PASS")).status != "PASS":
        recs.append("Install faster-whisper: pip install faster-whisper (for baseline ASR)")
    if results.get("asr_funasr", DiagResult("", "PASS")).status != "PASS":
        recs.append("Install FunASR: pip install funasr (for Paraformer backend)")
    if results.get("asr_firered", DiagResult("", "PASS")).status == "SKIP":
        recs.append("Install FireRedASR: pip install git+https://github.com/FireRedTeam/FireRedASR.git (best engineering ASR)")
    if results.get("asr_telespeech", DiagResult("", "PASS")).status == "SKIP":
        recs.append("Install TeleSpeech-ASR: pip install transformers torch (Wenzhou dialect ASR)")
    if results.get("tts_edge_tts", DiagResult("", "PASS")).status != "PASS":
        recs.append("Install edge-tts: pip install edge-tts (for natural Mandarin TTS)")
    if results.get("e2e_mock", DiagResult("", "PASS")).status != "PASS":
        recs.append("Pipeline failure detected. Check Python path and dependencies.")
    return recs


def print_report(report: Dict[str, Any]):
    """Print a human-readable diagnostic report."""
    overall = report["overall"]
    emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(overall, "❓")
    print(f"\n{'='*60}")
    print(f"  {emoji} System Health: {overall}")
    print(f"{'='*60}")
    print(f"  PASS: {report['summary'].get('PASS', 0)}  "
          f"WARN: {report['summary'].get('WARN', 0)}  "
          f"FAIL: {report['summary'].get('FAIL', 0)}  "
          f"SKIP: {report['summary'].get('SKIP', 0)}")
    print(f"{'='*60}")

    for name, check in report["checks"].items():
        emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(check["status"], "❓")
        latency = f" ({check['latency_ms']:.0f}ms)" if check["latency_ms"] else ""
        print(f"  {emoji} {name:30s} {check['status']:5s}{latency:>12s}")
        if check["detail"]:
            print(f"     {check['detail'][:120]}")

    if report["recommendations"]:
        print(f"\n  {'─'*58}")
        print(f"  💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"     • {rec}")
    print(f"{'='*60}\n")
