"""Wenzhou2Mandarin Agent test suite."""

from scripts.evaluate_cer import cer


def test_cer_zero():
    assert cer("你好", "你好") == 0.0


def test_cer_nonzero():
    assert cer("你好", "您好") > 0.0


def test_cer_empty():
    assert cer("", "") == 0.0


def test_cer_fully_wrong():
    assert cer("你好世界", "再见朋友") > 0.0


def test_cer_partial():
    c = cer("今天天气不错", "今天天气很好")
    assert c > 0.0
    assert c < 1.0


def test_pipeline_import():
    import os
    os.environ["ASR_BACKEND"] = "mock"
    os.environ["TTS_BACKEND"] = "mock"
    os.environ["LLM_BACKEND"] = "rules"

    from backend.asr_engine import ASREngine
    from backend.pipeline import Wenzhou2MandarinPipeline
    from backend.llm_normalizer import MandarinNormalizer
    from backend.correction_store import CorrectionStore
    from backend.config import PROJECT_ROOT, OUTPUT_DIR

    assert PROJECT_ROOT.exists()
    assert OUTPUT_DIR.exists()


def test_diagnostics_import():
    from backend.diagnostics import run_all, print_report, DiagResult
    report = run_all()
    assert "overall" in report
    assert "checks" in report
    assert "summary" in report


def test_enhanced_pipeline_import():
    import os
    os.environ["ASR_BACKEND"] = "mock"
    os.environ["TTS_BACKEND"] = "mock"
    from backend.enhanced_pipeline import EnhancedPipeline, get_pipeline
    p = get_pipeline()
    assert p is not None


def test_streaming_engine_import():
    from backend.streaming_asr import StreamingASREngine, StreamSegment, StreamResult
    engine = StreamingASREngine(backend="mock")
    assert engine is not None


def test_correction_store_rw():
    import tempfile
    import csv
    import os
    from backend.correction_store import CorrectionStore

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(tmp_fd)
    os.unlink(tmp_path)

    store = CorrectionStore(tmp_path)
    row_id = store.append(
        audio_path="/tmp/test.wav",
        asr_raw="测试文本",
        mandarin_corrected="修正文本",
        speaker_id="spk01",
        gender="male",
        age="30",
        scene="test",
        noise_level="clean",
        note="unit test",
    )
    assert row_id

    with open(tmp_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["mandarin_corrected"] == "修正文本"

    os.unlink(tmp_path)
