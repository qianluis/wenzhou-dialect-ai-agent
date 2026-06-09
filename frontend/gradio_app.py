"""
Wenzhou2Mandarin Agent v4 — Gradio Frontend (Mobile Optimized, Memory-Safe)
"""
from __future__ import annotations
import sys, json, os, time, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    raise RuntimeError("gradio not installed. Run: pip install gradio")

GRADIO_THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="emerald")

# ── Lazy pipeline (won't load models until first use) ──
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from backend.bidirectional_pipeline import BidirectionalPipeline
        _pipeline = BidirectionalPipeline()
    return _pipeline

LAST_RESULT = {}


# ── Core processing ──
def process_audio(audio_path, direction, context, tts_enabled):
    global LAST_RESULT
    t_start = time.time()
    if audio_path is None:
        return "请先上传或录制音频。", "", None, ""
    try:
        p = get_pipeline()
        result = p.process(audio_path, direction=direction, context=context or "",
                           tts_enabled=tts_enabled)
    except Exception as e:
        return f"处理出错: {e}", "", None, f"❌ {e}"
    LAST_RESULT = result
    timings = result.get("stage_timings_ms", {})
    dir_labels = {"wenzhou_to_mandarin": "温州话>普通话",
                  "mandarin_to_wenzhou": "普通话>温州话",
                  "wenzhou_to_wenzhou": "温州话>温州话文字"}
    dl = dir_labels.get(direction, direction)
    status = (f"[{dl}] ASR: {timings.get('asr_ms',0):.0f}ms | "
              f"翻译: {timings.get('translate_ms',0):.0f}ms | "
              f"TTS: {timings.get('tts_ms',0):.0f}ms | "
              f"总计: {(time.time()-t_start)*1000:.0f}ms | "
              f"后端: {result.get('backend','?')}")
    return result.get("asr_raw",""), result.get("target_text",""), result.get("tts_audio_path",None), status


def process_ensemble(audio_path, context, tts_enabled):
    global LAST_RESULT
    t_start = time.time()
    if audio_path is None:
        return "请先上传或录制音频。", "", None, ""
    try:
        p = get_pipeline()
        result = p.wenzhou_ensemble(audio_path, context=context or "", tts_enabled=tts_enabled)
    except Exception as e:
        return f"处理出错: {e}", "", None, f"❌ {e}"
    LAST_RESULT = result
    timings = result.get("stage_timings_ms", {})
    ev = result.get('ensemble_votes','?')
    et = result.get('ensemble_total','?')
    status = (f"[融合] ASR: {timings.get('asr_ms',0):.0f}ms | "
              f"翻译: {timings.get('translate_ms',0):.0f}ms | "
              f"TTS: {timings.get('tts_ms',0):.0f}ms | "
              f"总计: {(time.time()-t_start)*1000:.0f}ms | "
              f"投票: {ev}/{et}")
    return result.get("asr_raw",""), result.get("target_text",""), result.get("tts_audio_path",None), status


def run_diag():
    import psutil
    lines = ["## 🩺 系统状态\n"]
    mem = psutil.virtual_memory()
    ma = mem.available / 1024**3
    mt = mem.total / 1024**3
    emoji = "✅" if ma > 1.0 else "⚠️"
    lines.append(f"{emoji} **内存**: 可用 {ma:.1f}GB / 总计 {mt:.1f}GB ({mem.percent}%)\n")
    du = psutil.disk_usage("/")
    df = du.free / 1024**3
    emoji2 = "✅" if df > 2.0 else "⚠️"
    lines.append(f"{emoji2} **磁盘**: 剩余 {df:.1f}GB\n")
    for m in ["funasr","torch","edge_tts","gradio"]:
        try:
            __import__(m)
            lines.append(f"- ✅ **{m}**: 已安装\n")
        except ImportError:
            lines.append(f"- ❌ **{m}**: 未安装\n")
    if shutil.which("ffmpeg"):
        lines.append("- ✅ **ffmpeg**: 可用\n")
    return "".join(lines)


# ======================================================================
# Build UI — Mobile-first
# ======================================================================
with gr.Blocks(title="温州话 ↔ 普通话 Agent v4") as demo:

    gr.Markdown("""
# 🗣️ 温州话 ↔ 普通话 双向翻译

> 说温州话→出普通话文字，说普通话→出温州话文字，手机端录制即翻
""")

    with gr.Tabs():
        # ====== TAB 1: 翻译 ======
        with gr.Tab("🎤 翻译"):
            direction = gr.Radio(
                choices=[
                    ("温州话语音 → 普通话文字 (方言识别)", "wenzhou_to_mandarin"),
                    ("普通话语音 → 温州话文字 (翻译)", "mandarin_to_wenzhou"),
                    ("温州话语音 → 温州话文字 (转写)", "wenzhou_to_wenzhou"),
                ],
                value="wenzhou_to_mandarin",
                label="翻译方向",
                container=False,
            )

            audio_input = gr.Audio(
                label="录制或上传语音",
                sources=["upload", "microphone"],
                type="filepath",
            )

            context = gr.Textbox(
                label="场景提示（可选）",
                placeholder="例如：医院挂号、菜市场、家庭聊天",
                lines=1,
            )

            with gr.Row():
                tts_enabled = gr.Checkbox(label="🔊 语音播报", value=True)
                use_ensemble = gr.Checkbox(label="🎯 多模型融合(更准)", value=False)

            run_btn = gr.Button("🚀 开始翻译", variant="primary", size="lg")

            with gr.Accordion("📝 结果", open=True):
                asr_raw = gr.Textbox(label="原始识别", lines=2, interactive=False)
                target_text = gr.Textbox(label="翻译文本", lines=3, interactive=False)
                tts_audio = gr.Audio(label="语音播报", type="filepath")
                status_bar = gr.Textbox(label="运行状态", lines=1)

        # ====== TAB 2: 系统状态 ======
        with gr.Tab("🔧 系统状态"):
            diag_btn = gr.Button("🔄 刷新系统状态", variant="secondary", size="sm")
            diag_output = gr.Markdown("点击刷新查看组件健康状态。")
            gr.Markdown("""
### 可用后端
| 后端 | 大小 | 适用 |
|------|------|------|
| **SenseVoiceSmall** | 893MB | 多方言 |
| **Whisper small** | 461MB | 通用 |
| **FireRedASR2-AED** | 2.2GB | 吴语专用 |
| **mock** | 0 | 调试 |
""")

        # ====== TAB 3: 上次结果 ======
        with gr.Tab("📄 上次结果"):
            try:
                with open("/tmp/wenzhou_transcription.txt") as f: raw_t = f.read()
                with open("/tmp/wenzhou_mandarin_translated.txt") as f: tr_t = f.read()
            except FileNotFoundError:
                raw_t = tr_t = "请先运行转写"
            gr.Markdown("### ✅ 已补完：《白晓教你学温州话》转写+翻译")
            gr.Textbox(value=raw_t[:1500], label="原始听写", lines=5, interactive=False)
            gr.Textbox(value=tr_t[:1500], label="翻译结果", lines=5, interactive=False)

    # ── Events ──
    def handle_translate(audio, dir_val, ctx, tts_on, ensemble):
        if ensemble and dir_val == "wenzhou_to_mandarin":
            return process_ensemble(audio, ctx, tts_on)
        return process_audio(audio, dir_val, ctx, tts_on)

    run_btn.click(fn=handle_translate,
                  inputs=[audio_input, direction, context, tts_enabled, use_ensemble],
                  outputs=[asr_raw, target_text, tts_audio, status_bar])
    diag_btn.click(fn=run_diag, inputs=[], outputs=[diag_output])


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    print(f"\n{'='*50}")
    print(f"  温州话 ↔ 普通话 双向翻译 Agent v4")
    print(f"  {'='*50}")
    print(f"  Local:  http://localhost:{port}")
    print(f"  {'='*50}\n")
    demo.queue(max_size=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        theme=GRADIO_THEME,
    )
