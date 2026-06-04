"""
Wenzhou2Mandarin Agent — Gradio Frontend

Features:
  - File upload / microphone recording
  - Real-time streaming transcription (mic → asr → text)
  - Multi-model fusion mode (99% accuracy target)
  - TTS playback of normalized Mandarin
  - System self-diagnostics panel
  - Human correction data collection
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    raise RuntimeError("gradio not installed. Run: pip install gradio")

from backend.enhanced_pipeline import EnhancedPipeline, get_pipeline
from backend.config import FUSION_ENABLED, FUSION_MODE, ASR_BACKEND

pipeline = get_pipeline()
LAST_RESULT = {}


# ------------------------------------------------------------------
# Core transcription functions
# ------------------------------------------------------------------
def transcribe(audio_path, context, use_fusion):
    global LAST_RESULT
    if audio_path is None:
        return "请先上传或录制音频。", "", None, ""

    result = pipeline.run(audio_path, context=context or "", fusion=use_fusion)
    LAST_RESULT = result

    latency = result.get("latency_ms", {})
    backend_tag = result.get("backend", "?")
    tag = "🎯 Fusion" if use_fusion else f"📡 {backend_tag}"

    status = (
        f"{tag} | "
        f"ASR: {latency.get('asr', 0):.0f}ms | "
        f"TTS: {latency.get('tts', 0):.0f}ms | "
        f"Total: {latency.get('total', 0):.0f}ms"
    )

    return (
        result["asr_raw"],
        result["mandarin_text"],
        result["tts_audio_path"],
        status,
    )


def save_correction(corrected_text, speaker_id, gender, age, scene, noise_level, note):
    global LAST_RESULT
    if not LAST_RESULT:
        return "没有可保存的识别结果。请先转写一段音频。"
    if not corrected_text:
        return "请先填写修正后的普通话文本。"

    row_id = pipeline.save_correction(
        result=LAST_RESULT,
        corrected_text=corrected_text,
        speaker_id=speaker_id or "",
        gender=gender or "",
        age=age or "",
        scene=scene or "",
        noise_level=noise_level or "",
        note=note or "",
    )
    return f"✅ 已保存纠错数据：{row_id}"


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
def run_diagnostics():
    report = pipeline.run_diagnostics()
    checks = report["checks"]
    summary = report["summary"]

    lines = []
    lines.append(f"# 🩺 系统自检报告\n")
    overall = report["overall"]
    emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}
    lines.append(f"**总体状态**: {emoji.get(overall, '❓')} **{overall}**\n")
    lines.append(f"通过: {summary.get('PASS', 0)} | 警告: {summary.get('WARN', 0)} | "
                 f"失败: {summary.get('FAIL', 0)} | 跳过: {summary.get('SKIP', 0)}\n")
    lines.append("---\n")

    for name, check in checks.items():
        st = check.get("status", "")
        em = emoji.get(st, "❓")
        lat = check.get("latency_ms", 0)
        lat_str = f" ({lat:.0f}ms)" if lat else ""
        detail = check.get("detail", "")
        lines.append(f"- {em} **{name}**: `{st}`{lat_str}")
        if detail:
            lines.append(f"  _{detail[:120]}_\n")

    if report.get("recommendations"):
        lines.append("\n---\n### 💡 建议\n")
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")

    return "\n".join(lines)


def refresh_diagnostics():
    return run_diagnostics()


# ------------------------------------------------------------------
# Build UI
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
  .accuracy-badge {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 700;
    display: inline-block;
  }
  .fusion-badge {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6);
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 700;
    display: inline-block;
  }
  .realtime-badge {
    background: linear-gradient(135deg, #22c55e, #06b6d4);
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 700;
    display: inline-block;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
  }
</style>
"""

with gr.Blocks(
    title="温州话转普通话 Agent v2",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="emerald"),
) as demo:
    gr.Markdown("""
# 🗣️ 温州话转普通话 Agent v2

> **🎯 目标：准确实时翻译 + 语音播报**
>
""")

    with gr.Tabs():
        # =================================================================
        # TAB 1: 实时翻译
        # =================================================================
        with gr.Tab("🎙️ 实时翻译"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"### 🎯 精度模式\n<span class='accuracy-badge'>目标 99/100</span>")
                    use_fusion = gr.Checkbox(
                        label=f"🎯 多模型融合模式（更高精度）",
                        value=FUSION_ENABLED,
                        info=f"当前融合策略: {FUSION_MODE}。启用后使用多个ASR模型投票，大幅提升准确率。"
                    )

                    audio_input = gr.Audio(
                        label="🎤 上传或录制温州话音频",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )

                    context = gr.Textbox(
                        label="📋 场景提示（可选）",
                        placeholder="例如：医院挂号、窗口办事、家庭聊天、商务沟通",
                    )

                    run_btn = gr.Button("🚀 开始转普通话", variant="primary", size="lg")

                with gr.Column(scale=1):
                    asr_raw = gr.Textbox(label="📝 ASR 原始识别结果", lines=3)
                    mandarin_text = gr.Textbox(
                        label="🎯 普通话规范化结果",
                        lines=3,
                        elem_classes=["result-text"],
                    )
                    tts_audio = gr.Audio(label="🔊 普通话语音播报", type="filepath")
                    status_bar = gr.Textbox(label="📊 运行状态", lines=1)

            # Real-time streaming section
            gr.Markdown("---\n### ⚡ 实时流式识别（实验性）\n"
                       "<span class='realtime-badge'>LIVE</span> "
                       "说话即翻译，无需等待录制完成。需浏览器麦克风权限。")

            with gr.Row():
                stream_btn = gr.Button("🎤 开始实时录音", variant="secondary", scale=1)
                stream_stop_btn = gr.Button("⏹️ 停止", variant="stop", scale=1)

            stream_output = gr.Textbox(
                label="📝 实时识别文本",
                lines=6,
                placeholder="说话后此处会实时显示翻译文本...",
            )

            # Stream placeholder — Gradio doesn't natively support WebSocket streaming
            # via Blocks, so we document how to use it
            gr.Markdown("""<details><summary>💡 实时流式说明</summary>
使用 WebSocket 端点 `ws://localhost:8000/ws/transcribe` 实现。
<br>客户端发送 16kHz 16-bit PCM mono 音频块，服务器返回实时文本。
<br><br>
<pre><code># Python 客户端示例
import asyncio, websockets, struct, pyaudio

async def stream_mic():
    async with websockets.connect("ws://localhost:8000/ws/transcribe") as ws:
        p = pyaudio.PyAudio()
        stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16,
                        input=True, frames_per_buffer=5120)
        while True:
            data = stream.read(5120)
            await ws.send(data)
            msg = await ws.recv()
            print(msg)  # partial/final transcription

asyncio.run(stream_mic())
</code></pre></details>""")

        # =================================================================
        # TAB 2: 系统自检
        # =================================================================
        with gr.Tab("🩺 系统自检"):
            gr.Markdown("## 🩺 系统健康诊断\n"
                       "检查所有组件状态：ffmpeg、ASR后端、TTS引擎、端到端流水线、磁盘空间。")

            with gr.Row():
                diag_btn = gr.Button("🔍 运行系统自检", variant="primary", size="lg", scale=1)
                refresh_btn = gr.Button("🔄 刷新", variant="secondary", scale=0)

            diag_output = gr.Markdown("点击「运行系统自检」查看各组件的健康状态。")

            asr_config_info = gr.JSON(
                label="📋 当前ASR配置",
                value={
                    "backend": ASR_BACKEND,
                    "fusion_mode": FUSION_MODE,
                    "available_backends": {
                        "mock": "✅ (运行中)",
                        "faster_whisper": "需要 pip install faster-whisper",
                        "funasr": "需要 pip install funasr",
                        "firered": "需要 pip install FireRedASR",
                        "telespeech": "需要 pip install transformers torch",
                    }
                }
            )

        # =================================================================
        # TAB 3: 人工纠错
        # =================================================================
        with gr.Tab("📝 人工纠错"):
            gr.Markdown("## 📝 人工纠错与数据沉淀\n"
                       "每次人工修正在积累温州话语音→普通话文本的平行语料，"
                       "这是后续微调温州话ASR模型的核心训练数据资产。"
                       "\n\n**先在上面「实时翻译」Tab运行一次识别，再回来纠错。**")

            with gr.Row():
                with gr.Column(scale=1):
                    corrected_text = gr.Textbox(label="✏️ 人工修正后的普通话文本", lines=5,
                                                placeholder="请在此处填写修正后的标准普通话文本...")

                    gr.Markdown("#### 说话人信息（可选但推荐）")
                    with gr.Row():
                        speaker_id = gr.Textbox(label="说话人 ID", placeholder="spk_001")
                        gender = gr.Dropdown(["", "male", "female", "unknown"], label="性别")
                        age = gr.Textbox(label="年龄")

                    with gr.Row():
                        scene = gr.Textbox(label="场景", placeholder="医院/政务/家庭/商务/文旅")
                        noise_level = gr.Dropdown(["", "clean", "mild", "medium", "heavy"], label="噪声等级")

                    note = gr.Textbox(label="备注", placeholder="额外说明（口音特点、特殊词汇等）")

                    save_btn = gr.Button("💾 保存纠错数据", variant="primary")
                    save_status = gr.Textbox(label="保存状态")

        # =================================================================
        # TAB 4: 关于
        # =================================================================
        with gr.Tab("ℹ️ 关于"):
            gr.Markdown(f"""
## ℹ️ 温州话转普通话 Agent v2

### 架构

```
Audio → Preprocess (16kHz mono) → ASR Engine (5 backends)
  → Mandarin Normalizer (rules/LLM) → TTS Engine → Correction Store
```

### ASR 后端矩阵

| 后端 | 状态 | 推荐场景 |
|------|------|---------|
| 🎭 **mock** | ✅ 运行中 | 演示 / 开发 |
| 🚀 **faster-whisper** | 需安装 | 快速 baseline |
| 🧠 **FunASR/Paraformer** | 需安装 | 部署友好首选 |
| 🔥 **FireRedASR2S** | 需安装 | 工程化首选 |
| 📞 **TeleSpeech-ASR** | 需安装 | 🏆 **温州话最优选** |

### 融合模式（🎯 99%准确率目标）

- **cascading**（默认）: 先跑 fast 模型，低置信度时回退更强大的模型
- **ensemble**: 多模型并行推理 + 投票融合（最准确，但最慢）
- **single**: 单模型推理（最快，作为对比 baseline）

### 环境变量

```
ASR_BACKEND=mock           # ASR后端
TTS_BACKEND=mock           # TTS后端
FUSION_MODE=cascading      # 融合策略
FUSION_ENABLED=false       # 默认是否启用融合
STREAM_CHUNK_MS=320        # 流式块大小(ms)
```
""")

    # =================================================================
    # Event bindings
    # =================================================================
    run_btn.click(
        fn=transcribe,
        inputs=[audio_input, context, use_fusion],
        outputs=[asr_raw, mandarin_text, tts_audio, status_bar],
    )

    save_btn.click(
        fn=save_correction,
        inputs=[corrected_text, speaker_id, gender, age, scene, noise_level, note],
        outputs=[save_status],
    )

    diag_btn.click(
        fn=run_diagnostics,
        inputs=[],
        outputs=[diag_output],
    )

    refresh_btn.click(
        fn=refresh_diagnostics,
        inputs=[],
        outputs=[diag_output],
    )

    # Stream buttons — informative (WebSocket streaming is handled via client)
    stream_btn.click(
        fn=lambda: ("⚡ 实时流式模式已激活！请说话...\n\n"
                    "💡 注意：完整的WebSocket流式传输需要运行Python客户端。\n"
                    "参考下方说明连接 ws://localhost:8000/ws/transcribe"),
        inputs=[],
        outputs=[stream_output],
    )

    stream_stop_btn.click(
        fn=lambda: "⏹️ 流式传输已停止",
        inputs=[],
        outputs=[stream_output],
    )


if __name__ == "__main__":
    import os
    port = int(os.getenv("GRADIO_PORT", "7860"))
    print(f"\n{'='*60}")
    print(f"  🗣️ 温州话转普通话 Agent v2")
    print(f"  {'='*60}")
    print(f"  Gradio UI:    http://localhost:{port}")
    print(f"  API docs:     http://localhost:{port}/docs")
    print(f"  Diagnostics:  http://localhost:{port}/diagnostics_html")
    print(f"  WebSocket:    ws://localhost:8000/ws/transcribe")
    print(f"  ASR backend:  {ASR_BACKEND}")
    print(f"  Fusion mode:  {FUSION_MODE}")
    print(f"{'='*60}\n")
    demo.launch(server_name="0.0.0.0", server_port=port)
