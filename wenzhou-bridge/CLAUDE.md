# 温州话桥 (wenzhou-bridge) — 第一阶段构建说明

> 给 Claude Code 的指令文件。目标:在 ECS 上跑通 **A 方向**——
> ① 温州话/普通话录音 → 准实时转普通话文字 ② 打字或语音的普通话 → 普通话语音。
> 全部自托管小模型,不依赖外部 API。温州话识别先用 Qwen3-ASR 实测,不够再去 M9 微调(第二阶段,本仓库暂不涉及)。

## 架构(句级准实时)

```
浏览器(麦克风 16k PCM, AudioWorklet)
        │  WebSocket 推流
        ▼
FastAPI 编排层 (server/main.py)
   ├─ VAD 断句(webrtcvad)→ 整句音频
   ├─ ASR:本地 Qwen3-ASR 服务(:8001, OpenAI 兼容 /v1/audio/transcriptions)
   ├─ 规范化(可选):本地 Qwen3-1.7B(:8002, vLLM OpenAI /v1/chat/completions)
   └─ TTS:本地 CosyVoice2(:8003)→ 流式音频回浏览器
```

**为什么句级而不是逐字流式**:温州话识别准确率压倒延迟。VAD 切出整句再识别,既能拿到 Qwen3-ASR 的最佳方言效果,单句延迟又能压在 1 秒内。逐字流式(FunASR paraformer-online)作为第二阶段的"预览轨"可选加,不在第一阶段。

## 服务分工(重要)

- **本机/ECS 只做推理**,常驻服务。
- **模型微调一律在 M9 跑**,不要在这台 ECS 上训练。
- 严格沿用既有习惯:启动脚本(run)/ 监控 / 修复分离,destructive 操作显式确认。

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| ASR (Qwen3-ASR) | 8001 | 官方 docker,OpenAI 兼容音频转写 |
| 规范化 LLM (Qwen3-1.7B) | 8002 | vLLM,可选,默认开 |
| TTS (CosyVoice2-0.5B) | 8003 | CosyVoice 自带 server |
| 编排 + 前端 (FastAPI) | 8000 | 用户访问入口 |

## 构建任务清单(Claude Code 按序完成)

1. **环境**:Python 3.10+,CUDA 12.x。`pip install -r requirements.txt`。
2. **下模型**:运行 `scripts/setup_models.sh`(走 ModelScope)。脚本里的 repo id 标了 `# 待确认`,先 `modelscope` 搜一下确认准确名再跑。
3. **起底层服务**:`scripts/run_services.sh` 拉起 8001/8002/8003。先逐个 curl 健康检查,全绿再继续。
4. **编排层**:`server/main.py` 已给出可运行骨架。补全标 `TODO` 的地方:
   - ASR 服务返回字段的解析(以实际 docker 输出为准)。
   - CosyVoice 请求体字段(以实际 server 接口为准)。
5. **前端**:`web/index.html` 已给骨架。重点调 **48k→16k 重采样**(标了 TODO,先用线性插值跑通,再换成正经 resampler)。
6. **联调验收**(见下)。

## 第一阶段验收标准

- [ ] 对着麦克风说普通话,1.5 秒内屏幕出正确文字。
- [ ] 说一段温州话,出文字(准确率先记录基线,**别期待很高**;低于可用线就是第二阶段微调的触发条件)。
- [ ] 文本框打普通话 → 点合成 → 2 秒内听到自然普通话语音。
- [ ] 整条链路单句端到端延迟 < 2 秒。

## 决策点 / 已知风险

- **温州话识别**:Qwen3-ASR 官方点名上海话等吴语,**未单列温州话**。第一阶段把它当吴语泛化用,跑出来记录真实 WER。不达标 → 第二阶段在 M9 上 LoRA 微调 Qwen3-ASR,或微调 TeleSpeech 0.3B(预训练含温州话),需 ≥50 小时标注语料。
- **温州话 TTS**:第一阶段**只做普通话合成**。真温州话语音是独立的 M9 训练项目(DiaMoE-TTS / GPT-SoVITS 克隆),不在此仓库。
- **显存**:ASR 1.7B(≥6G)+ Qwen3-1.7B + CosyVoice 0.5B,整套约 8–16G。单张 24G 卡富余;紧张就把 ASR 换 0.6B、规范化 LLM 关掉。

## 多 provider 兜底(沿用你现有习惯)

`server/main.py` 里 ASR/TTS 的地址都走 `config`。本地服务挂了时,可在 config 里切到阿里云百炼 API 作为兜底,保持和你现有多 provider 切换一致的范式。第一阶段默认全本地。
