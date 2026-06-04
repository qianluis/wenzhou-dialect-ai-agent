# 模型对比：温州话→普通话 Agent

| 模型/工具 | 优点 | 缺点 | 项目建议 |
|---|---|---|---|
| FireRedASR2S | 工业级一体化，含 ASR/VAD/LID/Punc | 新模型集成成本较高 | 优先接入 |
| TeleSpeech-ASR | 面向多方言，含温州话方向 | 部署和依赖需要验证 | 优先接入 |
| FunASR/Paraformer | 中文 ASR 工具链成熟，速度快 | 温州话专用能力需实测 | 第二优先 |
| faster-whisper | 安装容易，baseline 快 | 温州话效果可能有限 | MVP baseline |
| Wav2Vec2-XLS-R | 适合低资源微调 | 需要数据和训练资源 | 后续研究 |
| LLM normalizer | 能修正词序和语义表达 | 依赖提示词和 ASR 输入质量 | 必须保留 |
