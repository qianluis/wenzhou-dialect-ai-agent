# Huawei Claw 后续执行任务书

请基于本项目继续提升，不要推倒重来。

## 第一阶段：工程完善

1. 检查所有代码能否运行：
   - python frontend/gradio_app.py
   - uvicorn backend.app:app --host 0.0.0.0 --port 8000
2. 修复依赖冲突。
3. 增加真实音频预处理接入：上传后自动转 16kHz mono wav。
4. 在 README 中补充截图和运行示例。
5. 生成项目结构图和流程图。

## 第二阶段：真实 ASR 后端

请优先尝试接入以下模型：

1. FireRedASR2S
2. TeleSpeech-ASR
3. FunASR / Paraformer
4. faster-whisper

要求：
- 每个后端必须保持统一接口 ASREngine.transcribe(audio_path)
- 不允许把模型调用写死在 Gradio 前端
- 每个后端写单独的 adapter
- 每个后端记录安装方式、模型下载方式、显存需求和推理速度

## 第三阶段：调研报告升级

扩展 docs/research_report.md：
- 增加论文引用
- 增加模型版本
- 增加数据集链接
- 增加真实部署路线
- 增加商业化场景分析
- 增加风险与合规说明

## 第四阶段：产品化

1. 增加用户纠错页面
2. 增加批量转写
3. 增加 CER dashboard
4. 增加 Hugging Face Space 部署
5. 增加 GitHub Actions 基础测试

## 交付标准

- 所有功能 mock 模式必须可运行
- README 足够清晰
- 不要删除人工纠错数据模块
- 保持低资源方言研究方向
