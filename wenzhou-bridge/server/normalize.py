"""
规范化:ASR 原始文本(可能含温州话词/口语残缺) → 自然标准普通话。

第一阶段默认 rules 后端(零依赖、确定性、离线安全),替代 CLAUDE.md 里的 vLLM Qwen3-1.7B:
  1) WenzhouTranslator.wenzhou_to_mandarin  —— 内置 ~500 词的方言→普通话映射
  2) CSV 词典(data/lexicon, source→mandarin)—— 可持续补充的精选词条
  3) MandarinNormalizer 清洗 —— 去口头语噪声 / 去重复 / 规范标点

可选 openai 后端:走用户现有网关(MiMo/百炼等),做一遍 LLM 重写。默认不启用。
"""
from __future__ import annotations

from . import config
from .wenzhou_translator import WenzhouTranslator
from .llm_normalizer import MandarinNormalizer

_translator = WenzhouTranslator()
_normalizer = MandarinNormalizer(lexicon_path=config.LEXICON_PATH, backend="rules")


def _rules(text: str) -> str:
    # 仅在检测到温州话标记时才做方言→普通话替换。
    # 否则(已经是普通话的 ASR 输出)直接放行,避免 好→可以、饭→米饭 这类误替换。
    if _translator.detect_language(text) == "wenzhou_to_mandarin":
        text = _translator.wenzhou_to_mandarin(text)
    return _normalizer.normalize(text)          # CSV 精选词条 + 清洗 + 标点(轻量、安全)


def _openai(text: str) -> str:
    """可选:走 OpenAI 兼容网关重写。失败则回退 rules,保证链路不断。"""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=config.OPENAI_BASE_URL or None,
            api_key=config.OPENAI_API_KEY or "EMPTY",
        )
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "把可能含方言词或口语残缺的中文改写成自然标准的普通话,只输出结果,不要解释。"},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or _rules(text)
    except Exception:  # noqa: BLE001
        return _rules(text)


def normalize(text: str) -> str:
    text = (text or "").strip()
    if not text or not config.USE_NORM:
        return text
    if config.NORM_BACKEND == "openai":
        return _openai(text)
    return _rules(text)


def lexicon_count() -> int:
    """词典规模(内置映射 + CSV 额外词条),供 /health 展示。"""
    from .wenzhou_translator import _WENZHOU_TO_MANDARIN
    return len(_WENZHOU_TO_MANDARIN) + len(getattr(_normalizer, "lexicon", {}) or {})
