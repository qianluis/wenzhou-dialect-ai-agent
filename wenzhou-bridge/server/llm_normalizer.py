from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Dict, Optional


SYSTEM_PROMPT = """
你是温州话到普通话的转写助手。请在不改变原意的前提下，将识别文本改写为自然、准确、简洁的普通话。
要求：
1. 不要扩写事实；
2. 不要添加识别文本中没有的信息；
3. 修复明显的口语残缺、重复、错别字和标点；
4. 保留人名、地名、机构名；
5. 输出最终普通话文本，不要解释。
""".strip()


class MandarinNormalizer:
    """
    Rule-based normalizer with optional LLM-ready prompt design.

    The first MVP uses deterministic rules for reliability.
    Later, replace normalize_with_llm() with an OpenAI-compatible client or local LLM.
    """

    def __init__(self, lexicon_path: Optional[str | Path] = None, backend: str = "rules"):
        self.backend = backend
        self.lexicon = self._load_lexicon(lexicon_path) if lexicon_path else {}

    def normalize(self, asr_text: str, context: str = "") -> str:
        text = asr_text.strip()
        text = self._remove_backend_noise(text)
        text = self._apply_lexicon(text)
        text = self._cleanup_repetition(text)
        text = self._normalize_punctuation(text)

        if self.backend == "openai_compatible":
            # Placeholder: the project stays offline-safe by default.
            # Implement API call here if needed.
            return self.build_llm_prompt(text, context=context)

        return text

    def build_llm_prompt(self, asr_text: str, context: str = "") -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"场景信息：{context or '无'}\n"
            f"ASR初始识别文本：{asr_text}\n"
            f"请输出普通话文本："
        )

    def _load_lexicon(self, lexicon_path: str | Path) -> Dict[str, str]:
        path = Path(lexicon_path)
        if not path.exists():
            return {}
        mapping = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = (row.get("source") or "").strip()
                tgt = (row.get("mandarin") or "").strip()
                if src and tgt:
                    mapping[src] = tgt
        return mapping

    def _apply_lexicon(self, text: str) -> str:
        for src, tgt in sorted(self.lexicon.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(src, tgt)
        return text

    def _remove_backend_noise(self, text: str) -> str:
        patterns = [
            r"\[.*?placeholder\]",
            r"mock\s*",
            r"嗯+",
            r"啊+",
            r"呃+",
            r"那个那个",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()

    def _cleanup_repetition(self, text: str) -> str:
        # Very conservative repetition cleanup.
        text = re.sub(r"(，\s*){2,}", "，", text)
        text = re.sub(r"(\。\s*){2,}", "。", text)
        text = re.sub(r"(.{2,6})\1{2,}", r"\1", text)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        text = text.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
        text = re.sub(r"\s+", "", text)
        if text and text[-1] not in "。！？；：":
            text += "。"
        return text
