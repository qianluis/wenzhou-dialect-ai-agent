"""
Wenzhou Dialect Data Collection — Comprehensive Pipeline

Strategy for building 1M+ Wenzhou ↔ Mandarin parallel entries.

Sources:
  1. Web scraping (Bilibili, Douyin, Weibo — Wenzhou dialect content)
  2. Crowdsourced corrections (via the deployed app)
  3. Synthetic parallel data (Mandarin → dialect generation)
  4. Existing audio corpus transcoding
"""

from __future__ import annotations

import json
import csv
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COLLECTION_DIR = DATA_DIR / "collection"
CORRECTIONS_DIR = DATA_DIR / "corrections"
LEXICON_PATH = DATA_DIR / "lexicon" / "wenzhou_mandarin_glossary.csv"

COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DataEntry:
    """Single Wenzhou ↔ Mandarin parallel data entry."""
    audio_id: str = ""
    audio_path: str = ""
    duration_sec: float = 0.0
    source: str = ""              # weibo / bilibili / douyin / user / synthetic
    mandarin_text: str = ""       # corrected Mandarin text
    asr_raw: str = ""             # raw ASR output (optional)
    speaker_id: str = ""
    gender: str = ""
    age_group: str = ""           # child / young / middle / elderly
    scene: str = ""               # hospital / gov / family / business / other
    noise_level: str = ""         # clean / mild / medium / heavy
    dialect_region: str = ""      # 温州鹿城 / 瓯海 / 龙湾 / 瑞安 / 乐清 / 永嘉 / 平阳 / 苍南 / 文成 / 泰顺
    wenzhou_ipa: str = ""         # optional IPA transcription
    confidence: float = 0.0       # confidence score
    verified: bool = False        # human verified
    created_at: str = ""
    tags: list = field(default_factory=list)


@dataclass
class LexiconEntry:
    """Wenzhou ↔ Mandarin word/phrase mapping."""
    wenzhou_chars: str = ""
    wenzhou_pron: str = ""        # pronunciation (e.g. 温州话拼音)
    mandarin: str = ""
    pos: str = ""                 # part of speech
    notes: str = ""
    frequency: int = 0


# =========================================================================
# Strategy 1: Scrape Wenzhou dialect content from web platforms
# =========================================================================

WEB_SOURCES = {
    "bilibili": {
        "description": "Bilibili Wenzhou dialect videos with subtitles",
        "search_queries": [
            "温州话教学",
            "温州话视频",
            "温州搞笑方言",
            "温州话段子",
            "温州话配音",
            "温州话连续剧",
        ],
        "estimated_yield": "5000-20000 entries (with subtitle extraction)",
    },
    "douyin": {
        "description": "Douyin/TikTok Wenzhou dialect short videos",
        "search_queries": [
            "温州话",
            "温州方言",
            "温州话日常",
        ],
        "estimated_yield": "10000-50000 entries",
    },
    "weibo": {
        "description": "Weibo posts with Wenzhou dialect text",
        "search_queries": [
            "温州话",
            "温州方言 日常",
        ],
        "estimated_yield": "5000-10000 entries (text-only)",
    },
    "wenzhou_forum": {
        "description": "703804.com / 温州论坛 dialect discussions",
        "estimated_yield": "2000-5000 entries",
    },
}

# =========================================================================
# Strategy 2: Synthetic data via TTS + parallel corpus
# =========================================================================

SYNTHETIC_SOURCES = {
    "mandarin_tts_to_dialect": {
        "description": "Generate Wenzhou dialect audio from Mandarin text using TTS",
        "note": "This is a research direction — needs a Wenzhou-dialect TTS model",
        "status": "future_work",
    },
    "parallel_text_corpus": {
        "description": "Build parallel text corpus from existing translations",
        "sources": [
            "Classical Chinese texts with Wenzhou annotations",
            "Wenzhou opera scripts (瓯剧)",
            "Wenzhou folk songs lyrics",
            "Wenzhou news broadcasts transcripts",
        ],
        "estimated_yield": "10000-50000 text pairs",
    },
}

# =========================================================================
# Strategy 3: Active collection via the deployed app
# =========================================================================

APP_COLLECTION = {
    "correction_loop": {
        "description": "Users correct ASR output → stored as training data",
        "fields": [
            "audio_id", "audio_path", "asr_raw", "mandarin_corrected",
            "speaker_id", "gender", "age", "scene", "noise_level",
        ],
        "current_count": 0,
        "target_count": 100000,
    },
    "voice_recording_campaign": {
        "description": "Targeted recording sessions with native speakers",
        "scenes": [
            "医院问诊 (Hospital visit)",
            "家庭日常 (Family daily life)",
            "买菜购物 (Market shopping)",
            "政务办事 (Government services)",
            "商务交流 (Business communication)",
            "出行问路 (Travel directions)",
            "学校场景 (School settings)",
            "情绪表达 (Emotional expressions)",
        ],
        "speakers_needed": 200,
        "utterances_per_speaker": 500,
        "total_estimate": 100000,
    },
}

# =========================================================================
# Target: 1,000,000 entries
# =========================================================================

PHASED_TARGET = {
    "phase_1_week_1_2": {
        "target": 10000,
        "method": "Scrape Bilibili + Douyin Wenzhou videos with subtitles",
        "status": "in_progress",
    },
    "phase_2_month_1": {
        "target": 100000,
        "method": "Crowdsource corrections + active recording 50 speakers × 2000 utterances",
        "status": "planned",
    },
    "phase_3_month_2_3": {
        "target": 300000,
        "method": "Scale recording to 200 speakers + scrape expanded sources",
        "status": "planned",
    },
    "phase_4_month_4_6": {
        "target": 600000,
        "method": "Data augmentation + semi-supervised labeling + synthetic generation",
        "status": "planned",
    },
    "phase_5_month_6_12": {
        "target": 1000000,
        "method": "Combine all sources, dedup, quality filter, release v1.0",
        "status": "planned",
    },
}


def generate_audio_id(source: str = "collected") -> str:
    """Generate a unique audio ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(f"{timestamp}_{os.urandom(4).hex()}".encode()).hexdigest()[:8]
    return f"wz_{source}_{timestamp}_{hash_suffix}"


def save_entry(entry: DataEntry, output_dir: Path = None) -> str:
    """Save a single data entry to CSV."""
    if output_dir is None:
        output_dir = COLLECTION_DIR

    csv_path = output_dir / "wenzhou_dataset.csv"
    is_new = not csv_path.exists()

    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(DataEntry()).keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(entry))

    return entry.audio_id


def stats() -> dict:
    """Get current collection statistics."""
    csv_path = COLLECTION_DIR / "wenzhou_dataset.csv"
    if not csv_path.exists():
        return {"total": 0, "by_source": {}, "by_scene": {}}

    total = 0
    by_source = {}
    by_scene = {}
    by_gender = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            src = row.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
            scene = row.get("scene", "unknown")
            by_scene[scene] = by_scene.get(scene, 0) + 1
            gender = row.get("gender", "unknown")
            by_gender[gender] = by_gender.get(gender, 0) + 1

    return {
        "total": total,
        "by_source": by_source,
        "by_scene": by_scene,
        "by_gender": by_gender,
    }


def print_report():
    """Print a human-readable report."""
    s = stats()
    print(f"\n{'='*50}")
    print(f"📊 温州话数据集现状")
    print(f"{'='*50}")
    print(f"  总计条目: {s['total']:,}")
    print(f"\n  按来源:")
    for src, count in sorted(s.get("by_source", {}).items(), key=lambda x: -x[1]):
        print(f"    {src:20s} {count:>8,}")
    print(f"\n  按场景:")
    for scene, count in sorted(s.get("by_scene", {}).items(), key=lambda x: -x[1]):
        print(f"    {scene:20s} {count:>8,}")
    print(f"\n  目标: 1,000,000 条")
    print(f"  进度: {s['total'] / 1000000 * 100:.2f}%" if s['total'] > 0 else "  进度: 刚刚开始")
    print(f"{'='*50}\n")

    print("=== 收集路线图 ===")
    for phase, info in PHASED_TARGET.items():
        emoji = "🔴" if info["status"] == "planned" else "🟡" if info["status"] == "in_progress" else "✅"
        print(f"  {emoji} {phase:20s} → {info['target']:>8,} 条 | {info['method'][:50]}")
    print()


if __name__ == "__main__":
    print_report()
