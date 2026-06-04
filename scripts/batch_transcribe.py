from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import Wenzhou2MandarinPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="outputs/batch_results.csv")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipeline = Wenzhou2MandarinPipeline()

    files = []
    for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
        files.extend(audio_dir.glob(ext))

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "asr_raw", "mandarin_text", "tts_audio_path", "backend"])
        writer.writeheader()
        for audio in files:
            result = pipeline.run(audio)
            writer.writerow({k: result.get(k, "") for k in ["audio_path", "asr_raw", "mandarin_text", "tts_audio_path", "backend"]})
            print(f"Processed: {audio}")

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
