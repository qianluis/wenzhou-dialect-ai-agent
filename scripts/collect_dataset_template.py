from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "audio_id", "audio_path", "asr_raw", "mandarin_corrected",
    "speaker_id", "gender", "age", "scene", "noise_level", "note"
]


def create_template(output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
    print(f"Template saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/corrections/template.csv")
    args = parser.parse_args()
    create_template(args.output)


if __name__ == "__main__":
    main()
