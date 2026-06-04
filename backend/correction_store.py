from __future__ import annotations

from pathlib import Path
import csv
import uuid
from datetime import datetime
from typing import Optional


FIELDS = [
    "id", "created_at", "audio_id", "audio_path", "asr_raw", "mandarin_corrected",
    "speaker_id", "gender", "age", "scene", "noise_level", "note"
]


class CorrectionStore:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()

    def append(
        self,
        audio_path: str,
        asr_raw: str,
        mandarin_corrected: str,
        speaker_id: str = "",
        gender: str = "",
        age: str = "",
        scene: str = "",
        noise_level: str = "",
        note: str = "",
    ) -> str:
        row_id = str(uuid.uuid4())
        row = {
            "id": row_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_id": Path(audio_path).stem,
            "audio_path": audio_path,
            "asr_raw": asr_raw,
            "mandarin_corrected": mandarin_corrected,
            "speaker_id": speaker_id,
            "gender": gender,
            "age": age,
            "scene": scene,
            "noise_level": noise_level,
            "note": note,
        }
        with self.csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerow(row)
        return row_id
