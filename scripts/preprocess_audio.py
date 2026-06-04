from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def preprocess_audio(input_path: str, output_path: str, sample_rate: int = 16000):
    """
    Convert audio to 16 kHz mono wav.

    If ffmpeg is not available and input is already a WAV file,
    copies it directly as a fallback.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    ext = Path(input_path).suffix.lower()

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-vn",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available
        if ext == ".wav":
            # Copy as-is
            if Path(output_path) != Path(input_path):
                shutil.copy2(input_path, output_path)
            return output_path
        else:
            raise RuntimeError(
                f"ffmpeg is required to convert {ext} audio to 16kHz mono WAV. "
                "Install ffmpeg or use .wav files directly."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input audio path")
    parser.add_argument("output", help="output wav path")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    preprocess_audio(args.input, args.output, args.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
