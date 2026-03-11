#!/usr/bin/env python3
"""Build an XTTS-ready dataset from a local video/audio file or a YouTube URL.

Outputs a dataset directory like:

    output_dir/
      source/
      wavs/
      metadata_train.csv
      metadata_eval.csv
      speaker_reference.wav

The metadata files use Coqui's `coqui` formatter format:
`audio_file|text|speaker_name`
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def run_command(command: list[str], description: str) -> None:
    print(f"\n[RUN] {description}")
    print(" ".join(command))
    subprocess.run(command, check=True)


def download_source(input_source: str, source_dir: Path) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    if is_url(input_source):
        output_template = str(source_dir / "source.%(ext)s")
        run_command(
            [
                "yt-dlp",
                "-f",
                "bestaudio/best",
                "-o",
                output_template,
                input_source,
            ],
            "Downloading source media with yt-dlp",
        )
        matches = sorted(source_dir.glob("source.*"))
        if not matches:
            raise FileNotFoundError("yt-dlp completed but no source media file was found.")
        return matches[0]

    source_path = Path(input_source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input source not found: {source_path}")

    destination = source_dir / source_path.name
    if source_path != destination:
        shutil.copy2(source_path, destination)
    return destination


def extract_audio(
    input_media: Path,
    output_wav: Path,
    sample_rate: int,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    end_seconds: float | None = None,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.3f}"])

    command.extend(["-i", str(input_media)])

    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    elif end_seconds is not None:
        command.extend(["-to", f"{end_seconds:.3f}"])

    command.extend([
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav),
    ])

    trim_desc = "Extracting mono WAV audio"
    if start_seconds > 0 or duration_seconds is not None or end_seconds is not None:
        trim_desc += " from selected time range"
    run_command(command, trim_desc)


def cut_audio_segment(full_audio: Path, output_audio: Path, start_time: float, end_time: float, sample_rate: int) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-to",
            f"{end_time:.3f}",
            "-i",
            str(full_audio),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_audio),
        ],
        f"Creating segment {output_audio.name}",
    )


def normalize_text(text: str) -> str:
    text = " ".join(text.strip().split())
    return text


def build_segments(
    full_audio: Path,
    output_dir: Path,
    language: str,
    whisper_model_name: str,
    speaker_name: str,
    buffer_seconds: float,
    min_duration: float,
    max_duration: float,
    eval_split: float,
    sample_rate: int,
    compute_type: str,
) -> tuple[Path, Path, Path]:
    from faster_whisper import WhisperModel

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    print(f"\n[INFO] Loading Whisper model '{whisper_model_name}' on {device} ({compute_type})")
    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)

    print("[INFO] Transcribing source audio with word timestamps")
    segments, _ = model.transcribe(str(full_audio), word_timestamps=True, language=language, vad_filter=True)
    segments = list(segments)

    words = []
    for segment in segments:
        words.extend(list(segment.words or []))

    if not words:
        raise RuntimeError("No words were produced by Whisper. Try another source or larger whisper model.")

    metadata_rows: list[dict[str, str]] = []
    sentence = ""
    sentence_start = None
    sentence_index = 0

    for idx, word in enumerate(words):
        token = word.word or ""
        if sentence_start is None:
            if idx == 0:
                sentence_start = max(float(word.start) - buffer_seconds, 0.0)
            else:
                previous_end = float(words[idx - 1].end)
                sentence_start = max(float(word.start) - buffer_seconds, (previous_end + float(word.start)) / 2.0)
            sentence = token
        else:
            sentence += token

        if token.strip().endswith((".", "!", "?")):
            sentence_text = normalize_text(sentence)
            next_word_start = float(words[idx + 1].start) if idx + 1 < len(words) else float(word.end)
            sentence_end = min((float(word.end) + next_word_start) / 2.0, float(word.end) + buffer_seconds)
            duration = sentence_end - sentence_start

            if sentence_text and min_duration <= duration <= max_duration:
                filename = f"segment_{sentence_index:06d}.wav"
                relative_audio_path = Path("wavs") / filename
                output_audio = output_dir / relative_audio_path
                cut_audio_segment(full_audio, output_audio, sentence_start, sentence_end, sample_rate)
                metadata_rows.append(
                    {
                        "audio_file": relative_audio_path.as_posix(),
                        "text": sentence_text,
                        "speaker_name": speaker_name,
                        "duration": f"{duration:.3f}",
                    }
                )
                sentence_index += 1

            sentence = ""
            sentence_start = None

    if len(metadata_rows) < 10:
        raise RuntimeError(f"Only {len(metadata_rows)} usable segments were created. Need more data for XTTS fine-tuning.")

    total_duration = sum(float(row["duration"]) for row in metadata_rows)
    print(f"[INFO] Created {len(metadata_rows)} segments with total duration {total_duration:.1f}s")

    if total_duration < 120:
        raise RuntimeError(
            f"Total usable audio is only {total_duration:.1f}s. XTTS fine-tuning usually needs at least 120s."
        )

    random.shuffle(metadata_rows)
    eval_count = max(1, int(len(metadata_rows) * eval_split))
    eval_rows = metadata_rows[:eval_count]
    train_rows = metadata_rows[eval_count:]

    train_rows = sorted(train_rows, key=lambda row: row["audio_file"])
    eval_rows = sorted(eval_rows, key=lambda row: row["audio_file"])

    train_csv = output_dir / "metadata_train.csv"
    eval_csv = output_dir / "metadata_eval.csv"
    fieldnames = ["audio_file", "text", "speaker_name"]

    for csv_path, rows in ((train_csv, train_rows), (eval_csv, eval_rows)):
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="|")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})

    longest_sample = max(metadata_rows, key=lambda row: float(row["duration"]))
    speaker_reference = output_dir / "speaker_reference.wav"
    shutil.copy2(output_dir / longest_sample["audio_file"], speaker_reference)

    print(f"[INFO] Train CSV: {train_csv}")
    print(f"[INFO] Eval CSV:  {eval_csv}")
    print(f"[INFO] Speaker reference: {speaker_reference}")
    return train_csv, eval_csv, speaker_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an XTTS-ready dataset from a video/audio source.")
    parser.add_argument("input_source", help="Local media path or YouTube URL.")
    parser.add_argument("--output_dir", default="xtts_dataset", help="Output directory for XTTS dataset files.")
    parser.add_argument("--language", default="en", help="Whisper/XTTS language code.")
    parser.add_argument("--speaker_name", default="speaker", help="Speaker name written into XTTS metadata.")
    parser.add_argument("--whisper_model", default="small", help="faster-whisper model name.")
    parser.add_argument("--compute_type", default="float16", help="Whisper compute type: float16, int8, int8_float16, etc.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Dataset sample rate.")
    parser.add_argument("--buffer_seconds", type=float, default=0.20, help="Padding around sentence boundaries.")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum segment duration in seconds.")
    parser.add_argument("--max_duration", type=float, default=11.0, help="Maximum segment duration in seconds.")
    parser.add_argument("--eval_split", type=float, default=0.15, help="Fraction of samples reserved for eval.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for train/eval split.")
    parser.add_argument("--start_minutes", type=float, default=0.0, help="Start processing from this minute offset.")
    parser.add_argument("--duration_minutes", type=float, default=None, help="Only process this many minutes from the start offset.")
    parser.add_argument("--end_minutes", type=float, default=None, help="Stop processing at this minute offset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = output_dir / "source"
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    start_seconds = args.start_minutes * 60.0
    duration_seconds = args.duration_minutes * 60.0 if args.duration_minutes is not None else None
    end_seconds = args.end_minutes * 60.0 if args.end_minutes is not None else None

    if start_seconds < 0:
        raise ValueError("--start_minutes must be >= 0")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("--duration_minutes must be > 0")
    if end_seconds is not None and end_seconds <= 0:
        raise ValueError("--end_minutes must be > 0")
    if duration_seconds is not None and end_seconds is not None:
        raise ValueError("Use either --duration_minutes or --end_minutes, not both.")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("--end_minutes must be greater than --start_minutes")

    source_media = download_source(args.input_source, source_dir)
    full_audio = artifacts_dir / "full_audio.wav"
    extract_audio(
        source_media,
        full_audio,
        sample_rate=args.sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        end_seconds=end_seconds,
    )

    if duration_seconds is not None:
        print(f"[INFO] Processing source window: start={args.start_minutes:.2f} min, duration={args.duration_minutes:.2f} min")
    elif end_seconds is not None:
        print(f"[INFO] Processing source window: start={args.start_minutes:.2f} min, end={args.end_minutes:.2f} min")
    elif start_seconds > 0:
        print(f"[INFO] Processing source starting from minute {args.start_minutes:.2f}")

    train_csv, eval_csv, speaker_reference = build_segments(
        full_audio=full_audio,
        output_dir=output_dir,
        language=args.language,
        whisper_model_name=args.whisper_model,
        speaker_name=args.speaker_name,
        buffer_seconds=args.buffer_seconds,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        eval_split=args.eval_split,
        sample_rate=args.sample_rate,
        compute_type=args.compute_type,
    )

    print("\n[OK] XTTS dataset ready")
    print(f"[OK] Train metadata: {train_csv}")
    print(f"[OK] Eval metadata:  {eval_csv}")
    print(f"[OK] Reference audio: {speaker_reference}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(130)
