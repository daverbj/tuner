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
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


def _resolve_bin(name: str) -> str:
    """Return the absolute path to a binary, preferring the current Python env's bin dir."""
    env_bin = str(Path(sys.executable).parent)
    env_path = env_bin + os.pathsep + os.environ.get("PATH", "")
    resolved = shutil.which(name, path=env_path)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"'{name}' not found. Install it in your Python environment: pip install {name}"
    )


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def run_command(command: list[str], description: str) -> None:
    resolved_command = [_resolve_bin(command[0])] + command[1:]
    print(f"\n[RUN] {description}")
    print(" ".join(resolved_command))
    subprocess.run(resolved_command, check=True)


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


def _silero_pause_chunks(
    full_audio: Path,
    buffer_seconds: float,
    min_duration: float,
    max_duration: float,
    vad_sample_rate: int,
    vad_min_silence_ms: int,
    vad_min_speech_ms: int,
) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    print(
        "[INFO] Detecting pause-based chunks with Silero VAD "
        f"(sr={vad_sample_rate}, min_silence_ms={vad_min_silence_ms}, min_speech_ms={vad_min_speech_ms})"
    )
    model = load_silero_vad()
    audio = read_audio(str(full_audio), sampling_rate=vad_sample_rate)
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=vad_sample_rate,
        min_silence_duration_ms=vad_min_silence_ms,
        min_speech_duration_ms=vad_min_speech_ms,
    )

    chunks: list[tuple[float, float]] = []
    for timestamp in speech_timestamps:
        start = max((timestamp["start"] / vad_sample_rate) - buffer_seconds, 0.0)
        end = (timestamp["end"] / vad_sample_rate) + buffer_seconds
        duration = end - start

        if duration < min_duration:
            continue

        if duration <= max_duration:
            chunks.append((start, end))
            continue

        cursor = start
        while cursor < end:
            window_end = min(cursor + max_duration, end)
            if (window_end - cursor) >= min_duration:
                chunks.append((cursor, window_end))
            cursor = window_end

    if not chunks:
        raise RuntimeError("Silero VAD did not detect usable speech chunks.")

    print(f"[INFO] Silero produced {len(chunks)} pause-based chunks")
    return chunks


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
    vad_sample_rate: int,
    vad_min_silence_ms: int,
    vad_min_speech_ms: int,
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

    chunks = _silero_pause_chunks(
        full_audio=full_audio,
        buffer_seconds=buffer_seconds,
        min_duration=min_duration,
        max_duration=max_duration,
        vad_sample_rate=vad_sample_rate,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_min_speech_ms=vad_min_speech_ms,
    )

    metadata_rows: list[dict[str, str]] = []
    sentence_index = 0

    for start_time, end_time in chunks:
        duration = end_time - start_time
        if not (min_duration <= duration <= max_duration):
            continue

        filename = f"segment_{sentence_index:06d}.wav"
        relative_audio_path = Path("wavs") / filename
        output_audio = output_dir / relative_audio_path
        cut_audio_segment(full_audio, output_audio, start_time, end_time, sample_rate)

        segments, _ = model.transcribe(str(output_audio), language=language, vad_filter=False)
        text = normalize_text(" ".join(seg.text for seg in segments))
        if not text:
            output_audio.unlink(missing_ok=True)
            continue

        metadata_rows.append(
            {
                "audio_file": relative_audio_path.as_posix(),
                "text": text,
                "speaker_name": speaker_name,
                "duration": f"{duration:.3f}",
            }
        )
        sentence_index += 1

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
    parser.add_argument("--vad_sample_rate", type=int, default=16000, help="Sample rate used by Silero VAD.")
    parser.add_argument("--vad_min_silence_ms", type=int, default=450, help="Pause threshold for Silero chunk split.")
    parser.add_argument("--vad_min_speech_ms", type=int, default=250, help="Minimum speech length considered by Silero VAD.")
    parser.add_argument("--eval_split", type=float, default=0.15, help="Fraction of samples reserved for eval.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for train/eval split.")
    parser.add_argument("--start_minutes", type=float, default=0.0, help="Start processing from this minute offset.")
    parser.add_argument("--duration_minutes", type=float, default=None, help="Only process this many minutes from the start offset.")
    parser.add_argument("--end_minutes", type=float, default=None, help="Stop processing at this minute offset.")
    return parser.parse_args()


def build_xtts_dataset(
    input_source: str,
    output_dir: str = "xtts_dataset",
    language: str = "en",
    speaker_name: str = "speaker",
    whisper_model: str = "small",
    compute_type: str = "float16",
    sample_rate: int = 22050,
    buffer_seconds: float = 0.20,
    min_duration: float = 0.5,
    max_duration: float = 11.0,
    eval_split: float = 0.15,
    seed: int = 1337,
    start_minutes: float = 0.0,
    duration_minutes: float | None = None,
    end_minutes: float | None = None,
    vad_sample_rate: int = 16000,
    vad_min_silence_ms: int = 450,
    vad_min_speech_ms: int = 250,
) -> tuple[Path, Path, Path]:
    random.seed(seed)

    output_path = Path(output_dir).expanduser().resolve()
    source_dir = output_path / "source"
    artifacts_dir = output_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    start_seconds = start_minutes * 60.0
    duration_seconds = duration_minutes * 60.0 if duration_minutes is not None else None
    end_seconds = end_minutes * 60.0 if end_minutes is not None else None

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

    source_media = download_source(input_source, source_dir)
    full_audio = artifacts_dir / "full_audio.wav"
    extract_audio(
        source_media,
        full_audio,
        sample_rate=sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        end_seconds=end_seconds,
    )

    if duration_seconds is not None:
        print(f"[INFO] Processing source window: start={start_minutes:.2f} min, duration={duration_minutes:.2f} min")
    elif end_seconds is not None:
        print(f"[INFO] Processing source window: start={start_minutes:.2f} min, end={end_minutes:.2f} min")
    elif start_seconds > 0:
        print(f"[INFO] Processing source starting from minute {start_minutes:.2f}")

    train_csv, eval_csv, speaker_reference = build_segments(
        full_audio=full_audio,
        output_dir=output_path,
        language=language,
        whisper_model_name=whisper_model,
        speaker_name=speaker_name,
        buffer_seconds=buffer_seconds,
        min_duration=min_duration,
        max_duration=max_duration,
        eval_split=eval_split,
        sample_rate=sample_rate,
        compute_type=compute_type,
        vad_sample_rate=vad_sample_rate,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_min_speech_ms=vad_min_speech_ms,
    )

    print("\n[OK] XTTS dataset ready")
    print(f"[OK] Train metadata: {train_csv}")
    print(f"[OK] Eval metadata:  {eval_csv}")
    print(f"[OK] Reference audio: {speaker_reference}")
    return train_csv, eval_csv, speaker_reference


def main() -> None:
    args = parse_args()
    build_xtts_dataset(
        input_source=args.input_source,
        output_dir=args.output_dir,
        language=args.language,
        speaker_name=args.speaker_name,
        whisper_model=args.whisper_model,
        compute_type=args.compute_type,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        eval_split=args.eval_split,
        seed=args.seed,
        start_minutes=args.start_minutes,
        duration_minutes=args.duration_minutes,
        end_minutes=args.end_minutes,
        vad_sample_rate=args.vad_sample_rate,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(130)
