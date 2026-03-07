#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube video, detect speech with Silero VAD, and transcribe clips with faster-whisper."
    )
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset_output"),
        help="Directory to save downloads, clips, audio and manifest.",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=0,
        help="Maximum number of clips to export (0 = all).",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.8,
        help="Minimum clip duration in seconds.",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum clip duration in seconds.",
    )
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold (0.0-1.0, higher = more strict).",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=int,
        default=250,
        help="Minimum speech chunk duration in milliseconds.",
    )
    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        default=300,
        help="Minimum silence duration between speech chunks in milliseconds.",
    )
    parser.add_argument(
        "--speech_pad_ms",
        type=int,
        default=100,
        help="Padding added to speech boundaries in milliseconds.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="small",
        help="faster-whisper model size/path (tiny, base, small, medium, large-v3, etc).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language code for transcription, e.g. en, hi, bn.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        help="faster-whisper compute type (int8, float16, float32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for faster-whisper.",
    )
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"{name} is required but was not found in PATH.")


def require_python_package(import_name: str, install_name: str) -> None:
    try:
        __import__(import_name)
    except ModuleNotFoundError as error:
        raise SystemExit(f"Missing dependency `{install_name}`. Install with: pip install {install_name}") from error


def detect_speech_segments_vad(
    audio_path: Path,
    vad_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> List[Tuple[float, float]]:
    """Use Silero VAD to detect speech segments."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (get_speech_timestamps, *_) = utils

    # Load audio with soundfile
    wav, sr = sf.read(str(audio_path), dtype='float32')
    
    # Convert to mono if stereo
    if len(wav.shape) > 1 and wav.shape[1] > 1:
        wav = wav.mean(axis=1)
    
    # Resample to 16kHz if needed (using simple decimation)
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(wav) * 16000 / sr)
        wav = signal.resample(wav, num_samples)
    
    # Convert to torch tensor
    wav = torch.from_numpy(wav)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=vad_threshold,
        sampling_rate=16000,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    segments: List[Tuple[float, float]] = []
    for ts in speech_timestamps:
        start_sec = ts["start"] / 16000.0
        end_sec = ts["end"] / 16000.0
        segments.append((start_sec, end_sec))

    return segments


def download_video(url: str, download_dir: Path) -> Tuple[Path, str]:
    import yt_dlp

    download_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id")
    if not video_id:
        raise RuntimeError("Could not determine YouTube video id.")

    candidates = sorted(download_dir.glob(f"{video_id}.*"))
    video_path = next((p for p in candidates if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}), None)

    if video_path is None:
        raise RuntimeError("Video downloaded but no video file found.")

    return video_path, video_id


def extract_audio_for_detection(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut_video_clip(video_path: Path, out_path: Path, start: float, end: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut_audio_clip(video_path: Path, out_path: Path, start: float, end: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(audio_path: Path, model, language: str) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    text_parts: List[str] = []
    for seg in segments:
        chunk = seg.text.strip()
        if chunk:
            text_parts.append(chunk)
    return " ".join(text_parts).strip()


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def build_manifest(
    video_path: Path,
    video_id: str,
    segments: List[Tuple[float, float]],
    output_dir: Path,
    min_duration: float,
    max_duration: float,
    max_clips: int,
    whisper_model: str,
    language: str,
    compute_type: str,
    device: str,
) -> Path:
    from faster_whisper import WhisperModel

    clips_dir = output_dir / "clips"
    audio_clips_dir = output_dir / "audio_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    audio_clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"

    whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    kept = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["clip_path", "audio_path", "start", "end", "duration", "text"])

        for idx, (start, end) in enumerate(segments, start=1):
            duration = end - start
            if duration < min_duration or duration > max_duration:
                continue

            clip_name = f"{video_id}_{idx:06d}.mp4"
            wav_name = f"{video_id}_{idx:06d}.wav"
            clip_path = clips_dir / clip_name
            wav_path = audio_clips_dir / wav_name

            cut_video_clip(video_path, clip_path, start, end)
            cut_audio_clip(video_path, wav_path, start, end)

            text = transcribe_audio(wav_path, whisper, language=language)
            if not text:
                clip_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                continue

            writer.writerow([
                str(clip_path.relative_to(output_dir)),
                str(wav_path.relative_to(output_dir)),
                f"{start:.3f}",
                f"{end:.3f}",
                f"{duration:.3f}",
                text,
            ])
            kept += 1

            if max_clips > 0 and kept >= max_clips:
                break

    return manifest_path


def main() -> None:
    args = parse_args()

    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_python_package("yt_dlp", "yt-dlp")
    require_python_package("faster_whisper", "faster-whisper")
    require_python_package("torch", "torch")
    require_python_package("soundfile", "soundfile")
    require_python_package("scipy", "scipy")

    output_dir = args.output_dir.resolve()
    downloads_dir = output_dir / "downloads"
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading video...")
    video_path, video_id = download_video(args.url, downloads_dir)

    print("[2/4] Extracting audio + detecting speech with Silero VAD...")
    detection_wav = artifacts_dir / f"{video_id}_full.wav"
    extract_audio_for_detection(video_path, detection_wav)
    segments = detect_speech_segments_vad(
        audio_path=detection_wav,
        vad_threshold=args.vad_threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    print("[3/4] Slicing clips + whisper transcription...")
    device = resolve_device(args.device)
    manifest = build_manifest(
        video_path=video_path,
        video_id=video_id,
        segments=segments,
        output_dir=output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_clips=args.max_clips,
        whisper_model=args.whisper_model,
        language=args.language,
        compute_type=args.compute_type,
        device=device,
    )

    print("[4/4] Done.")
    print(f"Video:      {video_path}")
    print(f"Segments:   {len(segments)}")
    print(f"Manifest:   {manifest}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
