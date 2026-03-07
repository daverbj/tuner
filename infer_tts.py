#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run text-to-speech inference with Coqui TTS.")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a Coqui TTS inference test.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output.wav",
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/glow-tts",
        help="Model name from Coqui model zoo.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument("--language", type=str, default=None, help="Language code for multilingual models.")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker name/id for multi-speaker models.")
    parser.add_argument(
        "--speaker_wav",
        type=str,
        default=None,
        help="Reference WAV path for voice cloning models (e.g., XTTS).",
    )
    return parser


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    try:
        from TTS.api import TTS
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency while importing Coqui TTS. "
            "Install project dependencies first, e.g. `pip install -e .`"
        ) from error

    output_path = Path(args.out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = TTS(model_name=args.model_name, progress_bar=True).to(device)

    synth_kwargs = {}
    if args.language:
        synth_kwargs["language"] = args.language
    if args.speaker:
        synth_kwargs["speaker"] = args.speaker
    if args.speaker_wav:
        synth_kwargs["speaker_wav"] = args.speaker_wav

    tts.tts_to_file(text=args.text, file_path=str(output_path), **synth_kwargs)
    print(f"Saved synthesized audio to: {output_path}")


if __name__ == "__main__":
    main()