#!/usr/bin/env python3
"""
Test script for finetuned model.
Loads the finetuned checkpoint and generates speech.
"""

import argparse
from pathlib import Path

import torch
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesizer import Synthesizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test finetuned TTS model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config.json",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The US Iran Israel war has reached a critical phase.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetuned_output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {args.device}")
    
    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)
    
    # Load model
    model = setup_model(config)
    checkpoint = torch.load(str(checkpoint_path), map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    if args.device == "cuda":
        model = model.cuda()
    
    # Create synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint=str(checkpoint_path),
        tts_config_path=str(config_path),
        use_cuda=(args.device == "cuda"),
    )
    
    # Generate speech
    print(f"\nGenerating speech for: '{args.text}'")
    wav = synthesizer.tts(args.text)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ap.save_wav(wav, str(output_path))
    
    print(f"\nSaved to: {output_path}")
    print(f"Duration: {len(wav) / ap.sample_rate:.2f} seconds")


if __name__ == "__main__":
    main()
