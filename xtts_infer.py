#!/usr/bin/env python3
"""Run inference with a fine-tuned XTTS model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned XTTS model.")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned XTTS checkpoint.")
    parser.add_argument("--config", required=True, help="Path to XTTS config.json.")
    parser.add_argument("--vocab", required=True, help="Path to XTTS vocab.json.")
    parser.add_argument("--speaker_wav", required=True, help="Reference speaker WAV.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--language", default="en", help="Language code.")
    parser.add_argument("--output", default="xtts_output.wav", help="Output WAV path.")
    return parser.parse_args()


def run_xtts_infer(
    checkpoint: str,
    config_path: str,
    vocab: str,
    speaker_wav: str,
    text: str,
    language: str = "en",
    output: str = "xtts_output.wav",
) -> Path:
    import torch
    import torchaudio

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)

    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(xtts_config, checkpoint_path=checkpoint, vocab_path=vocab, use_deepspeed=False)

    if torch.cuda.is_available():
        model.cuda()

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_wav,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    inference_result = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
    )

    waveform = torch.tensor(inference_result["wav"]).unsqueeze(0)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, 24000)
    print(f"Saved audio to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    run_xtts_infer(
        checkpoint=args.checkpoint,
        config_path=args.config,
        vocab=args.vocab,
        speaker_wav=args.speaker_wav,
        text=args.text,
        language=args.language,
        output=args.output,
    )


if __name__ == "__main__":
    main()
