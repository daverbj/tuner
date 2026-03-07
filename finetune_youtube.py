#!/usr/bin/env python3
"""
Quick finetuning script for YouTube dataset using FastPitch.
Uses the Silero VAD + Whisper generated dataset.
"""

import argparse
import threading
import time
from pathlib import Path

from youtube_formatter import youtube_vad


def _stream_trainer_log_to_stdout(trainer, stop_event, poll_interval=0.5):
    """Continuously stream trainer log file to stdout."""
    log_path = Path(trainer.output_path) / "trainer_0_log.txt"
    printed_header = False
    file_pos = 0

    while not stop_event.is_set():
        if log_path.exists():
            if not printed_header:
                print(f"Streaming trainer logs from: {log_path}")
                printed_header = True

            try:
                with open(log_path, "r", encoding="utf-8") as log_file:
                    log_file.seek(file_pos)
                    chunk = log_file.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                        file_pos = log_file.tell()
            except FileNotFoundError:
                pass

        time.sleep(poll_interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune FastPitch on a YouTube VAD dataset.")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to dataset directory containing manifest.csv and audio clips.",
    )
    parser.add_argument(
        "--meta_file",
        default="manifest.csv",
        help="Metadata file name inside dataset path (default: manifest.csv).",
    )
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).parent / "finetune_output"),
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--run_name",
        default="youtube_fastpitch_finetune",
        help="Run name used for trainer outputs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
    from TTS.tts.configs.fast_pitch_config import FastPitchConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    output_path = str(Path(args.output_path).expanduser().resolve())
    f0_cache_path = str(Path(output_path) / "f0_cache")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    meta_path = Path(dataset_path) / args.meta_file
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    dataset_config = BaseDatasetConfig(
        formatter="youtube_vad",
        meta_file_train=args.meta_file,
        path=dataset_path,
    )

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        do_trim_silence=False,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
        num_mels=80,
        hop_length=256,
        win_length=1024,
    )

    config = FastPitchConfig(
        run_name=args.run_name,
        audio=audio_config,
        batch_size=8,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        eval_split_size=0.2,
        epochs=args.epochs,
        run_eval=True,
        test_delay_epochs=10,
        text_cleaner="english_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=False,
        compute_f0=True,
        f0_cache_path=f0_cache_path,
        print_step=5,
        print_eval=True,
        save_step=500,
        save_best_after=100,
        mixed_precision=False,
        max_seq_len=500000,
        output_path=output_path,
        datasets=[dataset_config],
    )

    print(f"Dataset path: {dataset_path}")
    print(f"Metadata file: {meta_path}")
    print(f"Output path: {output_path}")

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("Loading dataset...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=youtube_vad,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    if len(train_samples) == 0:
        raise ValueError("No training samples found! Check your dataset path and manifest.csv")

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(
            continue_path="",
            restore_path="",
        ),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting training...")
    print("=" * 60)
    stop_log_stream = threading.Event()
    log_thread = threading.Thread(
        target=_stream_trainer_log_to_stdout,
        args=(trainer, stop_log_stream),
        daemon=True,
    )
    log_thread.start()

    try:
        trainer.fit()
    finally:
        stop_log_stream.set()
        log_thread.join(timeout=2)


if __name__ == "__main__":
    main()
