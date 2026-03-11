#!/usr/bin/env python3
"""Standalone XTTS v2 fine-tuning script for datasets in Coqui metadata format.

Expected dataset layout:

    dataset_dir/
      metadata_train.csv
      metadata_eval.csv
      wavs/*.wav

The metadata CSV files must use pipe separators and columns:
`audio_file|text|speaker_name`
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XTTS v2 on a local dataset.")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory containing metadata_train.csv and metadata_eval.csv.")
    parser.add_argument("--language", default="en", help="Dataset language code.")
    parser.add_argument("--output_dir", default="xtts_runs", help="Directory where training runs/checkpoints will be written.")
    parser.add_argument("--run_name", default="xtts_v2_ft", help="Run name.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max_audio_seconds", type=float, default=11.0, help="Max training clip length in seconds.")
    parser.add_argument("--num_loader_workers", type=int, default=4, help="Train dataloader workers.")
    parser.add_argument("--eval_split_max_size", type=int, default=256, help="Max number of eval samples.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--print_step", type=int, default=10, help="How often to print training stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    import TTS.tts.layers.xtts.trainer.gpt_trainer as gpt_trainer_module
    import TTS.tts.models.xtts as xtts_model_module
    import TTS.utils.io as tts_io_module
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
    from TTS.utils.manage import ModelManager

    original_load_fsspec = tts_io_module.load_fsspec

    def trusted_load_fsspec(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load_fsspec(*load_args, **load_kwargs)

    tts_io_module.load_fsspec = trusted_load_fsspec
    xtts_model_module.load_fsspec = trusted_load_fsspec
    gpt_trainer_module.load_fsspec = trusted_load_fsspec

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_csv = dataset_dir / "metadata_train.csv"
    eval_csv = dataset_dir / "metadata_eval.csv"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not train_csv.exists():
        raise FileNotFoundError(f"Train metadata not found: {train_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval metadata not found: {eval_csv}")

    run_output = output_dir / args.run_name
    checkpoints_dir = run_output / "xtts_v2_base"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir:   {dataset_dir}")
    print(f"Train CSV:     {train_csv}")
    print(f"Eval CSV:      {eval_csv}")
    print(f"Output dir:    {run_output}")
    print(f"Language:      {args.language}")

    dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    vocab_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    model_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    config_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    dvae_checkpoint = checkpoints_dir / "dvae.pth"
    mel_stats = checkpoints_dir / "mel_stats.pth"
    vocab_file = checkpoints_dir / "vocab.json"
    xtts_checkpoint = checkpoints_dir / "model.pth"
    xtts_config_file = checkpoints_dir / "config.json"

    needed_files = [dvae_checkpoint, mel_stats, vocab_file, xtts_checkpoint, xtts_config_file]
    if not all(path.exists() for path in needed_files):
        print("Downloading XTTS base model files...")
        ModelManager._download_model_files(
            [mel_link, dvae_link, vocab_link, model_link, config_link],
            str(checkpoints_dir),
            progress_bar=True,
        )

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="xtts_ft_dataset",
        path=str(dataset_dir),
        meta_file_train=train_csv.name,
        meta_file_val=eval_csv.name,
        language=args.language,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=int(args.max_audio_seconds * 22050),
        max_text_length=200,
        mel_norm_file=str(mel_stats),
        dvae_checkpoint=str(dvae_checkpoint),
        xtts_checkpoint=str(xtts_checkpoint),
        tokenizer_file=str(vocab_file),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        epochs=args.epochs,
        output_path=str(run_output),
        model_args=model_args,
        run_name=args.run_name,
        project_name="XTTS_trainer",
        run_description="Standalone XTTS fine-tuning run",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=args.num_loader_workers,
        eval_split_max_size=args.eval_split_max_size,
        print_step=args.print_step,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    print("Initializing XTTS GPT trainer...")
    model = GPTTrainer.init_from_config(config)

    print("Loading dataset samples...")
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples:  {len(eval_samples)}")
    if not train_samples:
        raise RuntimeError("No training samples found.")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=str(run_output),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting XTTS fine-tuning...")
    trainer.fit()

    longest_sample = max(train_samples, key=lambda item: len(item["text"].split()))
    speaker_reference = longest_sample["audio_file"]

    print("\nXTTS fine-tuning complete.")
    print(f"Run output:        {trainer.output_path}")
    print(f"Base XTTS config:  {xtts_config_file}")
    print(f"Base XTTS vocab:   {vocab_file}")
    print(f"Speaker reference: {speaker_reference}")

    del model, trainer, train_samples, eval_samples
    gc.collect()


if __name__ == "__main__":
    main()
