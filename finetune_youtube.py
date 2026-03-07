#!/usr/bin/env python3
"""
Quick finetuning script for YouTube dataset using FastPitch.
Uses the Silero VAD + Whisper generated dataset.
"""

import os
import sys
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


# Configuration
DATASET_PATH = str(Path(__file__).parent / "dataset_silero_vad")
OUTPUT_PATH = str(Path(__file__).parent / "finetune_output")
RUN_NAME = "youtube_fastpitch_finetune"

# Dataset config with custom formatter
dataset_config = BaseDatasetConfig(
    formatter="youtube_vad",
    meta_file_train="manifest.csv",
    path=DATASET_PATH,
)

# Audio config matching the YouTube audio (16kHz mono)
audio_config = BaseAudioConfig(
    sample_rate=16000,
    do_trim_silence=False,  # Already trimmed by VAD
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

# FastPitch config optimized for quick finetuning
config = FastPitchConfig(
    run_name=RUN_NAME,
    audio=audio_config,
    
    # Training settings
    batch_size=8,  # Small batch for limited data
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    
    # Dataset splits
    eval_split_size=0.2,  # 20% for eval (10 samples out of 50)
    
    # Reduced epochs for quick test
    epochs=50,
    run_eval=True,
    test_delay_epochs=10,
    
    # Text processing
    text_cleaner="english_cleaners",
    use_phonemes=False,  # Disable phonemes for faster training
    
    # Caching and preprocessing
    compute_input_seq_cache=False,
    compute_f0=False,  # Disable F0 for faster training
    
    # Logging
    print_step=5,
    print_eval=True,
    save_step=500,
    save_best_after=100,
    
    # Performance
    mixed_precision=False,
    max_seq_len=500000,
    
    # Output
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
)

print(f"Dataset path: {DATASET_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# Initialize audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load data samples
print("Loading dataset...")
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

print(f"Train samples: {len(train_samples)}")
print(f"Eval samples: {len(eval_samples)}")

if len(train_samples) == 0:
    raise ValueError("No training samples found! Check your dataset path and manifest.csv")

# Initialize model
model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)

# Initialize trainer
trainer = Trainer(
    TrainerArgs(
        continue_path="",  # Set to checkpoint path to resume
        restore_path="",   # Set to pretrained model to finetune from checkpoint
    ),
    config,
    OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Start training
print("\nStarting training...")
print("=" * 60)
trainer.fit()
