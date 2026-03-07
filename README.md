# TTS Inference Runner

This directory contains a simple inference script:

- `infer_tts.py` (runs text-to-speech using Coqui TTS)
- `youtube_dataset_builder.py` (downloads YouTube video, detects speech with Silero VAD, transcribes with faster-whisper)

## Quickstart (fresh clone)

```bash
git clone <your-repo-url>
cd tuner
conda create -y -n tts python=3.10
conda activate tts
git clone https://github.com/coqui-ai/TTS.git
pip install -e ./TTS
pip install yt-dlp faster-whisper scipy soundfile
python infer_tts.py --text "Hello from Coqui TTS" --out_path output.wav
```

macOS system dependency:

```bash
brew install ffmpeg
```

## 1) Create and prepare Conda environment

```bash
conda create -y -n tts python=3.10
conda activate tts
git clone https://github.com/coqui-ai/TTS.git
pip install -e ./TTS
```

## 2) Run the script

```bash
python infer_tts.py --help
```

## 3) Generate speech (quick test)

```bash
python infer_tts.py \
  --text "Hello from Coqui TTS" \
  --out_path output.wav \
  --model_name tts_models/en/ljspeech/glow-tts
```

The first run downloads model files, so it can take a while.

## 4) Play output audio (macOS)

```bash
afplay output.wav
```

## Build a finetuning dataset from YouTube (Silero VAD + Whisper)

### Install extra dependencies

```bash
conda run -n tts pip install yt-dlp faster-whisper
```

Also make sure `ffmpeg` and `ffprobe` are installed and available in your PATH.

### Run dataset builder

```bash
python youtube_dataset_builder.py \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output_dir dataset_output \
  --language en \
  --whisper_model small
```

### Output structure

- `dataset_output/downloads/` original downloaded video.
- `dataset_output/artifacts/` extracted full-audio WAV used for VAD detection.
- `dataset_output/clips/` video segments cut on speech boundaries detected by Silero VAD.
- `dataset_output/audio_clips/` per-segment WAV files used for transcription.
- `dataset_output/manifest.csv` rows with clip path, audio path, timestamps, and Whisper transcript.

### Useful options

- `--max_clips 200` limit number of exported clips.
- `--min_duration 0.8` skip very short clips.
- `--max_duration 20.0` skip very long clips.
- `--vad_threshold 0.5` Silero VAD sensitivity (0.0-1.0, higher = stricter).
- `--min_speech_duration_ms 250` minimum speech chunk duration.
- `--min_silence_duration_ms 300` minimum silence between speech chunks.
- `--speech_pad_ms 100` padding around detected speech boundaries.
- `--compute_type int8` lower memory usage for transcription.

## Finetune FastPitch on your dataset

Prerequisite: create a dataset first (for example `dataset_silero_vad/manifest.csv` and clip WAVs).

### Run finetuning

```bash
python finetune_youtube.py --dataset_path /path/to/your/dataset
```

Notes:

- `--dataset_path` is required and must contain `manifest.csv` plus referenced audio files.
- Optional: set `--meta_file` if your metadata file is not named `manifest.csv`.
- Optional: set `--output_path` and `--run_name` to control checkpoint/log location.
- Training logs are streamed live to your terminal.
- Checkpoints and config are saved under `finetune_output/<run-name>/`.
- On a GPU machine, make sure CUDA-enabled `torch` is installed in the `tts` env.

### Test a finetuned checkpoint

```bash
python test_finetuned.py \
  --checkpoint finetune_output/<run-name>/best_model.pth \
  --config finetune_output/<run-name>/config.json \
  --text "This is a finetuned voice test" \
  --output finetuned_test.wav
```

Play result on macOS:

```bash
afplay finetuned_test.wav
```

## Optional: activate environment and run normally

```bash
conda activate tts
python infer_tts.py --text "Test" --out_path output.wav
```

## Common options

- `--text`: Text to synthesize.
- `--out_path`: Output WAV path.
- `--model_name`: Coqui model name.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--language`: Language code (for multilingual models).
- `--speaker`: Speaker id/name (for multi-speaker models).
- `--speaker_wav`: Reference WAV for voice cloning models.

## Troubleshooting

- If you see missing package errors, reinstall in env:

```bash
pip install -e ./TTS
```

- If Conda is not found, initialize shell first:

```bash
conda init zsh
exec zsh
```

- If transcription is poor, try a larger Whisper model (for example `--whisper_model medium`) and tune `--vad_threshold` / `--min_silence_duration_ms`.
