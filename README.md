# XTTS Fine-Tuning Pipeline

This project provides an end-to-end XTTS workflow:

- `xtts_dataset_builder.py`: build XTTS-ready dataset from local media or YouTube.
- `xtts_finetune.py`: fine-tune XTTS-v2 on that dataset.
- `xtts_infer.py`: run inference with the fine-tuned checkpoint.

## Quickstart

```bash
git clone <your-repo-url>
cd tuner
conda create -y -n tts python=3.10
conda activate tts
git clone https://github.com/coqui-ai/TTS.git
pip install -e ./TTS
pip install yt-dlp faster-whisper torchcodec
```

macOS dependency:

```bash
brew install ffmpeg
```

## 1) Build dataset from video

Example using only a 6-minute slice starting at 504s with speaker name `gd`:

```bash
conda run -n tts python xtts_dataset_builder.py \
  "https://www.youtube.com/watch?v=6ag7FU0D-SA&t=504s" \
  --output_dir xtts_dataset_6ag7FU0D-SA \
  --language en \
  --speaker_name gd \
  --start_minutes 8.4 \
  --duration_minutes 6
```

Dataset output layout:

- `metadata_train.csv`
- `metadata_eval.csv`
- `wavs/*.wav`
- `speaker_reference.wav`

## 2) Fine-tune XTTS

```bash
conda run -n tts python xtts_finetune.py \
  --dataset_dir xtts_dataset_6ag7FU0D-SA \
  --language en \
  --output_dir xtts_runs \
  --run_name xtts_gd_ft \
  --batch_size 2 \
  --epochs 6
```

Training artifacts are written under:

- `xtts_runs/xtts_gd_ft/<run-folder>/best_model.pth`
- `xtts_runs/xtts_gd_ft/<run-folder>/trainer_0_log.txt`
- `xtts_runs/xtts_gd_ft/xtts_v2_base/config.json`
- `xtts_runs/xtts_gd_ft/xtts_v2_base/vocab.json`

## 3) Run inference

```bash
conda run -n tts python xtts_infer.py \
  --checkpoint xtts_runs/xtts_gd_ft/<run-folder>/best_model.pth \
  --config xtts_runs/xtts_gd_ft/xtts_v2_base/config.json \
  --vocab xtts_runs/xtts_gd_ft/xtts_v2_base/vocab.json \
  --speaker_wav xtts_dataset_6ag7FU0D-SA/speaker_reference.wav \
  --text "One quiet evening, a traveler found a lantern by the bridge." \
  --language en \
  --output xtts_story.wav
```

Play on macOS:

```bash
afplay xtts_story.wav
```

## Notes

- Keep text under ~250 characters per inference call for cleaner XTTS output.
- `xtts_finetune.py` includes a PyTorch 2.6 checkpoint-load compatibility patch.
- `torchcodec` is required in this environment.
