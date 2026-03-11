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

Example using a YouTube URL with placeholders and a 6-minute trim window:

```bash
conda run -n tts python xtts_dataset_builder.py \
  "https://www.youtube.com/watch?v=<VIDEO_ID>&t=<START_SECONDS>s" \
  --output_dir xtts_dataset_<VIDEO_ID> \
  --language en \
  --speaker_name <SPEAKER_NAME> \
  --start_minutes <START_MINUTES> \
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
  --dataset_dir xtts_dataset_<VIDEO_ID> \
  --language en \
  --output_dir xtts_runs \
  --run_name xtts_<SPEAKER_NAME>_ft \
  --batch_size 2 \
  --epochs 6
```

Training artifacts are written under:

- `xtts_runs/xtts_<SPEAKER_NAME>_ft/<run-folder>/best_model.pth`
- `xtts_runs/xtts_<SPEAKER_NAME>_ft/<run-folder>/trainer_0_log.txt`
- `xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/config.json`
- `xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/vocab.json`

## 3) Run inference

```bash
conda run -n tts python xtts_infer.py \
  --checkpoint xtts_runs/xtts_<SPEAKER_NAME>_ft/<run-folder>/best_model.pth \
  --config xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/config.json \
  --vocab xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/vocab.json \
  --speaker_wav xtts_dataset_<VIDEO_ID>/speaker_reference.wav \
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
