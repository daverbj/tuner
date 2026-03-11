# XTTS Fine-Tuning Pipeline

An end-to-end XTTS v2 workflow: build a dataset from a YouTube video, fine-tune the model, and run inference — all from a web UI or the command line.

| Script | Purpose |
|--------|----------|
| `xtts_dataset_builder.py` | Build XTTS-ready dataset from local media or YouTube. Chunks audio using Silero VAD pause detection. |
| `xtts_finetune.py` | Fine-tune XTTS v2 on the dataset. |
| `xtts_infer.py` | Synthesize speech from a fine-tuned checkpoint. |
| `backend/` | FastAPI server: dataset build + finetune pipeline, SSE logs, inference API. |
| `client/` | Next.js web UI: fine-tune control panel + inference page. |

---

## Setup

```bash
git clone <your-repo-url>
cd tuner
conda create -y -n tts python=3.10
conda activate tts
git clone https://github.com/coqui-ai/TTS.git
pip install -e ./TTS
pip install yt-dlp faster-whisper silero-vad torchcodec
```

macOS system dependency:

```bash
brew install ffmpeg
```

---

## Option A — Web UI

### 1) Start the backend

From the project root with the `tts` env:

```bash
$(conda info --base)/envs/tts/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Start the frontend

```bash
cd client
npm install
npm run dev
```

Open `http://localhost:3000`.

**Fine-tune page** (`/`): Paste a YouTube URL, set options (speaker name, language, duration, epochs, etc.), press **Start Fine-tune**. Live SSE logs stream in the browser. Press **Stop** to cancel.

**Inference page** (`/inference`): Select a completed fine-tuned model and a speaker reference, preview the reference audio, enter text, press **Generate Speech** to synthesise and play back the result.

---

## Option B — Command Line

### 1) Build dataset from video

```bash
conda run -n tts python xtts_dataset_builder.py \
  "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --output_dir xtts_dataset_<VIDEO_ID> \
  --language en \
  --speaker_name <SPEAKER_NAME> \
  --start_minutes <START_MINUTES> \
  --duration_minutes 6
```

Dataset output layout:

```
xtts_dataset_<VIDEO_ID>/
  metadata_train.csv
  metadata_eval.csv
  wavs/
  speaker_reference.wav
```

Optional VAD tuning flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--vad_min_silence_ms` | 450 | Pause threshold for chunk split (ms) |
| `--vad_min_speech_ms` | 250 | Minimum speech length kept (ms) |

### 2) Fine-tune XTTS

```bash
conda run -n tts python xtts_finetune.py \
  --dataset_dir xtts_dataset_<VIDEO_ID> \
  --language en \
  --output_dir xtts_runs \
  --run_name xtts_<SPEAKER_NAME>_ft \
  --batch_size 2 \
  --epochs 6
```

Artifacts written under `xtts_runs/xtts_<SPEAKER_NAME>_ft/`.

### 3) Run inference

```bash
conda run -n tts python xtts_infer.py \
  --checkpoint xtts_runs/xtts_<SPEAKER_NAME>_ft/<run-folder>/best_model.pth \
  --config    xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/config.json \
  --vocab     xtts_runs/xtts_<SPEAKER_NAME>_ft/xtts_v2_base/vocab.json \
  --speaker_wav xtts_dataset_<VIDEO_ID>/speaker_reference.wav \
  --text "Enter your text here." \
  --language en \
  --output output.wav
```

macOS playback:

```bash
afplay output.wav
```

---

## Notes

- Audio chunks are split on natural speech pauses via Silero VAD, not arbitrary timestamps.
- Keep inference text under ~250 characters per call for best quality.
- `xtts_finetune.py` includes a PyTorch 2.6 checkpoint-load compatibility patch.
- `torchcodec` is required for torchaudio audio loading in this environment.
