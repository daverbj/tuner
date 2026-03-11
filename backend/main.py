from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import threading
import traceback
from uuid import uuid4
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from queue import Empty, Queue
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from xtts_dataset_builder import build_xtts_dataset
from xtts_finetune import run_xtts_finetune
from xtts_infer import run_xtts_infer


class StartRequest(BaseModel):
    youtube_url: str = Field(..., description="Full YouTube URL")
    language: str = Field(default="en")
    speaker_name: str = Field(default="speaker")
    start_minutes: float = Field(default=0.0)
    duration_minutes: float = Field(default=6.0)
    output_dataset_dir: Optional[str] = Field(default=None)
    run_name: str = Field(default="xtts_ui_ft")
    batch_size: int = Field(default=2)
    epochs: int = Field(default=6)
    output_runs_dir: str = Field(default="xtts_runs")


class InferRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier from /api/inference/models")
    speaker_wav: str = Field(..., description="Speaker reference wav path")
    text: str = Field(..., description="Text to synthesize")
    language: str = Field(default="en")


@dataclass
class RuntimeState:
    process: Optional[mp.Process] = None
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    logs: Queue[str] = field(default_factory=Queue)
    worker_logs: Optional[mp.Queue] = None
    command: Optional[list[str]] = None
    stage: Optional[str] = None
    running: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None


state = RuntimeState()
app = FastAPI(title="XTTS Finetune API")
INFERENCE_OUTPUT_DIR = ROOT_DIR / "xtts_infer_outputs"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enqueue(line: str) -> None:
    if not line.endswith("\n"):
        line += "\n"
    state.logs.put(line)


def _extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL. Missing video id.")
    return video_id


class QueueWriter(TextIOBase):
    def __init__(self, q: mp.Queue):
        super().__init__()
        self.q = q

    def write(self, s: str) -> int:
        if s:
            self.q.put(s)
        return len(s)

    def flush(self) -> None:
        return None


def _pipeline_worker_function(payload_data: dict, q: mp.Queue) -> None:
    writer = QueueWriter(q)
    payload = StartRequest(**payload_data)

    try:
        with redirect_stdout(writer), redirect_stderr(writer):
            video_id = _extract_video_id(payload.youtube_url)
            dataset_dir = payload.output_dataset_dir or f"xtts_dataset_{video_id}"

            print(f"[dataset-build] video_id={video_id}")
            print(f"[dataset-build] output_dir={dataset_dir}")
            build_xtts_dataset(
                input_source=payload.youtube_url,
                output_dir=dataset_dir,
                language=payload.language,
                speaker_name=payload.speaker_name,
                start_minutes=payload.start_minutes,
                duration_minutes=payload.duration_minutes,
            )

            print("[finetune] starting")
            run_xtts_finetune(
                dataset_dir=dataset_dir,
                language=payload.language,
                output_dir=payload.output_runs_dir,
                run_name=payload.run_name,
                batch_size=payload.batch_size,
                epochs=payload.epochs,
            )

            print("[pipeline] completed successfully")
    except Exception:
        print("[pipeline-error]" + traceback.format_exc())
        raise
    finally:
        q.put("[worker-end]\n")


def _log_pump_thread() -> None:
    while True:
        with state.process_lock:
            proc = state.process
            q = state.worker_logs
        if q is None:
            return

        try:
            line = q.get(timeout=0.3)
            _enqueue(line)
        except Empty:
            pass

        if proc is None:
            continue
        if not proc.is_alive():
            code = proc.exitcode
            with state.process_lock:
                state.return_code = code
                state.finished_at = datetime.utcnow().isoformat()
                state.running = False
                state.stage = None
                state.process = None
                state.worker_logs = None
            _enqueue(f"\n[process-exit] return_code={code}\n")
            return


def _list_model_entries(base_dir: Path) -> list[dict]:
    if not base_dir.exists():
        return []

    entries: list[dict] = []
    for run_root in sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda item: item.name):
        base_config = run_root / "xtts_v2_base" / "config.json"
        base_vocab = run_root / "xtts_v2_base" / "vocab.json"
        if not base_config.exists() or not base_vocab.exists():
            continue

        training_runs = [path for path in run_root.iterdir() if path.is_dir() and path.name != "xtts_v2_base"]
        for training_run in sorted(training_runs, key=lambda item: item.stat().st_mtime, reverse=True):
            checkpoints = sorted(training_run.glob("best_model*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
            if not checkpoints:
                checkpoints = sorted(training_run.glob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
            if not checkpoints:
                continue

            checkpoint = checkpoints[0]
            model_id = f"{run_root.name}:{training_run.name}:{checkpoint.name}"
            entries.append(
                {
                    "model_id": model_id,
                    "run_name": run_root.name,
                    "training_run": training_run.name,
                    "checkpoint": str(checkpoint),
                    "config": str(base_config),
                    "vocab": str(base_vocab),
                    "modified_at": datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat(),
                }
            )

    entries.sort(key=lambda item: item["modified_at"], reverse=True)
    return entries


def _resolve_model_entry(model_id: str, base_dir: Path) -> dict:
    entries = _list_model_entries(base_dir)
    for entry in entries:
        if entry["model_id"] == model_id:
            return entry
    raise HTTPException(status_code=404, detail="Selected fine-tuned model was not found.")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/inference/models")
def inference_models(output_runs_dir: str = "xtts_runs") -> dict:
    base_dir = (ROOT_DIR / output_runs_dir).resolve()
    return {"items": _list_model_entries(base_dir)}


@app.get("/api/inference/speakers")
def inference_speakers() -> dict:
    files = sorted(ROOT_DIR.glob("xtts_dataset_*/speaker_reference.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
    items = []
    for path in files:
        full_path = str(path.resolve())
        items.append(
            {
                "path": full_path,
                "label": path.parent.name,
                "preview_url": f"/api/inference/speaker-audio?path={quote_plus(full_path)}",
            }
        )
    return {"items": items}


@app.get("/api/inference/speaker-audio")
def inference_speaker_audio(path: str):
    speaker_path = Path(path).expanduser().resolve()
    if not speaker_path.exists() or speaker_path.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="Speaker audio file not found.")
    if ROOT_DIR.resolve() not in speaker_path.parents:
        raise HTTPException(status_code=400, detail="Invalid speaker audio path.")
    return FileResponse(speaker_path, media_type="audio/wav", filename=speaker_path.name)


@app.post("/api/inference/generate")
def inference_generate(payload: InferRequest, output_runs_dir: str = "xtts_runs") -> dict:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    speaker_wav = Path(payload.speaker_wav).expanduser().resolve()
    if not speaker_wav.exists():
        raise HTTPException(status_code=404, detail="Speaker wav not found.")

    model_entry = _resolve_model_entry(payload.model_id, (ROOT_DIR / output_runs_dir).resolve())

    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = f"infer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.wav"
    output_path = INFERENCE_OUTPUT_DIR / output_name

    try:
        run_xtts_infer(
            checkpoint=model_entry["checkpoint"],
            config_path=model_entry["config"],
            vocab=model_entry["vocab"],
            speaker_wav=str(speaker_wav),
            text=payload.text,
            language=payload.language,
            output=str(output_path),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

    return {
        "ok": True,
        "audio_url": f"/api/inference/audio/{output_name}",
        "audio_path": str(output_path),
        "model_id": payload.model_id,
    }


@app.get("/api/inference/audio/{file_name}")
def inference_audio(file_name: str):
    path = (INFERENCE_OUTPUT_DIR / file_name).resolve()
    if not path.exists() or path.parent != INFERENCE_OUTPUT_DIR.resolve():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/finetune/status")
def finetune_status() -> dict:
    with state.process_lock:
        return {
            "running": state.running,
            "stage": state.stage,
            "command": state.command,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "return_code": state.return_code,
        }


@app.post("/api/finetune/start")
def finetune_start(payload: StartRequest) -> dict:
    with state.process_lock:
        if state.running:
            raise HTTPException(status_code=409, detail="A finetune process is already running.")

        if payload.batch_size < 1 or payload.epochs < 1:
            raise HTTPException(status_code=400, detail="batch_size and epochs must be >= 1")
        if payload.duration_minutes <= 0:
            raise HTTPException(status_code=400, detail="duration_minutes must be > 0")
        if payload.start_minutes < 0:
            raise HTTPException(status_code=400, detail="start_minutes must be >= 0")

        _extract_video_id(payload.youtube_url)

        while True:
            try:
                state.logs.get_nowait()
            except Empty:
                break

        state.running = True
        state.command = ["function-call", "build_xtts_dataset", "run_xtts_finetune"]
        state.stage = "starting"
        state.started_at = datetime.utcnow().isoformat()
        state.finished_at = None
        state.return_code = None
        _enqueue("[process-start] pipeline accepted\n")

        worker_logs: mp.Queue = mp.Queue()
        proc = mp.Process(target=_pipeline_worker_function, args=(payload.model_dump(), worker_logs), daemon=False)
        proc.start()
        state.process = proc
        state.worker_logs = worker_logs
        state.stage = "dataset-build"

        threading.Thread(target=_log_pump_thread, daemon=True).start()
        return {"started": True, "stage": state.stage}


@app.post("/api/finetune/stop")
def finetune_stop() -> dict:
    with state.process_lock:
        if not state.running:
            return {"stopped": False, "detail": "No running process."}
        if state.process is not None and state.process.is_alive():
            state.process.terminate()
            _enqueue("[process-stop] terminate sent\n")
        else:
            _enqueue("[process-stop] waiting stage, stop will apply on next process\n")
        return {"stopped": True}


@app.get("/api/finetune/stream")
async def finetune_stream() -> StreamingResponse:
    async def event_generator():
        yield "event: hello\ndata: connected\n\n"
        while True:
            try:
                line = state.logs.get(timeout=0.5)
                payload = json.dumps({"line": line})
                yield f"event: log\ndata: {payload}\n\n"
            except Empty:
                await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")