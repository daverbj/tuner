"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

type StatusResponse = {
  running: boolean;
  stage: string | null;
  command: string[] | null;
  started_at: string | null;
  finished_at: string | null;
  return_code: number | null;
};

export default function Home() {
  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000", []);
  const [youtubeUrl, setYoutubeUrl] = useState("https://www.youtube.com/watch?v=<VIDEO_ID>&t=<START_SECONDS>s");
  const [speakerName, setSpeakerName] = useState("gd");
  const [language, setLanguage] = useState("en");
  const [startMinutes, setStartMinutes] = useState("0");
  const [durationMinutes, setDurationMinutes] = useState("6");
  const [runName, setRunName] = useState("xtts_gd_ft");
  const [batchSize, setBatchSize] = useState("2");
  const [epochs, setEpochs] = useState("6");
  const [datasetDir, setDatasetDir] = useState("");
  const [outputRunsDir, setOutputRunsDir] = useState("xtts_runs");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [logs, setLogs] = useState("");
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const es = new EventSource(`${apiBase}/api/finetune/stream`);
    eventSourceRef.current = es;

    es.addEventListener("log", (event) => {
      const message = event as MessageEvent;
      try {
        const parsed = JSON.parse(message.data) as { line: string };
        setLogs((prev) => prev + parsed.line);
      } catch {
        setLogs((prev) => prev + message.data + "\n");
      }
    });

    return () => {
      es.close();
    };
  }, [apiBase]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    const checkStatus = async () => {
      const response = await fetch(`${apiBase}/api/finetune/status`, { cache: "no-store" });
      if (!response.ok) return;
      const json = (await response.json()) as StatusResponse;
      setStatus(json);
    };

    checkStatus();
    const intervalId = window.setInterval(checkStatus, 2500);
    return () => window.clearInterval(intervalId);
  }, [apiBase]);

  const startFinetune = async () => {
    setStarting(true);
    try {
      const response = await fetch(`${apiBase}/api/finetune/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          youtube_url: youtubeUrl,
          speaker_name: speakerName,
          language,
          start_minutes: Number(startMinutes),
          duration_minutes: Number(durationMinutes),
          output_dataset_dir: datasetDir.trim() ? datasetDir.trim() : null,
          run_name: runName,
          batch_size: Number(batchSize),
          epochs: Number(epochs),
          output_runs_dir: outputRunsDir,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        setLogs((prev) => prev + `\n[start-error] ${errorText}\n`);
        return;
      }

      setLogs((prev) => prev + "\n[start-request] accepted\n");
    } finally {
      setStarting(false);
    }
  };

  const stopFinetune = async () => {
    setStopping(true);
    try {
      await fetch(`${apiBase}/api/finetune/stop`, { method: "POST" });
    } finally {
      setStopping(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-100 p-6 text-zinc-900">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-6">
        <section className="rounded-xl border border-zinc-300 bg-white p-5 shadow-sm">
          <h1 className="text-2xl font-semibold">XTTS Fine-Tune Runner</h1>
          <p className="mt-2 text-sm text-zinc-600">
            Paste a full YouTube URL, set options, and start. The backend will build the dataset and then run
            fine-tuning automatically.
          </p>
          <p className="mt-2 text-sm">
            <Link href="/inference" className="text-blue-600 underline">
              Go to Inference Page
            </Link>
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <label className="grid gap-2 sm:col-span-2">
              <span className="text-sm font-medium">YouTube URL</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={youtubeUrl}
                onChange={(event) => setYoutubeUrl(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Speaker name</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={speakerName}
                onChange={(event) => setSpeakerName(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Language</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Start minutes</span>
              <input
                type="number"
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={startMinutes}
                onChange={(event) => setStartMinutes(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Duration minutes</span>
              <input
                type="number"
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={durationMinutes}
                onChange={(event) => setDurationMinutes(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Run name</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={runName}
                onChange={(event) => setRunName(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Output runs dir</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={outputRunsDir}
                onChange={(event) => setOutputRunsDir(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Batch size</span>
              <input
                type="number"
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Epochs</span>
              <input
                type="number"
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={epochs}
                onChange={(event) => setEpochs(event.target.value)}
              />
            </label>

            <label className="grid gap-2 sm:col-span-2">
              <span className="text-sm font-medium">Output dataset dir (optional)</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={datasetDir}
                onChange={(event) => setDatasetDir(event.target.value)}
                placeholder="Leave empty to auto-name from video id"
              />
            </label>

            <div className="flex flex-wrap gap-3 sm:col-span-2">
              <button
                className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                onClick={startFinetune}
                disabled={starting || status?.running}
              >
                {starting ? "Starting..." : "Start Fine-tune"}
              </button>

              <button
                className="rounded bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                onClick={stopFinetune}
                disabled={stopping || !status?.running}
              >
                {stopping ? "Stopping..." : "Stop"}
              </button>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-zinc-300 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold">Process status</h2>
          <div className="mt-2 text-sm text-zinc-700">
            <p>Running: {status?.running ? "Yes" : "No"}</p>
            <p>Stage: {status?.stage ?? "-"}</p>
            <p>Return code: {status?.return_code ?? "-"}</p>
            <p>Started at: {status?.started_at ?? "-"}</p>
            <p>Finished at: {status?.finished_at ?? "-"}</p>
            <p className="break-all">Command: {status?.command?.join(" ") ?? "-"}</p>
          </div>
        </section>

        <section className="rounded-xl border border-zinc-300 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold">Live logs (SSE)</h2>
          <pre className="mt-3 max-h-[420px] overflow-auto rounded bg-zinc-900 p-4 text-xs text-zinc-100">
            {logs || "Waiting for logs..."}
            <div ref={logEndRef} />
          </pre>
        </section>
      </main>
    </div>
  );
}
