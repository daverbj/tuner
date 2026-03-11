"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

type ModelEntry = {
  model_id: string;
  run_name: string;
  training_run: string;
  checkpoint: string;
  config: string;
  vocab: string;
  modified_at: string;
};

type SpeakerRef = {
  path: string;
  label: string;
  preview_url: string;
};

export default function InferencePage() {
  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000", []);

  const [models, setModels] = useState<ModelEntry[]>([]);
  const [speakerRefs, setSpeakerRefs] = useState<SpeakerRef[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [speakerWav, setSpeakerWav] = useState("");
  const [language, setLanguage] = useState("en");
  const [text, setText] = useState("Hello, this is my fine-tuned XTTS voice.");
  const [generating, setGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState("");
  const [error, setError] = useState("");

  const refreshData = useCallback(async () => {
    setError("");
    const [modelsResp, speakersResp] = await Promise.all([
      fetch(`${apiBase}/api/inference/models`, { cache: "no-store" }),
      fetch(`${apiBase}/api/inference/speakers`, { cache: "no-store" }),
    ]);

    if (!modelsResp.ok) {
      throw new Error(`Failed to load models: ${await modelsResp.text()}`);
    }
    if (!speakersResp.ok) {
      throw new Error(`Failed to load speaker refs: ${await speakersResp.text()}`);
    }

    const modelsJson = (await modelsResp.json()) as { items: ModelEntry[] };
    const speakersJson = (await speakersResp.json()) as { items: SpeakerRef[] };

    setModels(modelsJson.items);
    setSpeakerRefs(speakersJson.items);

    setSelectedModel((current) => current || modelsJson.items[0]?.model_id || "");
    setSpeakerWav((current) => current || speakersJson.items[0]?.path || "");
  }, [apiBase]);

  useEffect(() => {
    refreshData().catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    });
  }, [refreshData]);

  const selectedSpeaker = speakerRefs.find((item) => item.path === speakerWav);
  const selectedSpeakerPreviewUrl = selectedSpeaker ? `${apiBase}${selectedSpeaker.preview_url}` : "";

  const onGenerate = async () => {
    setGenerating(true);
    setError("");
    setAudioUrl("");
    try {
      const response = await fetch(`${apiBase}/api/inference/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: selectedModel,
          speaker_wav: speakerWav,
          text,
          language,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const json = (await response.json()) as { audio_url: string };
      setAudioUrl(`${apiBase}${json.audio_url}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-100 p-6 text-zinc-900">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-6">
        <section className="rounded-xl border border-zinc-300 bg-white p-5 shadow-sm">
          <h1 className="text-2xl font-semibold">XTTS Inference</h1>
          <p className="mt-2 text-sm text-zinc-600">Select a fine-tuned model, enter text, and generate speech.</p>
          <p className="mt-2 text-sm">
            <Link href="/" className="text-blue-600 underline">
              Back to Fine-Tune Page
            </Link>
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <label className="grid gap-2 sm:col-span-2">
              <span className="text-sm font-medium">Fine-tuned model</span>
              <select
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                <option value="">Select model...</option>
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.run_name} / {model.training_run} / {model.checkpoint.split("/").pop()}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 sm:col-span-2">
              <span className="text-sm font-medium">Speaker reference wav</span>
              <select
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={speakerWav}
                onChange={(event) => setSpeakerWav(event.target.value)}
              >
                <option value="">Select speaker reference...</option>
                {speakerRefs.map((ref) => (
                  <option key={ref.path} value={ref.path}>
                    {ref.label} — {ref.path}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-medium">Language</span>
              <input
                className="rounded border border-zinc-300 px-3 py-2 text-sm"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
              />
            </label>

            <div className="hidden sm:block" />

            <label className="grid gap-2 sm:col-span-2">
              <span className="text-sm font-medium">Text</span>
              <textarea
                className="min-h-32 rounded border border-zinc-300 px-3 py-2 text-sm"
                value={text}
                onChange={(event) => setText(event.target.value)}
              />
            </label>

            <div className="flex flex-wrap gap-3 sm:col-span-2">
              <button
                className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                onClick={onGenerate}
                disabled={generating || !selectedModel || !speakerWav || !text.trim()}
              >
                {generating ? "Generating..." : "Generate Speech"}
              </button>
              <button
                className="rounded bg-zinc-800 px-4 py-2 text-sm font-medium text-white"
                onClick={() => {
                  refreshData().catch((err: unknown) => {
                    const message = err instanceof Error ? err.message : String(err);
                    setError(message);
                  });
                }}
              >
                Refresh Models
              </button>
            </div>

            <div className="sm:col-span-2">
              <span className="mb-2 block text-sm font-medium">Reference preview</span>
              {selectedSpeakerPreviewUrl ? (
                <audio controls src={selectedSpeakerPreviewUrl} className="w-full" />
              ) : (
                <p className="text-sm text-zinc-600">No speaker reference selected.</p>
              )}
            </div>
          </div>

          {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
        </section>

        <section className="rounded-xl border border-zinc-300 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold">Output</h2>
          {audioUrl ? (
            <div className="mt-3 flex flex-col gap-3">
              <audio controls src={audioUrl} className="w-full" />
              <a className="text-sm text-blue-600 underline" href={audioUrl} target="_blank" rel="noreferrer">
                Open generated WAV
              </a>
            </div>
          ) : (
            <p className="mt-3 text-sm text-zinc-600">No generated audio yet.</p>
          )}
        </section>
      </main>
    </div>
  );
}
