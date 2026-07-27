"use client";

/**
 * Load a clip and its tracking data from the analyst's own disk.
 *
 * What this deliberately does *not* do is run the pipeline. Detection and
 * tracking are seconds of CPU per frame and need model weights, so they belong
 * in the Python pipeline, not in a browser tab. What this does is close the
 * loop after that: run the pipeline once on a clip, then open the result here
 * without copying files into `public/data` and restarting the dev server.
 *
 * Both files are held as object URLs, so nothing is uploaded anywhere and the
 * video never leaves the machine. That matters for real footage, which is
 * usually licensed and not ours to send to a server.
 */

import { useCallback, useRef, useState } from "react";

import { loadTracksFromFile, type LoadedSession } from "@/lib/session";

interface Props {
  onLoad: (session: LoadedSession, videoUrl: string | null, name: string) => void;
  currentName: string | null;
}

export default function ClipLoader({ onLoad, currentName }: Props) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const videoUrlRef = useRef<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      setError(null);
      setBusy(true);

      try {
        const list = Array.from(files);
        const tracksFile = list.find((f) => f.name.endsWith(".json"));
        const videoFile = list.find((f) => f.type.startsWith("video/"));

        if (!tracksFile) {
          throw new Error(
            "Pick the tracks.json as well as the clip. Without it there is " +
              "nothing to draw on the map, only a video.",
          );
        }

        const session = await loadTracksFromFile(tracksFile);

        // Release the previous clip's blob before replacing it, or every load
        // leaks a whole video into memory for the life of the tab.
        if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
        videoUrlRef.current = videoFile ? URL.createObjectURL(videoFile) : null;

        onLoad(session, videoUrlRef.current, videoFile?.name ?? tracksFile.name);
        setOpen(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [onLoad],
  );

  return (
    <div className="text-right">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="rounded border border-white/15 bg-slate-800/80 px-2.5 py-1 text-[11px] text-slate-300 transition hover:bg-slate-700"
      >
        {open ? "Close" : "Open a clip"}
      </button>

      {open && (
        <div className="mt-2 w-[340px] rounded-lg border border-white/10 bg-slate-900 p-3 text-left shadow-xl">
          <p className="mb-2 text-[11px] leading-relaxed text-slate-400">
            Select a clip and the <code className="text-slate-300">tracks.json</code>{" "}
            the pipeline produced for it. Both stay on this machine.
          </p>

          <input
            type="file"
            multiple
            accept="video/*,application/json,.json"
            disabled={busy}
            onChange={(e) => void handleFiles(e.target.files)}
            className="block w-full text-[11px] text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-slate-700 file:px-2.5 file:py-1 file:text-[11px] file:text-slate-200 hover:file:bg-slate-600"
          />

          {busy && <p className="mt-2 text-[11px] text-slate-400">Reading…</p>}

          {error && (
            <p className="mt-2 rounded border border-red-500/30 bg-red-950/40 p-2 text-[11px] leading-relaxed text-red-300">
              {error}
            </p>
          )}

          <details className="mt-2.5 text-[11px] text-slate-500">
            <summary className="cursor-pointer text-slate-400">
              How do I make a tracks.json?
            </summary>
            <pre className="mt-1.5 overflow-x-auto rounded bg-slate-950 p-2 text-[10px] leading-relaxed text-slate-400">
              {`python -m footballvisual track \\
  --video your_clip.mp4 \\
  --out tracks.json \\
  --auto-calibrate \\
  --camera-side minus_y \\
  --left-goal-side left`}
            </pre>
          </details>

          {currentName && (
            <p className="mt-2 truncate text-[11px] text-slate-500">
              Showing: <span className="text-slate-400">{currentName}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
