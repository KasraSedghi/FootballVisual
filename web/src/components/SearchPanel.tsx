"use client";

/**
 * Ask the clip a question and jump to the moments that answer it.
 *
 * The search runs entirely in the browser against measurements the engine
 * computes. The only thing the network is used for is translating the question
 * into filters, and that step degrades to heuristics when there is no API key,
 * so the feature works offline.
 *
 * Each result shows the numbers that qualified it rather than a description of
 * them, because the claim being made is that the passage provably has the
 * property asked for, and a number is how that gets checked.
 */

import { useCallback, useState } from "react";

import {
  describeQuery,
  groupIntoSequences,
  searchFrames,
  type SearchSequence,
  type TacticalQuery,
} from "@/lib/tactics/search";
import { formatTime, type LoadedSession } from "@/lib/session";

interface Props {
  session: LoadedSession | null;
  onSeek: (index: number) => void;
}

const EXAMPLES = [
  "a big gap in the last line",
  "players between the lines with an open lane",
  "a compact block with no way through",
  "someone in behind the defence",
];

export default function SearchPanel({ session, onSeek }: Props) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchSequence[] | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);

  const run = useCallback(
    async (text: string) => {
      if (!session || !text.trim()) return;
      setBusy(true);
      setError(null);

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null)) as
            | { error?: string }
            | null;
          throw new Error(detail?.error ?? `search failed (${response.status})`);
        }

        const { query, source: from } = (await response.json()) as {
          query: TacticalQuery;
          source: string;
        };

        const hits = searchFrames(session.snapshots, query);
        const sequences = groupIntoSequences(hits);

        setResults(sequences);
        setSummary(describeQuery(query));
        setSource(from);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setResults(null);
      } finally {
        setBusy(false);
      }
    },
    [session],
  );

  return (
    <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Find a moment
      </h3>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void run(question);
        }}
        className="flex gap-1.5"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="when was there a gap to play into?"
          disabled={!session || busy}
          className="min-w-0 flex-1 rounded border border-white/10 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-600 focus:border-emerald-500/50 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!session || busy || !question.trim()}
          className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-40"
        >
          {busy ? "…" : "Search"}
        </button>
      </form>

      {!results && !error && (
        <div className="mt-2 flex flex-wrap gap-1">
          {EXAMPLES.map((e) => (
            <button
              key={e}
              type="button"
              onClick={() => {
                setQuestion(e);
                void run(e);
              }}
              disabled={!session || busy}
              className="rounded border border-white/10 bg-slate-800/60 px-2 py-0.5 text-[10px] text-slate-400 transition hover:bg-slate-700 disabled:opacity-40"
            >
              {e}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="mt-2 rounded border border-red-500/30 bg-red-950/40 p-2 text-[11px] text-red-300">
          {error}
        </p>
      )}

      {summary && (
        <p className="mt-2.5 text-[11px] leading-relaxed text-slate-500">
          Searched for{" "}
          <span className="text-slate-400">{summary}</span>
          {source === "heuristic" || source === "heuristic-fallback" ? (
            <span className="text-slate-600">
              {" "}
              (matched without a model, so set ANTHROPIC_API_KEY for better
              phrasing support)
            </span>
          ) : null}
        </p>
      )}

      {results && results.length === 0 && (
        <p className="mt-2 text-[11px] text-slate-400">
          No passage in this clip satisfies that. Try loosening it.
        </p>
      )}

      {results && results.length > 0 && (
        <ul className="mt-2 space-y-1">
          {results.slice(0, 6).map((s) => (
            <li key={s.startFrame}>
              <button
                type="button"
                onClick={() => onSeek(s.peak.index)}
                className="w-full rounded border border-white/10 bg-slate-800/50 px-2.5 py-1.5 text-left text-[11px] transition hover:bg-slate-700/60"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-slate-200">
                    {formatTime(s.startTimeS)} to {formatTime(s.endTimeS)}
                  </span>
                  <span className="text-slate-500">
                    {`${s.frameCount} frame${s.frameCount === 1 ? "" : "s"}`}
                  </span>
                </div>
                <div className="mt-0.5 flex flex-wrap gap-x-3 text-[10px] text-slate-400">
                  <span>
                    {s.peak.measurements.openLanes} open lane
                    {s.peak.measurements.openLanes === 1 ? "" : "s"}
                  </span>
                  {s.peak.measurements.backLineGapM != null && (
                    <span>gap {s.peak.measurements.backLineGapM.toFixed(0)}m</span>
                  )}
                  {s.peak.measurements.blockWidthM != null && (
                    <span>width {s.peak.measurements.blockWidthM.toFixed(0)}m</span>
                  )}
                  {s.peak.measurements.playersBetweenLines > 0 && (
                    <span className="text-emerald-400">
                      {s.peak.measurements.playersBetweenLines} between lines
                    </span>
                  )}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
