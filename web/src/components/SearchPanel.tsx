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
import {
  embedFrame,
  findSimilar,
  type FrameEmbedding,
  type SimilarFrame,
} from "@/lib/tactics/embedding";
import { analyseSnapshot, type TacticalReport } from "@/lib/tactics";
import { formatTime, type LoadedSession } from "@/lib/session";

interface Props {
  session: LoadedSession | null;
  onSeek: (index: number) => void;
  /** The frame currently on screen, used as the query for "like this". */
  currentIndex: number;
}

/**
 * The measurements shown beside a similarity score.
 *
 * A cosine similarity on its own is unfalsifiable to the person reading it. The
 * rest of this app shows the number behind every claim, so a match has to be
 * checkable the same way: if a frame is returned as similar, the analyst should
 * be able to see that its shape numbers really are close to the query's.
 */
interface FrameFacts {
  openLanes: number;
  gapM: number | null;
  widthM: number | null;
  betweenLines: number;
}

function factsOf(report: TacticalReport): FrameFacts {
  return {
    openLanes: report.lanes.filter((l) => l.verdict === "open").length,
    gapM: report.block?.largestBackLineGapM ?? null,
    widthM: report.block?.widthM ?? null,
    betweenLines: report.playersBetweenLines.length,
  };
}

function describeFacts(f: FrameFacts): string {
  const parts = [`${f.openLanes} open`];
  if (f.gapM != null) parts.push(`gap ${f.gapM.toFixed(0)}m`);
  if (f.widthM != null) parts.push(`width ${f.widthM.toFixed(0)}m`);
  if (f.betweenLines > 0) parts.push(`${f.betweenLines} between lines`);
  return parts.join(" · ");
}

const EXAMPLES = [
  "a big gap in the last line",
  "players between the lines with an open lane",
  "a compact block with no way through",
  "someone in behind the defence",
];

export default function SearchPanel({ session, onSeek, currentIndex }: Props) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchSequence[] | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [similar, setSimilar] = useState<SimilarFrame[] | null>(null);
  const [facts, setFacts] = useState<Map<number, FrameFacts>>(new Map());
  const [queryFacts, setQueryFacts] = useState<FrameFacts | null>(null);

  /**
   * Embed every frame once, lazily, and keep it.
   *
   * A clip is a few hundred frames and each embedding needs a full tactical
   * report, so this is the expensive part. Doing it on first use rather than on
   * load keeps the page responsive for the majority of visitors who never ask
   * for a similarity search.
   */
  const findLikeThis = useCallback(() => {
    if (!session) return;
    setError(null);
    setResults(null);
    setSummary(null);

    const embeddings: FrameEmbedding[] = [];
    const measured = new Map<number, FrameFacts>();
    for (let i = 0; i < session.snapshots.length; i += 1) {
      const report = analyseSnapshot(session.snapshots[i], {
        previous: i > 0 ? session.snapshots[i - 1] : null,
        computeSpace: false,
      });
      if (!report) continue;
      embeddings.push({
        frame: session.snapshots[i].frame,
        timeS: session.snapshots[i].timeS,
        index: i,
        vector: embedFrame(report),
      });
      measured.set(i, factsOf(report));
    }

    const here = embeddings.find((e) => e.index === currentIndex);
    if (!here) {
      setError("This frame could not be analysed, so there is nothing to match on.");
      return;
    }

    setFacts(measured);
    setQueryFacts(measured.get(currentIndex) ?? null);
    setSimilar(findSimilar(here.vector, embeddings, { excludeIndex: currentIndex }));
  }, [session, currentIndex]);

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
        setSimilar(null);
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

      <div className="mt-2 border-b border-white/5 pb-2">
        <button
          type="button"
          onClick={findLikeThis}
          disabled={!session}
          className="w-full rounded border border-sky-500/40 bg-sky-500/10 px-2.5 py-1.5 text-[11px] text-sky-300 transition hover:bg-sky-500/20 disabled:opacity-40"
        >
          Moments like this frame
        </button>
        <p className="mt-1 text-[10px] leading-tight text-slate-600">
          Shape match rather than a threshold, with mirrored ends and wings
          treated as the same situation.
        </p>
      </div>

      {!results && !error && !similar && (
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

      {similar && (
        <div className="mt-2.5">
          <p className="text-[11px] leading-relaxed text-slate-500">
            Frames whose shape most resembles this one. Matching runs on a
            canonicalised pitch, so the same situation at the other end or down
            the other wing still counts.
          </p>
          {queryFacts && (
            <p className="mt-1.5 rounded border border-white/5 bg-slate-950/60 px-2 py-1 text-[10px] text-slate-500">
              This frame:{" "}
              <span className="text-slate-300">{describeFacts(queryFacts)}</span>
            </p>
          )}
          {similar.length === 0 ? (
            <p className="mt-1.5 text-[11px] text-slate-400">
              Nothing else in this clip is far enough away in time to compare
              against.
            </p>
          ) : (
            <ul className="mt-1.5 space-y-1">
              {similar.map((m) => (
                <li key={m.frame}>
                  <button
                    type="button"
                    onClick={() => onSeek(m.index)}
                    className="w-full rounded border border-white/10 bg-slate-800/50 px-2.5 py-1.5 text-left text-[11px] transition hover:bg-slate-700/60"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-slate-200">
                        {formatTime(m.timeS)}
                      </span>
                      {/*
                        A number, not a bar. These vectors have no negative
                        components, so cosine similarity between any two frames
                        of football sits well above zero and a bar scaled over
                        [0, 1] renders every result nearly full. Three decimal
                        places separate them; a bar hides that they differ.
                      */}
                      <span className="font-mono text-sky-300">
                        {m.similarity.toFixed(3)}
                      </span>
                    </div>
                    {facts.has(m.index) && (
                      <div className="mt-0.5 text-[10px] text-slate-400">
                        {describeFacts(facts.get(m.index)!)}
                      </div>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
