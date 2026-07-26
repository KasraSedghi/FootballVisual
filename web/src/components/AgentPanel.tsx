"use client";

/**
 * The analysis panel: asks a question about the frozen frame and shows the answer.
 *
 * The badge showing whether the answer came from Claude or the deterministic
 * template is deliberately always visible. A tool that quietly swaps a computed
 * answer for a generated one, or the reverse, teaches its user to trust the
 * wrong thing.
 */

import { useCallback, useState } from "react";

import { reportFacts, type TacticalReport } from "@/lib/tactics";

interface Props {
  report: TacticalReport | null;
  edited: boolean;
}

interface Answer {
  headline: string;
  analysis: string;
  recommendedTarget: string | null;
  source: string;
}

const PRESETS = [
  "Analyse this defensive block. Where is the open passing lane?",
  "What is the biggest weakness in this shape right now?",
  "If I am the ball carrier, what is my safest progressive option?",
];

export default function AgentPanel({ report, edited }: Props) {
  const [question, setQuestion] = useState(PRESETS[0]);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = useCallback(
    async (q: string) => {
      if (!report) return;
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/analyse", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ facts: reportFacts(report), question: q, edited }),
        });
        if (!response.ok) throw new Error(`request failed (${response.status})`);
        setAnswer((await response.json()) as Answer);
      } catch (err) {
        setError(err instanceof Error ? err.message : "analysis failed");
      } finally {
        setLoading(false);
      }
    },
    [report, edited],
  );

  return (
    <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Ask the analyst
      </h3>

      <div className="mb-2 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => {
              setQuestion(p);
              void ask(p);
            }}
            className="rounded border border-white/10 bg-slate-800/60 px-2 py-1 text-[11px] text-slate-300 transition hover:bg-slate-700"
          >
            {p.length > 34 ? `${p.slice(0, 32)}…` : p}
          </button>
        ))}
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={2}
        className="w-full resize-none rounded border border-white/10 bg-slate-950/70 p-2 text-xs text-slate-200 outline-none focus:border-emerald-500/50"
      />

      <button
        type="button"
        onClick={() => void ask(question)}
        disabled={loading || !report}
        className="mt-2 w-full rounded bg-emerald-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-700"
      >
        {loading ? "Analysing…" : "Analyse frame"}
      </button>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {answer && (
        <div className="mt-3 rounded border border-white/10 bg-slate-950/60 p-3">
          <div className="mb-1.5 flex items-start justify-between gap-2">
            <p className="text-sm font-semibold text-emerald-400">{answer.headline}</p>
            <span
              className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-400"
              title={
                answer.source === "claude"
                  ? "Narrated by Claude from the computed measurements"
                  : "Written by the deterministic template from the same measurements"
              }
            >
              {answer.source === "claude" ? "claude" : "computed"}
            </span>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">{answer.analysis}</p>
        </div>
      )}

      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        Every number in the answer is computed by the geometry engine. The model only
        phrases them, so with no API key configured the analysis is still correct, just
        less fluent.
      </p>
    </section>
  );
}
