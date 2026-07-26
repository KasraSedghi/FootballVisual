"use client";

/**
 * The measurements panel.
 *
 * Every number shown here is computed by the deterministic engine, so this
 * panel is the audit trail for whatever the analysis says. If the narration
 * claims a lane is open, the margin that justifies it is visible one panel
 * over.
 */

import type { TacticalReport } from "@/lib/tactics";
import { teamLabel } from "@/lib/session";

interface Props {
  report: TacticalReport | null;
  selectedLaneId: number | null;
  onSelectLane: (id: number | null) => void;
}

const VERDICT_STYLE: Record<string, string> = {
  open: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10",
  contested: "text-amber-400 border-amber-500/40 bg-amber-500/10",
  blocked: "text-slate-400 border-slate-600/40 bg-slate-600/10",
};

export default function TacticsPanel({ report, selectedLaneId, onSelectLane }: Props) {
  if (!report) {
    return (
      <div className="rounded-lg border border-white/10 bg-slate-900/50 p-4 text-sm text-slate-400">
        Load a frame to see the analysis.
      </div>
    );
  }

  const { block, lanes, space } = report;

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Passing options
        </h3>
        {report.carrierId == null ? (
          <p className="text-sm text-slate-400">
            No player is close enough to the ball to be carrying it in this frame.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {lanes.slice(0, 7).map((lane) => {
              const isSelected = selectedLaneId === lane.targetId;
              return (
                <li key={lane.targetId}>
                  <button
                    type="button"
                    onClick={() => onSelectLane(isSelected ? null : lane.targetId)}
                    className={`w-full rounded border px-2.5 py-2 text-left text-xs transition ${
                      VERDICT_STYLE[lane.verdict]
                    } ${isSelected ? "ring-1 ring-white/40" : "hover:brightness-125"}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-slate-100">
                        &rarr; #{lane.targetId}
                      </span>
                      <span className="uppercase tracking-wide">{lane.verdict}</span>
                    </div>
                    <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] text-slate-400">
                      <span>
                        margin{" "}
                        <span className="text-slate-200">
                          {lane.safetyMarginS === Infinity
                            ? "free"
                            : `${lane.safetyMarginS.toFixed(2)}s`}
                        </span>
                      </span>
                      <span>
                        length <span className="text-slate-200">{lane.distanceM.toFixed(0)}m</span>
                      </span>
                      <span>
                        gains{" "}
                        <span className="text-slate-200">{lane.progressionM.toFixed(0)}m</span>
                      </span>
                      <span>
                        bypasses{" "}
                        <span className="text-slate-200">{lane.defendersBypassed}</span>
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {block && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            {teamLabel(report.defendingTeam)} block
          </h3>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
            <Metric label="Players" value={String(block.playerCount)} />
            <Metric label="Hull area" value={`${Math.round(block.hullAreaM2)} m²`} />
            <Metric label="Width" value={`${block.widthM.toFixed(0)} m`} />
            <Metric label="Depth" value={`${block.blockDepthM.toFixed(0)} m`} />
            <Metric label="Back line" value={`x = ${block.defensiveLineX.toFixed(0)}`} />
            <Metric
              label="Largest gap"
              value={`${block.largestBackLineGapM.toFixed(1)} m`}
              highlight={block.largestBackLineGapM > 12}
            />
          </dl>
        </section>
      )}

      {space && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Space control
          </h3>
          <div className="flex h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="bg-blue-500"
              style={{ width: `${(space.teamAShare * 100).toFixed(1)}%` }}
            />
            <div
              className="bg-red-500"
              style={{ width: `${(space.teamBShare * 100).toFixed(1)}%` }}
            />
          </div>
          <div className="mt-1.5 flex justify-between text-[11px] text-slate-400">
            <span>Team A {(space.teamAShare * 100).toFixed(0)}%</span>
            <span>Team B {(space.teamBShare * 100).toFixed(0)}%</span>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Share of the pitch each team reaches first, from a time to arrive model
            including current momentum, not a plain nearest player split.
          </p>
        </section>
      )}

      {(report.playersBetweenLines.length > 0 || report.playersBeyondLine.length > 0) && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4 text-xs">
          {report.playersBetweenLines.length > 0 && (
            <p className="text-emerald-400">
              Between the lines: {report.playersBetweenLines.map((id) => `#${id}`).join(", ")}
            </p>
          )}
          {report.playersBeyondLine.length > 0 && (
            <p className="mt-1 text-amber-400">
              Beyond the last line: {report.playersBeyondLine.map((id) => `#${id}`).join(", ")}
            </p>
          )}
        </section>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className={highlight ? "font-semibold text-amber-400" : "text-slate-200"}>{value}</dd>
    </div>
  );
}
