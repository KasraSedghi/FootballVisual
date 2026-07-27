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
  /**
   * Show every measurement rather than just the passing options.
   *
   * The default is off. This panel can render four sections of numbers at once,
   * and on a first look that reads as a wall rather than an answer. The
   * question the tool exists to answer is which lane is open, so that section
   * stands alone until the analyst asks for the rest.
   */
  detailed?: boolean;
}

/**
 * How many passing options to list before collapsing the rest.
 *
 * A carrier can have a dozen team mates, and ranking them is the point, so the
 * tail is nearly always noise. Three is what fits in a glance.
 */
const FOCUS_LANES = 3;

/*
 * The same tokens the map draws each lane with. These previously named their
 * own colours, and had drifted: a lane drawn green-500 on the pitch was listed
 * as emerald-400 here, in the panel whose whole job is to be the audit trail
 * for what the map shows.
 */
const VERDICT_STYLE: Record<string, string> = {
  open: "text-verdict-open border-verdict-open/40 bg-verdict-open/10",
  contested: "text-verdict-contested border-verdict-contested/40 bg-verdict-contested/10",
  blocked: "text-verdict-blocked border-verdict-blocked/40 bg-verdict-blocked/10",
};

export default function TacticsPanel({
  report,
  selectedLaneId,
  onSelectLane,
  detailed = false,
}: Props) {
  if (!report) {
    return (
      <div className="rounded-lg border border-white/10 bg-slate-900/50 p-4 text-sm text-slate-400">
        Load a frame to see the analysis.
      </div>
    );
  }

  const { block, lanes, space } = report;
  const shownLanes = detailed ? lanes.slice(0, 7) : lanes.slice(0, FOCUS_LANES);
  const hiddenLanes = lanes.length - shownLanes.length;

  /*
   * A fragment, not a wrapping div. The parent lays these sections out: it
   * stacks them in focus view and flows them into a two column grid in detail
   * view, and it can only do the second if each section is a direct child.
   */
  return (
    <>
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
            {shownLanes.map((lane) => {
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
        {/*
          The sentence below is built as one string rather than interleaved
          with JSX expressions. The interleaved form rendered as
          "optionshidden": JSX drops the whitespace between an expression and
          the text following it when the line wraps, which is invisible in the
          source and obvious on screen.
        */}
        {hiddenLanes > 0 && (
          <p className="mt-2 text-[11px] text-slate-500">
            {`${hiddenLanes} lower ranked option${hiddenLanes > 1 ? "s" : ""} hidden. ` +
              `Use "All measurements" to see them.`}
          </p>
        )}
      </section>

      {detailed && block && (
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

      {detailed && space && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Space control
          </h3>
          <div className="flex h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="bg-team-a"
              style={{ width: `${(space.teamAShare * 100).toFixed(1)}%` }}
            />
            <div
              className="bg-team-b"
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

      {detailed &&
        (report.playersBetweenLines.length > 0 ||
          report.playersBeyondLine.length > 0) && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4 text-xs">
          {report.playersBetweenLines.length > 0 && (
            <p className="text-verdict-open">
              Between the lines: {report.playersBetweenLines.map((id) => `#${id}`).join(", ")}
            </p>
          )}
          {report.playersBeyondLine.length > 0 && (
            <p className="mt-1 text-marker-offside">
              Beyond the last line: {report.playersBeyondLine.map((id) => `#${id}`).join(", ")}
            </p>
          )}
        </section>
      )}
    </>
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
