"use client";

/**
 * The measurements panel.
 *
 * Every number shown here is computed by the deterministic engine, so this
 * panel is the audit trail for whatever the analysis says. If the narration
 * claims a lane is open, the margin that justifies it is visible one panel
 * over.
 */

import { useState } from "react";

import {
  BLOCKED_MARGIN_S,
  COMPLETION_MODEL,
  OPEN_MARGIN_S,
  XT_MODEL,
  type TacticalReport,
} from "@/lib/tactics";
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

/**
 * What each number on a lane row means.
 *
 * Every one of these is a term of art, and the panel showed six of them with no
 * way to find out what any of them measured. "bypasses 3" is either the most
 * useful number on the row or noise, depending entirely on whether you know it
 * counts defenders the pass takes out of the game.
 *
 * Written as an ordinary explanation rather than a formula, because the person
 * who needs it is the one who does not already know. The thresholds come from
 * the engine rather than being retyped, so the legend cannot end up confidently
 * describing a rule the code no longer follows.
 */
const LANE_TERMS: { term: string; meaning: string }[] = [
  {
    term: "verdict",
    meaning:
      `whether the pass survives the race. Blocked below ${BLOCKED_MARGIN_S.toFixed(2)}s, ` +
      `open above ${OPEN_MARGIN_S.toFixed(2)}s, contested between.`,
  },
  {
    term: "margin",
    meaning:
      "seconds the ball beats the best placed defender to the most dangerous " +
      "point on its path. Negative means that defender gets there first, so " +
      "the pass can be cut out.",
  },
  {
    term: "length",
    meaning: "how far the ball actually travels, in metres.",
  },
  {
    term: "gains",
    meaning:
      "ground gained toward the goal being attacked. Negative is a pass that " +
      "goes backwards, which is a normal and often correct thing to do.",
  },
  {
    term: "bypasses",
    meaning:
      "defenders the ball ends up behind, so how many the pass takes out of " +
      "the game. This is what separates a sideways ball from a line breaking one.",
  },
];

/**
 * Lateral gap in the last line above which the number is called out.
 *
 * Roughly the width a runner can attack before either centre back closes it.
 * Named rather than inline so the legend below states the same threshold the
 * highlight uses.
 */
const WIDE_GAP_M = 12;

const BLOCK_TERMS: { term: string; meaning: string }[] = [
  {
    term: "players",
    meaning:
      "how many outfielders make up the block. Keepers are excluded, because " +
      "one standing 40m behind the line would stretch every other number here.",
  },
  {
    term: "hull area",
    meaning:
      "ground enclosed by stretching a band around the outermost defenders. " +
      "Small is compact, large is a team spread thin.",
  },
  {
    term: "width",
    meaning: "how far the block spans across the pitch, touchline to touchline.",
  },
  {
    term: "depth",
    meaning:
      "distance from the deepest defender to the highest one. A deep number " +
      "means the lines are strung out, which is where space between them comes from.",
  },
  {
    term: "back line",
    meaning:
      "where the last line sits, as a pitch coordinate. The origin is the " +
      "centre spot, so negative is inside their own half and positive is up the pitch.",
  },
  {
    term: "largest gap",
    meaning:
      `the biggest sideways space between two adjacent defenders in the last ` +
      `line, which is where a run gets played through. Called out above ${WIDE_GAP_M}m.`,
  },
];

const VALUE_TERMS: { term: string; meaning: string }[] = [
  {
    term: "if it lands",
    meaning:
      "how dangerous the target position is, times the chance the pass gets " +
      "there. The first number is Expected Threat, the probability a " +
      "possession from there ends in a goal.",
  },
  {
    term: "xT",
    meaning:
      "the two multiplied, less the value of where the ball already is. It is " +
      "what the pass is worth. Negative is common and correct: a safe square " +
      "ball keeps possession and gives up the position it started from.",
  },
];

function Legend({ heading, terms }: { heading: string; terms: typeof LANE_TERMS }) {
  return (
    <div>
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
        {heading}
      </p>
      <dl className="space-y-1.5">
        {terms.map(({ term, meaning }) => (
          <div key={term}>
            <dt className="inline font-semibold text-slate-300">{term}</dt>
            <dd className="inline text-slate-400"> {meaning}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function TacticsPanel({
  report,
  selectedLaneId,
  onSelectLane,
  detailed = false,
}: Props) {
  const [isLegendOpen, setIsLegendOpen] = useState(false);
  const [isBlockLegendOpen, setIsBlockLegendOpen] = useState(false);

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
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Passing options
          </h3>
          <button
            type="button"
            onClick={() => setIsLegendOpen((open) => !open)}
            aria-expanded={isLegendOpen}
            className="rounded border border-white/10 px-1.5 py-0.5 text-[10px] text-slate-400 transition hover:border-white/25 hover:text-slate-200"
          >
            {isLegendOpen ? "hide" : "what do these mean?"}
          </button>
        </div>

        {isLegendOpen && (
          <div className="mb-3 space-y-3 rounded border border-white/10 bg-slate-950/60 p-3 text-[11px] leading-relaxed">
            <Legend heading="Geometry" terms={LANE_TERMS} />
            {lanes.some((lane) => lane.value) && (
              <Legend heading="Value" terms={VALUE_TERMS} />
            )}
            <p className="border-t border-white/10 pt-2 text-[10px] leading-relaxed text-slate-500">
              {`Every number here is computed from the frame, never estimated by a ` +
                `model. The threat grid is counted off ${XT_MODEL.matches} matches and the ` +
                `completion rate is fitted to ${COMPLETION_MODEL.passes.toLocaleString()} real passes ` +
                `whose outcomes are known.`}
            </p>
          </div>
        )}

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
                    {lane.value && (
                      /*
                       * Reward, risk and the product, in that order, because
                       * that is the order the trade is read in. Showing only
                       * the expected value would hide whether a low number
                       * means the pass gains nothing or means it probably does
                       * not arrive, and those call for opposite decisions.
                       */
                      <div className="mt-1.5 flex items-center justify-between gap-2 border-t border-white/10 pt-1.5 text-[11px]">
                        <span className="text-slate-500">
                          if it lands{" "}
                          <span className="text-slate-300">
                            {lane.value.toXT.toFixed(4)}
                          </span>
                          <span className="px-1 text-slate-600">x</span>
                          <span className="text-slate-300">
                            {(lane.value.completion * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span
                          className={
                            lane.value.expectedXT >= 0
                              ? "font-semibold text-verdict-open"
                              : "font-semibold text-slate-500"
                          }
                        >
                          {lane.value.expectedXT >= 0 ? "+" : ""}
                          {lane.value.expectedXT.toFixed(4)} xT
                        </span>
                      </div>
                    )}
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
          <div className="mb-3 flex items-center justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              {teamLabel(report.defendingTeam)} block
            </h3>
            <button
              type="button"
              onClick={() => setIsBlockLegendOpen((open) => !open)}
              aria-expanded={isBlockLegendOpen}
              className="rounded border border-white/10 px-1.5 py-0.5 text-[10px] text-slate-400 transition hover:border-white/25 hover:text-slate-200"
            >
              {isBlockLegendOpen ? "hide" : "what do these mean?"}
            </button>
          </div>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
            <Metric label="Players" value={String(block.playerCount)} />
            <Metric label="Hull area" value={`${Math.round(block.hullAreaM2)} m²`} />
            <Metric label="Width" value={`${block.widthM.toFixed(0)} m`} />
            <Metric label="Depth" value={`${block.blockDepthM.toFixed(0)} m`} />
            <Metric label="Back line" value={`x = ${block.defensiveLineX.toFixed(0)}`} />
            <Metric
              label="Largest gap"
              value={`${block.largestBackLineGapM.toFixed(1)} m`}
              highlight={block.largestBackLineGapM > WIDE_GAP_M}
            />
          </dl>
          {isBlockLegendOpen && (
            <div className="mt-3 rounded border border-white/10 bg-slate-950/60 p-3 text-[11px] leading-relaxed">
              <Legend heading="Shape" terms={BLOCK_TERMS} />
            </div>
          )}
        </section>
      )}

      {detailed && report.offBall.length > 0 && (
        <section className="rounded-lg border border-white/10 bg-slate-900/50 p-4">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Best positioned
          </h3>
          <p className="mb-2.5 text-[11px] leading-relaxed text-slate-500">
            Value of the ground each attacker is stood on, whether or not a pass
            can reach them. A player high here with no lane in the list above is
            a different problem to one with a lane and nowhere to go.
          </p>
          <ul className="space-y-1">
            {report.offBall.slice(0, 4).map((o) => (
              <li
                key={o.playerId}
                className="flex items-center justify-between gap-2 text-[11px]"
              >
                <span className="font-semibold text-slate-200">#{o.playerId}</span>
                <span className="flex items-center gap-2 text-slate-400">
                  <span className="font-mono">{o.threat.toFixed(4)} xT</span>
                  <span
                    className={
                      o.gainOverBall >= 0 ? "text-verdict-open" : "text-slate-600"
                    }
                  >
                    {o.gainOverBall >= 0 ? "+" : ""}
                    {o.gainOverBall.toFixed(4)}
                  </span>
                </span>
              </li>
            ))}
          </ul>
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
