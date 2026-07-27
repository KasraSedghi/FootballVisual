/**
 * Searching a clip for the moments an analyst actually wants to see.
 *
 * This is the retrieval layer, and it holds the same invariant as everything
 * else in `tactics/`: geometry decides, the model only writes. A language model
 * is good at turning "when did we play through the middle against a compact
 * block" into a *query*, and bad at deciding whether frame 137 qualifies. So
 * the query is a structured predicate over already-computed reports, evaluated
 * here, in code, over every frame.
 *
 * The model never sees positions and never scores a frame. It picks filters.
 * That means a search is reproducible, explainable, and cannot return a moment
 * that does not satisfy what was asked, which are exactly the properties that
 * make a retrieval tool trustworthy enough to put in front of a coach.
 */

import type { Snapshot, TacticalReport, TeamId } from "./types";
import { analyseSnapshot } from "./index";

/**
 * A structured tactical query.
 *
 * Every field is optional and they combine with AND. Ranges are inclusive, and
 * both ends are optional so a caller can bound one side only, which is what
 * most real questions want ("wider than 40m", with no upper limit).
 */
export interface TacticalQuery {
  /** At least this many passing lanes judged open. */
  minOpenLanes?: number;
  /** The best available lane must clear this margin, in seconds. */
  minBestMarginS?: number;
  /** Defensive block width, in metres. */
  minBlockWidthM?: number;
  maxBlockWidthM?: number;
  /** Distance from the deepest to the highest defender, in metres. */
  minBlockDepthM?: number;
  maxBlockDepthM?: number;
  /** The largest gap along the defensive line, in metres. */
  minBackLineGapM?: number;
  /** At least this many attackers positioned between the opponent's lines. */
  minPlayersBetweenLines?: number;
  /** At least this many attackers beyond the last defensive line. */
  minPlayersBeyondLine?: number;
  /** Restrict to frames where this team is in possession. */
  attackingTeam?: TeamId;
  /** A lane must gain at least this many metres toward goal to count as open. */
  minProgressionM?: number;
}

export interface SearchHit {
  frame: number;
  timeS: number;
  /** Index into the snapshot array, so a caller can seek without re-searching. */
  index: number;
  /** Why this frame matched, as measured values rather than prose. */
  measurements: {
    openLanes: number;
    bestMarginS: number | null;
    blockWidthM: number | null;
    blockDepthM: number | null;
    backLineGapM: number | null;
    playersBetweenLines: number;
    playersBeyondLine: number;
    attackingTeam: TeamId;
  };
}

/**
 * Consecutive matching frames, collapsed into one result.
 *
 * Without this a two second passage at 25fps returns fifty near-identical hits
 * and buries every other moment in the clip. An analyst asked for the passage,
 * not for its frames.
 */
export interface SearchSequence {
  startFrame: number;
  endFrame: number;
  startTimeS: number;
  endTimeS: number;
  frameCount: number;
  /** The single best frame in the run, by the ranking below. */
  peak: SearchHit;
}

function measure(report: TacticalReport, query: TacticalQuery): SearchHit["measurements"] {
  const progressionFloor = query.minProgressionM ?? 0;
  const open = report.lanes.filter(
    (l) => l.verdict === "open" && l.progressionM >= progressionFloor,
  );
  // `safetyMarginS` is Infinity for a lane with no defender in the race at all,
  // which is a legitimate value but not one to report as a number.
  const margins = report.lanes
    .map((l) => l.safetyMarginS)
    .filter((m) => Number.isFinite(m));

  return {
    openLanes: open.length,
    bestMarginS: margins.length ? Math.max(...margins) : null,
    blockWidthM: report.block?.widthM ?? null,
    blockDepthM: report.block?.blockDepthM ?? null,
    backLineGapM: report.block?.largestBackLineGapM ?? null,
    playersBetweenLines: report.playersBetweenLines.length,
    playersBeyondLine: report.playersBeyondLine.length,
    attackingTeam: report.attackingTeam,
  };
}

function satisfies(m: SearchHit["measurements"], q: TacticalQuery): boolean {
  // A missing measurement fails any filter that names it. The alternative,
  // treating "not computed" as "passes", would return frames as evidence for a
  // property that was never established on them.
  const atLeast = (value: number | null, bound: number | undefined) =>
    bound === undefined || (value !== null && value >= bound);
  const atMost = (value: number | null, bound: number | undefined) =>
    bound === undefined || (value !== null && value <= bound);

  return (
    atLeast(m.openLanes, q.minOpenLanes) &&
    atLeast(m.bestMarginS, q.minBestMarginS) &&
    atLeast(m.blockWidthM, q.minBlockWidthM) &&
    atMost(m.blockWidthM, q.maxBlockWidthM) &&
    atLeast(m.blockDepthM, q.minBlockDepthM) &&
    atMost(m.blockDepthM, q.maxBlockDepthM) &&
    atLeast(m.backLineGapM, q.minBackLineGapM) &&
    atLeast(m.playersBetweenLines, q.minPlayersBetweenLines) &&
    atLeast(m.playersBeyondLine, q.minPlayersBeyondLine) &&
    (q.attackingTeam === undefined || m.attackingTeam === q.attackingTeam)
  );
}

/**
 * Rank a hit so a run of frames can be collapsed to its most interesting one.
 *
 * Deliberately simple and deliberately fixed: an open lane with a large margin
 * through a stretched block is the moment worth showing. This is a presentation
 * choice about which frame to seek to, not a tactical judgment, and every frame
 * in the run already satisfies the query.
 */
function rank(m: SearchHit["measurements"]): number {
  return (
    m.openLanes * 10 +
    (m.bestMarginS ?? 0) * 4 +
    (m.backLineGapM ?? 0) * 0.5 +
    m.playersBetweenLines * 3
  );
}

/**
 * Evaluate a query over every frame of a clip.
 *
 * `reports` may be supplied when they have already been computed, which the
 * sandbox does frame by frame anyway. Otherwise they are computed here, and the
 * previous snapshot is threaded through so possession stays stable exactly as
 * it does during playback.
 */
export function searchFrames(
  snapshots: Snapshot[],
  query: TacticalQuery,
  reports?: (TacticalReport | null)[],
): SearchHit[] {
  const hits: SearchHit[] = [];

  for (let i = 0; i < snapshots.length; i += 1) {
    const report =
      reports?.[i] ??
      analyseSnapshot(snapshots[i], {
        previous: i > 0 ? snapshots[i - 1] : null,
        // Space control is a grid solve per frame and no query filters on it,
        // so computing it here would multiply the cost of a search for nothing.
        computeSpace: false,
      });
    if (!report) continue;

    const measurements = measure(report, query);
    if (!satisfies(measurements, query)) continue;

    hits.push({
      frame: snapshots[i].frame,
      timeS: snapshots[i].timeS,
      index: i,
      measurements,
    });
  }

  return hits;
}

/**
 * Collapse hits into passages, allowing a short gap inside one.
 *
 * `maxGapFrames` exists because tracking flickers. A player lost for two frames
 * can drop the open-lane count below the threshold and split one passage into
 * three, which is an artefact of detection rather than anything that happened
 * on the pitch.
 */
export function groupIntoSequences(
  hits: SearchHit[],
  maxGapFrames = 5,
): SearchSequence[] {
  if (!hits.length) return [];

  const sequences: SearchSequence[] = [];
  let run: SearchHit[] = [hits[0]];

  const flush = () => {
    const peak = run.reduce((best, h) => (rank(h.measurements) > rank(best.measurements) ? h : best));
    sequences.push({
      startFrame: run[0].frame,
      endFrame: run[run.length - 1].frame,
      startTimeS: run[0].timeS,
      endTimeS: run[run.length - 1].timeS,
      frameCount: run.length,
      peak,
    });
  };

  for (let i = 1; i < hits.length; i += 1) {
    if (hits[i].index - hits[i - 1].index <= maxGapFrames) {
      run.push(hits[i]);
    } else {
      flush();
      run = [hits[i]];
    }
  }
  flush();

  return sequences.sort((a, b) => rank(b.peak.measurements) - rank(a.peak.measurements));
}

/**
 * Say what a query asked for, in words, without a language model.
 *
 * The point is that a returned passage can always explain itself even when
 * there is no API key, and that the explanation is generated from the same
 * structure that did the filtering rather than from a model's recollection of
 * it. If these two ever disagree, the search is what is right.
 */
export function describeQuery(query: TacticalQuery): string {
  const parts: string[] = [];
  const push = (s: string) => parts.push(s);

  if (query.minOpenLanes !== undefined)
    push(`at least ${query.minOpenLanes} open passing lane${query.minOpenLanes === 1 ? "" : "s"}`);
  if (query.minBestMarginS !== undefined)
    push(`a lane clear by ${query.minBestMarginS}s or more`);
  if (query.minProgressionM !== undefined)
    push(`lanes gaining at least ${query.minProgressionM}m`);
  if (query.minBlockWidthM !== undefined) push(`block at least ${query.minBlockWidthM}m wide`);
  if (query.maxBlockWidthM !== undefined) push(`block no wider than ${query.maxBlockWidthM}m`);
  if (query.minBlockDepthM !== undefined) push(`block at least ${query.minBlockDepthM}m deep`);
  if (query.maxBlockDepthM !== undefined) push(`block no deeper than ${query.maxBlockDepthM}m`);
  if (query.minBackLineGapM !== undefined)
    push(`a gap of ${query.minBackLineGapM}m or more in the last line`);
  if (query.minPlayersBetweenLines !== undefined)
    push(`${query.minPlayersBetweenLines} or more players between the lines`);
  if (query.minPlayersBeyondLine !== undefined)
    push(`${query.minPlayersBeyondLine} or more players beyond the last line`);
  if (query.attackingTeam !== undefined) push(`${query.attackingTeam} in possession`);

  if (!parts.length) return "every frame (no filters were set)";
  return parts.join(", ");
}
