/**
 * The tactical analysis engine's public entry point.
 *
 * One rule governs this whole module and the LLM route that sits on top of it:
 * every tactical claim the product makes is computed here, deterministically,
 * from geometry. The language model narrates this output. It never decides
 * which lane is open, who is in possession, or where the space is. That
 * separation is what makes the analysis reproducible and checkable, and it
 * means a model outage degrades the product to "correct but terse" rather than
 * to "confidently wrong".
 */

import { analysePassingLanes, attackingGoalX, estimateVelocities } from "./lanes";
import {
  analyseShape,
  analyseSpace,
  nearestPlayer,
  playersBetweenLines,
  playersBeyondLine,
} from "./shape";
import type {
  MotionModel,
  PlayerState,
  Snapshot,
  TacticalReport,
  TeamId,
} from "./types";
import { DEFAULT_MOTION } from "./types";

export * from "./types";
export * from "./geometry";
export { analysePassingLanes, interceptionMargin, estimateVelocities, scoreLane } from "./lanes";
export { analyseShape, analyseSpace, playersBeyondLine, playersBetweenLines } from "./shape";

/**
 * Decide which team is in possession.
 *
 * Nearest player to the ball, but only if they are close enough to plausibly
 * have it. During a pass nobody is within that radius, so the previous carrier
 * is kept via `fallbackTeam`, which stops the whole report from flipping
 * allegiance for the half second a pass is in flight.
 */
export function inferPossession(
  snapshot: Snapshot,
  fallbackTeam: TeamId | null = null,
  controlRadiusM = 3.0,
): { attacking: TeamId; carrierId: number | null } {
  const teams = [...new Set(snapshot.players.map((p) => p.team))].filter(
    (t) => t === "team_a" || t === "team_b",
  ) as TeamId[];

  if (!snapshot.ball) {
    return { attacking: fallbackTeam ?? teams[0] ?? "unknown", carrierId: null };
  }

  const candidates = snapshot.players.filter((p) => p.team === "team_a" || p.team === "team_b");
  const nearest = nearestPlayer(candidates, snapshot.ball);
  if (!nearest) {
    return { attacking: fallbackTeam ?? teams[0] ?? "unknown", carrierId: null };
  }

  const d = Math.hypot(nearest.x - snapshot.ball.x, nearest.y - snapshot.ball.y);
  if (d > controlRadiusM) {
    return { attacking: fallbackTeam ?? nearest.team, carrierId: null };
  }
  return { attacking: nearest.team, carrierId: nearest.id };
}

/**
 * Run the full analysis for one instant.
 *
 * When nobody is clearly carrying the ball, the attacking team's player closest
 * to it stands in as the passer, so the sandbox still shows the options that
 * are about to exist rather than going blank mid-pass.
 */
export function analyseSnapshot(
  snapshot: Snapshot,
  options: {
    previous?: Snapshot | null;
    model?: MotionModel;
    fallbackTeam?: TeamId | null;
    spaceCellM?: number;
    computeSpace?: boolean;
  } = {},
): TacticalReport {
  const model = options.model ?? DEFAULT_MOTION;
  const computeSpace = options.computeSpace ?? true;

  const dtS = options.previous ? snapshot.timeS - options.previous.timeS : 0;
  const players = estimateVelocities(
    snapshot.players,
    options.previous?.players ?? null,
    dtS,
  );
  const resolved: Snapshot = { ...snapshot, players };

  // Possession is inferred from proximity to the ball, and with a position
  // error around a metre it flickers to the defending team whenever the ball is
  // in flight or passes near an opponent. Every lane then reads as backward,
  // because the analysis has silently switched which goal is being attacked.
  // Falling back to the previous frame's answer rides out those ambiguous
  // frames. Deriving it from `previous` rather than from caller-held state
  // keeps this a pure function of the two frames.
  const fallback =
    options.fallbackTeam ??
    (options.previous ? inferPossession(options.previous).attacking : null);

  const { attacking, carrierId } = inferPossession(resolved, fallback);
  const defending: TeamId = attacking === "team_a" ? "team_b" : "team_a";

  const attackers = players.filter((p) => p.team === attacking);
  const defenders = players.filter((p) => p.team === defending);
  // Keepers are deliberately kept out of both squads. Including one would
  // stretch the measured block depth by the 40m they stand behind the line, and
  // would put a goalkeeper in the list of passing options.
  const keepers = players.filter((p) => p.team === "keeper");

  const goalX = attackers.length && defenders.length ? attackingGoalX(attackers, defenders) : 52.5;

  let carrier: PlayerState | null =
    carrierId != null ? players.find((p) => p.id === carrierId) ?? null : null;
  if (!carrier && resolved.ball && attackers.length) {
    carrier = nearestPlayer(attackers, resolved.ball);
  }

  const lanes = carrier ? analysePassingLanes(carrier, attackers, defenders, model, goalX) : [];

  // The block defends the goal the attackers are running at.
  const block =
    defenders.length >= 3 ? analyseShape(defenders, defending, goalX, keepers) : null;
  const attackingShape =
    attackers.length >= 3 ? analyseShape(attackers, attacking, -goalX, keepers) : null;

  const space = computeSpace
    ? analyseSpace(players, "team_a", "team_b", options.spaceCellM ?? 2, model)
    : null;

  return {
    snapshot: resolved,
    attackingTeam: attacking,
    defendingTeam: defending,
    carrierId: carrier?.id ?? null,
    lanes,
    block,
    attackingShape,
    space,
    playersBeyondLine: block ? playersBeyondLine(attackers, block, goalX) : [],
    playersBetweenLines: playersBetweenLines(attackers, defenders, goalX),
  };
}

/**
 * Flatten a report into the facts the language model is allowed to talk about.
 *
 * Everything here is already computed and already true. The model's job is to
 * turn it into a sentence a coach would say, so it is given numbers and never
 * raw coordinates to reason from: handing it positions would invite it to redo
 * the geometry in its head and get a different answer to the one on screen.
 */
export function reportFacts(report: TacticalReport) {
  const nameOf = (id: number | null) => {
    if (id == null) return null;
    const p = report.snapshot.players.find((q) => q.id === id);
    return p?.label ?? `#${id}`;
  };

  return {
    attackingTeam: report.attackingTeam,
    defendingTeam: report.defendingTeam,
    ballCarrier: nameOf(report.carrierId),
    lanes: report.lanes.slice(0, 6).map((l) => ({
      target: nameOf(l.targetId),
      verdict: l.verdict,
      safetyMarginS: Number(l.safetyMarginS.toFixed(2)),
      distanceM: Number(l.distanceM.toFixed(1)),
      progressionM: Number(l.progressionM.toFixed(1)),
      defendersBypassed: l.defendersBypassed,
      receiverPressureM: Number(l.receiverPressureM.toFixed(1)),
      closestDefender: nameOf(l.threatId),
    })),
    defensiveBlock: report.block
      ? {
          players: report.block.playerCount,
          hullAreaM2: Math.round(report.block.hullAreaM2),
          widthM: Number(report.block.widthM.toFixed(1)),
          blockDepthM: Number(report.block.blockDepthM.toFixed(1)),
          defensiveLineX: Number(report.block.defensiveLineX.toFixed(1)),
          largestBackLineGapM: Number(report.block.largestBackLineGapM.toFixed(1)),
          gapCentreY: Number(report.block.backLineGapCentreY.toFixed(1)),
        }
      : null,
    spaceControl: report.space
      ? {
          teamAShare: Number(report.space.teamAShare.toFixed(3)),
          teamBShare: Number(report.space.teamBShare.toFixed(3)),
        }
      : null,
    playersBetweenLines: report.playersBetweenLines.map(nameOf),
    playersBeyondLine: report.playersBeyondLine.map(nameOf),
  };
}
