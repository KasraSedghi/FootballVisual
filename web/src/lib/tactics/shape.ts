/**
 * Team shape and space control.
 *
 * These are the metrics that describe a *block* rather than a pass: how
 * compact a defence is, where its lines sit, where the gaps between its players
 * are, and how much of the pitch each team is closest to controlling.
 */

import { convexHull, dist, mean, polygonArea } from "./geometry";
import { HALF_LENGTH, HALF_WIDTH, PITCH_LENGTH, PITCH_WIDTH } from "./geometry";
import type { BlockShape, MotionModel, PlayerState, SpaceControl, TeamId, Vec2 } from "./types";
import { DEFAULT_MOTION } from "./types";

/**
 * Describe a team's shape.
 *
 * `goalX` is the goal this team defends, which sets what "deep" means. Without
 * it every front-to-back measurement would be sign-ambiguous and the defensive
 * line would be computed from the wrong end of the team.
 */
export function analyseShape(
  players: PlayerState[],
  team: TeamId,
  goalX: number,
  keepers: PlayerState[] = [],
): BlockShape | null {
  if (players.length < 3) return null;

  const pts: Vec2[] = players.map((p) => ({ x: p.x, y: p.y }));
  const hull = convexHull(pts);

  const centroid: Vec2 = { x: mean(pts.map((p) => p.x)), y: mean(pts.map((p) => p.y)) };
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);

  // Sort by depth: nearest the defended goal first.
  const towardOwnGoal = Math.sign(goalX) || 1;
  const byDepth = [...players].sort((a, b) => (b.x - a.x) * towardOwnGoal);

  const backLine = byDepth.slice(0, Math.min(4, byDepth.length));
  const front = byDepth.slice(-Math.min(2, byDepth.length));

  const defensiveLineX = mean(backLine.map((p) => p.x));
  const forwardLineX = mean(front.map((p) => p.x));

  // Largest lateral gap along the defensive line. This is the number a coach
  // actually looks for: not how compact the block is on average, but where the
  // one hole in it is.
  const backSorted = [...backLine].sort((a, b) => a.y - b.y);
  let largestGap = 0;
  let gapCentreY = 0;
  for (let i = 1; i < backSorted.length; i++) {
    const gap = backSorted[i].y - backSorted[i - 1].y;
    if (gap > largestGap) {
      largestGap = gap;
      gapCentreY = (backSorted[i].y + backSorted[i - 1].y) / 2;
    }
  }

  // Offside line: the deepest *outfield* defender.
  //
  // With a tracked keeper this is exact, because `players` already excludes
  // them and the deepest remaining defender is the line. Without one it falls
  // back to the second-deepest, which is what the line resolves to whenever the
  // keeper is the deepest player, and is the best available guess when nobody
  // has been identified as a keeper. The distinction is reported rather than
  // hidden, because an offside line drawn off the keeper instead of the last
  // defender is wrong by the length of the penalty area.
  const relevantKeeper = keepers
    .filter((k) => Math.sign(k.x) === Math.sign(goalX) || goalX === 0)
    .sort((a, b) => (b.x - a.x) * towardOwnGoal)[0];

  const offsideLineUsesKeeper = relevantKeeper !== undefined;
  const offsideLineX = offsideLineUsesKeeper
    ? byDepth[0].x
    : byDepth.length >= 2
      ? byDepth[1].x
      : byDepth[0].x;

  return {
    team,
    playerCount: players.length,
    centroid,
    hullAreaM2: polygonArea(hull),
    hull,
    widthM: Math.max(...ys) - Math.min(...ys),
    depthM: Math.max(...xs) - Math.min(...xs),
    defensiveLineX,
    forwardLineX,
    blockDepthM: Math.abs(forwardLineX - defensiveLineX),
    largestBackLineGapM: largestGap,
    backLineGapCentreY: gapCentreY,
    offsideLineX,
    offsideLineUsesKeeper,
  };
}

/**
 * Time for a player to reach a point, including reaction and current momentum.
 *
 * Shared with the passing-lane model in spirit but kept separate, because this
 * one answers "how long to arrive" while that one answers "who wins a race",
 * and fusing them would make both harder to reason about.
 */
function timeToReach(player: PlayerState, target: Vec2, model: MotionModel): number {
  const driftX = player.x + (player.vx ?? 0) * model.reactionTimeS;
  const driftY = player.y + (player.vy ?? 0) * model.reactionTimeS;
  const d = Math.hypot(target.x - driftX, target.y - driftY);
  return model.reactionTimeS + d / model.playerMaxSpeed;
}

/**
 * Which team controls each part of the pitch.
 *
 * A cell is contested in proportion to how close the two teams' best arrival
 * times are, via a logistic on the time difference rather than a hard nearest
 * player rule. A hard rule produces a Voronoi diagram, which looks decisive and
 * lies: it draws a crisp border between two players a tenth of a second apart.
 * The softness parameter is the number of seconds of advantage needed before
 * control is close to certain.
 */
export function analyseSpace(
  players: PlayerState[],
  teamA: TeamId,
  teamB: TeamId,
  cellM = 2,
  model: MotionModel = DEFAULT_MOTION,
  softnessS = 0.6,
): SpaceControl | null {
  const a = players.filter((p) => p.team === teamA);
  const b = players.filter((p) => p.team === teamB);
  if (!a.length || !b.length) return null;

  const cols = Math.ceil(PITCH_LENGTH / cellM);
  const rows = Math.ceil(PITCH_WIDTH / cellM);
  const grid = new Float32Array(cols * rows);

  let aCells = 0;
  let bCells = 0;

  for (let r = 0; r < rows; r++) {
    const y = -HALF_WIDTH + (r + 0.5) * cellM;
    for (let c = 0; c < cols; c++) {
      const x = -HALF_LENGTH + (c + 0.5) * cellM;
      const target: Vec2 = { x, y };

      let bestA = Infinity;
      for (const p of a) bestA = Math.min(bestA, timeToReach(p, target, model));
      let bestB = Infinity;
      for (const p of b) bestB = Math.min(bestB, timeToReach(p, target, model));

      // +1 means team B controls, -1 means team A controls.
      const value = Math.tanh((bestA - bestB) / softnessS);
      grid[r * cols + c] = value;
      if (value < 0) aCells++;
      else bCells++;
    }
  }

  const total = cols * rows;
  return {
    cellM,
    cols,
    rows,
    grid,
    teamAShare: aCells / total,
    teamBShare: bCells / total,
  };
}

/**
 * Attackers standing beyond the defending team's last line.
 *
 * Not an offside ruling: it takes no account of when the ball was played, which
 * is what the actual law turns on. It answers the useful question of who is
 * currently in behind.
 */
export function playersBeyondLine(
  attackers: PlayerState[],
  block: BlockShape,
  goalX: number,
): number[] {
  const toward = Math.sign(goalX) || 1;
  return attackers
    .filter((p) => (p.x - block.offsideLineX) * toward > 0)
    .map((p) => p.id);
}

/**
 * Attackers occupying the corridor between the opponent's two banks.
 *
 * This is the space a mid block is built to deny, so finding somebody standing
 * in it is usually the most important thing on the pitch. `bandM` widens the
 * corridor slightly at both ends, because a player on the shoulder of a line is
 * functionally between them.
 */
export function playersBetweenLines(
  attackers: PlayerState[],
  defenders: PlayerState[],
  goalX: number,
  bandM = 2,
): number[] {
  if (defenders.length < 6) return [];
  const toward = Math.sign(goalX) || 1;
  const byDepth = [...defenders].sort((a, b) => (b.x - a.x) * toward);

  const back = byDepth.slice(0, 4);
  const midfield = byDepth.slice(4, 8);
  if (!back.length || !midfield.length) return [];

  const backX = mean(back.map((p) => p.x));
  const midX = mean(midfield.map((p) => p.x));

  const lo = Math.min(backX, midX) - bandM;
  const hi = Math.max(backX, midX) + bandM;

  return attackers.filter((p) => p.x > lo && p.x < hi).map((p) => p.id);
}

/** Nearest player to the ball, used to infer who is carrying it. */
export function nearestPlayer(players: PlayerState[], point: Vec2): PlayerState | null {
  let best: PlayerState | null = null;
  let bestD = Infinity;
  for (const p of players) {
    const d = dist({ x: p.x, y: p.y }, point);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}
