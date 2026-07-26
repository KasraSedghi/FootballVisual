/**
 * Passing lane analysis by interception feasibility.
 *
 * The naive way to decide whether a lane is open is to measure how close the
 * nearest defender is to the line between passer and receiver. That is wrong in
 * both directions, and wrong in ways that matter. A defender standing two
 * metres off a 40 metre pass has all the time in the world to step across it,
 * so the lane is shut despite the clearance looking fine. A defender half a
 * metre off a 6 metre pass, facing the wrong way, cannot get a foot to it
 * before it arrives, so the lane is open despite the clearance looking fatal.
 *
 * What actually decides a pass is a race. The ball travels its path at pass
 * speed; the defender has to reach some point on that path before the ball
 * passes through it. So the model here computes, for every point along the
 * lane, when the ball gets there and when the defender could get there, and
 * takes the worst case over the whole path:
 *
 *     margin = min over path points s of [ t_defender(s) - t_ball(s) ]
 *
 * A positive margin is the number of seconds the ball wins by. Negative means
 * the defender arrives first and the pass gets cut out. This is a simplified
 * form of the pitch-control models used in the tracking-data literature
 * (Spearman, Fernandez and Bornn), reduced to the single question a lane needs
 * answered.
 *
 * Deliberate simplifications, all of which make the model slightly pessimistic
 * about the attacker: the ball travels at constant speed and never leaves the
 * ground, defenders accelerate instantly to top speed after their reaction
 * delay, and no defender is credited with reading the pass before it is struck.
 */

import { dist, normalise, pointToSegment, sub } from "./geometry";
import type {
  LaneVerdict,
  MotionModel,
  PassingLane,
  PlayerState,
  Vec2,
} from "./types";
import { DEFAULT_MOTION } from "./types";

/** How finely the lane is sampled. 48 points over even a 60m pass is ~1.2m. */
const PATH_SAMPLES = 48;

/** Above this margin a lane is called open; below zero it is blocked. */
const OPEN_MARGIN_S = 0.32;
const BLOCKED_MARGIN_S = 0.0;

/**
 * Where a defender actually starts from, once their momentum is accounted for.
 *
 * Momentum genuinely changes outcomes: a defender already sprinting away from
 * the lane cannot simply stop. During the reaction window they keep their
 * current velocity, and only afterwards can they move freely toward the
 * interception point, so the race is run from where the drift leaves them.
 */
function driftedPosition(defender: PlayerState, model: MotionModel): Vec2 {
  const vx = defender.vx ?? 0;
  const vy = defender.vy ?? 0;
  return {
    x: defender.x + vx * model.reactionTimeS,
    y: defender.y + vy * model.reactionTimeS,
  };
}

/**
 * Seconds by which the ball beats `defender` to the lane, worst point on path.
 *
 * Also returns where along the path the defender is most dangerous, which is
 * what the sandbox draws as the pinch point.
 */
export function interceptionMargin(
  from: Vec2,
  to: Vec2,
  defender: PlayerState,
  model: MotionModel = DEFAULT_MOTION,
): { marginS: number; atDistanceM: number } {
  const total = dist(from, to);
  if (total < 1e-6) return { marginS: Infinity, atDistanceM: 0 };

  const dir = normalise(sub(to, from));
  const start = driftedPosition(defender, model);

  let worst = Infinity;
  let worstAt = 0;

  for (let i = 0; i <= PATH_SAMPLES; i++) {
    const s = (i / PATH_SAMPLES) * total;
    const point = { x: from.x + dir.x * s, y: from.y + dir.y * s };

    const tBall = s / model.passSpeed;
    // The defender only has to get within a boot's reach of the path, not to
    // the exact point, so the interception radius is subtracted from the
    // distance they must cover.
    const need = Math.max(0, dist(start, point) - model.interceptRadius);
    const tDef = model.reactionTimeS + need / model.playerMaxSpeed;

    const margin = tDef - tBall;
    if (margin < worst) {
      worst = margin;
      worstAt = s;
    }
  }
  return { marginS: worst, atDistanceM: worstAt };
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Which goal a team attacks, as a pitch x coordinate.
 *
 * Decided by comparing the two teams' median x, not their mean. The mean lets a
 * single player invert the answer: one defender caught upfield, one tracking
 * error, or one player dragged across the halfway line in the sandbox is enough
 * to swing a ten-player average past the other team's. When that happens the
 * engine silently decides the wrong goal is being attacked, and every forward
 * pass is then reported as losing ground, which is a confusing failure because
 * the lane verdicts stay correct while the progression numbers all flip sign.
 * A median shrugs that off.
 */
export function attackingGoalX(attackers: PlayerState[], defenders: PlayerState[]): number {
  if (!attackers.length || !defenders.length) return 52.5;
  const attackMid = median(attackers.map((p) => p.x));
  const defendMid = median(defenders.map((p) => p.x));
  if (attackMid === defendMid) return 52.5;
  // Whichever end the defenders sit closer to is the end being attacked.
  return defendMid > attackMid ? 52.5 : -52.5;
}

function verdictFor(marginS: number): LaneVerdict {
  if (marginS < BLOCKED_MARGIN_S) return "blocked";
  if (marginS < OPEN_MARGIN_S) return "contested";
  return "open";
}

/**
 * Evaluate every pass available to `carrier`.
 *
 * Returned lanes are sorted by `score`, which blends safety with how much the
 * pass actually achieves. Safety alone would rank a backward pass to an unmarked
 * keeper as the best option on the pitch, which is true and useless.
 */
export function analysePassingLanes(
  carrier: PlayerState,
  teammates: PlayerState[],
  defenders: PlayerState[],
  model: MotionModel = DEFAULT_MOTION,
  goalX?: number,
): PassingLane[] {
  const from: Vec2 = { x: carrier.x, y: carrier.y };
  const targetGoalX = goalX ?? attackingGoalX([carrier, ...teammates], defenders);
  const towardGoal = Math.sign(targetGoalX) || 1;

  const lanes: PassingLane[] = [];

  for (const mate of teammates) {
    if (mate.id === carrier.id) continue;
    const to: Vec2 = { x: mate.x, y: mate.y };
    const distanceM = dist(from, to);
    if (distanceM < 1.5) continue;

    let worstMargin = Infinity;
    let threatId: number | null = null;
    let threatAt = 0;
    let minClearance = Infinity;

    for (const def of defenders) {
      const { marginS, atDistanceM } = interceptionMargin(from, to, def, model);
      if (marginS < worstMargin) {
        worstMargin = marginS;
        threatId = def.id;
        threatAt = atDistanceM;
      }
      const clearance = pointToSegment({ x: def.x, y: def.y }, from, to);
      if (clearance < minClearance) minClearance = clearance;
    }

    if (!defenders.length) {
      worstMargin = Infinity;
      minClearance = Infinity;
    }

    const progressionM = (mate.x - carrier.x) * towardGoal;

    // A defender is bypassed when the ball ends up beyond them, relative to the
    // goal being attacked. This is the number that separates a sideways pass
    // from one that actually breaks a line.
    const defendersBypassed = defenders.filter(
      (d) => (mate.x - d.x) * towardGoal > 0 && (carrier.x - d.x) * towardGoal < 0,
    ).length;

    const receiverPressureM = defenders.length
      ? Math.min(...defenders.map((d) => dist({ x: d.x, y: d.y }, to)))
      : Infinity;

    lanes.push({
      targetId: mate.id,
      from,
      to,
      distanceM,
      safetyMarginS: worstMargin,
      threatId,
      threatPointM: threatAt,
      minClearanceM: minClearance,
      progressionM,
      defendersBypassed,
      receiverPressureM,
      verdict: verdictFor(worstMargin),
      score: 0,
    });
  }

  for (const lane of lanes) {
    lane.score = scoreLane(lane);
  }
  lanes.sort((a, b) => b.score - a.score);
  return lanes;
}

/**
 * Rank a lane from 0 to 1.
 *
 * The weights are a judgement call, not a fitted model, and they are stated
 * here rather than buried so they can be argued with. Safety dominates because
 * a pass that gets intercepted in midfield is the worst outcome available;
 * progression and line-breaking come next because they are why you would risk a
 * pass at all; receiver pressure is a smaller term because a good receiver can
 * often survive a tight one.
 */
export function scoreLane(lane: PassingLane): number {
  // Safety saturates: two seconds of margin is not meaningfully safer than one.
  const safety = lane.safetyMarginS === Infinity
    ? 1
    : Math.max(0, Math.min(1, (lane.safetyMarginS + 0.2) / 1.2));

  const progression = Math.max(0, Math.min(1, lane.progressionM / 25));
  const breaking = Math.max(0, Math.min(1, lane.defendersBypassed / 4));
  const freedom = lane.receiverPressureM === Infinity
    ? 1
    : Math.max(0, Math.min(1, lane.receiverPressureM / 12));

  // A blocked lane is worth nothing regardless of how attractive the
  // destination is, so safety multiplies rather than adds.
  if (lane.safetyMarginS < BLOCKED_MARGIN_S) {
    return 0.05 * safety;
  }

  return (
    0.45 * safety +
    0.22 * progression +
    0.20 * breaking +
    0.13 * freedom
  );
}

/**
 * Estimate velocities from consecutive snapshots, in m/s.
 *
 * Uses a centred difference over `windowS` rather than the last frame, because
 * per-frame position noise of even 20cm becomes a 5 m/s velocity error at 25fps
 * and would swamp the reaction-time term in the interception model.
 */
export function estimateVelocities(
  current: PlayerState[],
  previous: PlayerState[] | null,
  dtS: number,
): PlayerState[] {
  if (!previous || dtS <= 1e-6) {
    return current.map((p) => ({ ...p, vx: p.vx ?? 0, vy: p.vy ?? 0 }));
  }
  const byId = new Map(previous.map((p) => [p.id, p]));
  return current.map((p) => {
    const before = byId.get(p.id);
    if (!before) return { ...p, vx: 0, vy: 0 };
    const vx = (p.x - before.x) / dtS;
    const vy = (p.y - before.y) / dtS;
    // Cap at a plausible sprint. Anything faster is a tracking artefact, most
    // often an identity switch teleporting a track across the pitch.
    const speed = Math.hypot(vx, vy);
    const capped = speed > 9 ? 9 / speed : 1;
    return { ...p, vx: vx * capped, vy: vy * capped };
  });
}

export { OPEN_MARGIN_S, BLOCKED_MARGIN_S };
