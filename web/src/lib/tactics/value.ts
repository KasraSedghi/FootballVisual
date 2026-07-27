/**
 * What an action is worth: reward, risk, and the product of the two.
 *
 * The engine could already rank passing options, but only on a hand-weighted
 * blend of margin, progression and defenders bypassed. Those weights were
 * chosen by judgement, which is fine for ordering a list and not good enough to
 * say a pass is *worth* something. This module replaces judgement with two
 * measured quantities:
 *
 *   reward   how much threat the ball gains by arriving there, from an Expected
 *            Threat grid counted off several hundred real matches (`xt.ts`)
 *   risk     how often a pass with this interception margin actually gets
 *            through, fitted to real passes with known outcomes (`completion-model.json`)
 *
 * Neither is invented here, and both are checkable against the data they came
 * from, which is the standard the rest of this engine is held to.
 *
 * The valuation is deliberately "multi-agent" in the sense the ball-carrier view
 * is not: every team mate is a receiver with their own reward, their own risk
 * and their own value, and `offBallThreat` values where a player is standing
 * regardless of whether the ball ever comes. That is what lets the answer be
 * "the best thing about this shape is where #9 is stood", rather than only ever
 * being about the player on the ball.
 */

import fitted from "./completion-model.json";
import { threatAt } from "./xt";
import type {
  ActionValue,
  OffBallValue,
  PassingLane,
  PlayerState,
  TeamId,
  Vec2,
} from "./types";

const COEFFICIENTS: number[] = fitted.coefficients;
const USES_DISTANCE: boolean = fitted.uses_distance;
const [MARGIN_MIN, MARGIN_MAX] = fitted.margin_clip;
const [DISTANCE_MIN, DISTANCE_MAX] = fitted.distance_clip;
const DISTANCE_SCALE: number = fitted.distance_scale;

/** Provenance, so a number shown in the UI can always be attributed. */
export const COMPLETION_MODEL = {
  passes: fitted.train.samples,
  holdoutPasses: fitted.holdout.passes,
  baseRate: fitted.train.base_rate,
  skill: fitted.holdout.brier_skill,
  source: fitted.source,
};

/**
 * Probability that a pass with this margin and length is completed.
 *
 * The clipping is part of the fitted model rather than a detail: the
 * coefficients were estimated on clipped inputs, so applying them to an
 * unclipped margin would extrapolate a curve that was never fitted. A pass the
 * ball wins by four seconds is not more certain than one it wins by two, and
 * without the clip the logistic would claim it was.
 */
export function completionProbability(marginS: number, distanceM: number): number {
  const margin = Math.min(MARGIN_MAX, Math.max(MARGIN_MIN, marginS));
  const distance =
    Math.min(DISTANCE_MAX, Math.max(DISTANCE_MIN, distanceM)) / DISTANCE_SCALE;

  let z = COEFFICIENTS[0] + COEFFICIENTS[1] * margin;
  if (USES_DISTANCE) z += COEFFICIENTS[2] * distance;
  return 1 / (1 + Math.exp(-z));
}

export function valueOfPass(
  from: Vec2,
  to: Vec2,
  marginS: number,
  distanceM: number,
  attackingGoalX: number,
): ActionValue {
  const fromXT = threatAt(from, attackingGoalX);
  const toXT = threatAt(to, attackingGoalX);
  const completion = completionProbability(marginS, distanceM);

  return {
    fromXT,
    toXT,
    rewardXT: toXT - fromXT,
    completion,
    expectedXT: completion * toXT - fromXT,
  };
}

/** Value every lane in place, using the geometry the engine already computed. */
export function valueLanes(lanes: PassingLane[], attackingGoalX: number): PassingLane[] {
  return lanes.map((lane) => ({
    ...lane,
    value: valueOfPass(
      lane.from,
      lane.to,
      lane.safetyMarginS,
      lane.distanceM,
      attackingGoalX,
    ),
  }));
}

/**
 * Value where every attacker is standing, not just what the carrier can do.
 *
 * Excludes the carrier, whose position is the baseline everything else is
 * measured against.
 */
export function offBallThreat(
  players: PlayerState[],
  attackingTeam: TeamId,
  ball: Vec2 | null,
  carrierId: number | null,
  attackingGoalX: number,
): OffBallValue[] {
  const ballXT = ball ? threatAt(ball, attackingGoalX) : 0;

  return players
    .filter((p) => p.team === attackingTeam && p.id !== carrierId)
    .map((p) => {
      const threat = threatAt(p, attackingGoalX);
      return { playerId: p.id, threat, gainOverBall: threat - ballXT };
    })
    .sort((a, b) => b.threat - a.threat);
}
