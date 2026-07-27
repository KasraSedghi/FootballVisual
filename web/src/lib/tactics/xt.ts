/**
 * Expected Threat: what a position on the pitch is worth.
 *
 * The engine could already say whether a lane was open. It could not say
 * whether the pass was worth playing, because nothing in it knew that a metre
 * gained at the edge of the box matters more than a metre gained in your own
 * half. `progressionM` treated both the same. This module supplies the missing
 * scale: the probability that a possession at a given place ends in a goal.
 *
 * The grid is trained offline by `pipeline/footballvisual/xt.py` on StatsBomb
 * open data and committed as JSON, so nothing at runtime needs the network. See
 * that module for the recursion and why it converges.
 *
 * This stays on the measurement side of the project's central line. The values
 * come from counting what happened in several hundred real matches, so a claim
 * that one pass is worth more than another is arithmetic over observed
 * frequencies rather than an opinion.
 */

import model from "./xt-grid.json";
import { HALF_LENGTH, HALF_WIDTH } from "../pitch";
import type { Vec2 } from "./types";

const GRID: number[][] = model.grid;
const ROWS = model.rows;
const COLS = model.cols;

/** Provenance, surfaced in the UI so the number is never unattributed. */
export const XT_MODEL = {
  matches: model.matches,
  actions: model.actions,
  goals: model.goals,
  source: model.source,
  mirrored: model.mirrored,
};

function at(row: number, col: number): number {
  const r = Math.min(ROWS - 1, Math.max(0, row));
  const c = Math.min(COLS - 1, Math.max(0, col));
  return GRID[r][c];
}

/**
 * Value of a point on the pitch, for a team attacking toward `attackingGoalX`.
 *
 * Bilinear rather than nearest cell. A 12x8 grid has cells nearly 9m long, and
 * a nearest-cell lookup makes the value jump in steps at the boundaries, which
 * shows up as a pass gaining a suspiciously round amount of threat whenever it
 * happens to cross one. Interpolating makes the gradient continuous, which is
 * what "value gained by moving the ball five metres" needs in order to mean
 * anything.
 *
 * The `attackingGoalX` flip is the same reflection the retrieval embedding
 * uses, and for the same reason: the model is trained in StatsBomb's convention
 * where the attacking direction is always +x, so a team going the other way
 * reads the grid mirrored rather than needing a second grid.
 */
export function threatAt(point: Vec2, attackingGoalX: number): number {
  const x = attackingGoalX < 0 ? -point.x : point.x;
  const y = attackingGoalX < 0 ? -point.y : point.y;

  // To [0, 1] along each axis, then to fractional cell centres.
  const fx = (x + HALF_LENGTH) / (2 * HALF_LENGTH);
  const fy = (y + HALF_WIDTH) / (2 * HALF_WIDTH);

  const cx = Math.min(COLS - 1, Math.max(0, fx * COLS - 0.5));
  const cy = Math.min(ROWS - 1, Math.max(0, fy * ROWS - 0.5));

  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const tx = cx - x0;
  const ty = cy - y0;

  const top = at(y0, x0) * (1 - tx) + at(y0, x0 + 1) * tx;
  const bottom = at(y0 + 1, x0) * (1 - tx) + at(y0 + 1, x0 + 1) * tx;
  return top * (1 - ty) + bottom * ty;
}

/**
 * Threat gained by moving the ball from one point to another.
 *
 * Signed: a backward pass has negative value, which is correct and is the
 * reason this replaces `progressionM` as the measure of what a pass achieves. A
 * square ball across your own box loses value even though it gains no ground at
 * all, and a 30m pass down the touchline gains far less than 30m through the
 * middle.
 */
export function threatDelta(from: Vec2, to: Vec2, attackingGoalX: number): number {
  return threatAt(to, attackingGoalX) - threatAt(from, attackingGoalX);
}
