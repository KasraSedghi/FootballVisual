import { describe, expect, it } from "vitest";

import { threatAt, threatDelta } from "./xt";
import { completionProbability, offBallThreat, valueOfPass } from "./value";
import { HALF_LENGTH } from "../pitch";
import type { PlayerState } from "./types";

const RIGHT = 52.5;
const LEFT = -52.5;

describe("threatAt", () => {
  it("rises as the ball gets closer to the goal being attacked", () => {
    const own = threatAt({ x: -40, y: 0 }, RIGHT);
    const middle = threatAt({ x: 0, y: 0 }, RIGHT);
    const edgeOfBox = threatAt({ x: 35, y: 0 }, RIGHT);

    expect(middle).toBeGreaterThan(own);
    expect(edgeOfBox).toBeGreaterThan(middle);
  });

  it("values the middle above the touchline at the same distance from goal", () => {
    /*
     * The grid was never told this. It falls out of the recursion, because
     * possessions in the centre reach shooting positions more often than
     * possessions on the wing do. If this ever inverts, the training data or
     * the coordinate conversion is wrong.
     */
    expect(threatAt({ x: 35, y: 0 }, RIGHT)).toBeGreaterThan(
      threatAt({ x: 35, y: 30 }, RIGHT),
    );
  });

  it("gives the same value to a position and its rotation for the other goal", () => {
    /*
     * Attacking the other way is the pitch rotated 180 degrees, so both
     * coordinates flip. A mismatch here means a team going right to left is
     * being valued on someone else's half.
     */
    for (const point of [
      { x: 30, y: 12 },
      { x: -20, y: -25 },
      { x: 0, y: 0 },
    ]) {
      expect(threatAt(point, LEFT)).toBeCloseTo(
        threatAt({ x: -point.x, y: -point.y }, RIGHT),
        9,
      );
    }
  });

  it("is continuous, so a small step never jumps the value", () => {
    /*
     * What the bilinear interpolation buys. With a nearest cell lookup the value
     * is a staircase, and a pass gains threat only when it happens to cross a
     * cell boundary nearly 9m wide.
     */
    let biggest = 0;
    for (let x = -50; x < 50; x += 0.5) {
      const step = Math.abs(threatAt({ x: x + 0.5, y: 4 }, RIGHT) - threatAt({ x, y: 4 }, RIGHT));
      biggest = Math.max(biggest, step);
    }
    expect(biggest).toBeLessThan(0.01);
  });

  it("clamps beyond the byline rather than reading off the end of the grid", () => {
    const beyond = threatAt({ x: HALF_LENGTH + 20, y: 0 }, RIGHT);
    expect(Number.isFinite(beyond)).toBe(true);
    expect(beyond).toBeCloseTo(threatAt({ x: HALF_LENGTH, y: 0 }, RIGHT), 9);
  });
});

describe("threatDelta", () => {
  it("is negative for a pass away from the goal", () => {
    expect(threatDelta({ x: 30, y: 0 }, { x: 5, y: 0 }, RIGHT)).toBeLessThan(0);
  });

  it("is positive for a pass toward it", () => {
    expect(threatDelta({ x: 5, y: 0 }, { x: 30, y: 0 }, RIGHT)).toBeGreaterThan(0);
  });
});

describe("completionProbability", () => {
  it("increases with margin", () => {
    const cut = completionProbability(-0.8, 20);
    const tight = completionProbability(0, 20);
    const free = completionProbability(1.0, 20);

    expect(tight).toBeGreaterThan(cut);
    expect(free).toBeGreaterThan(tight);
  });

  it("rises with length at a fixed margin, which is not a mistake", () => {
    /*
     * The counterintuitive one, and it is what the data says. Unconditionally
     * longer passes complete far less often (0.91 at 10-20m against 0.40 beyond
     * 45m). Hold the interception margin fixed and the sign flips, in every
     * margin band measured:
     *
     *   margin -0.4..0.1   0.68 short, 0.80 long
     *   margin  0.1..0.3   0.79 short, 0.91 long
     *   margin  0.3..0.7   0.96 short, 0.97 long
     *
     * The reason is that the margin means different things at the two lengths.
     * A six metre pass that only just beats the race has a defender on top of
     * it in a tight area; a forty metre pass with the same margin is travelling
     * through genuinely open space. Length is already priced into the margin,
     * so the residual effect is about the space around the ball.
     */
    expect(completionProbability(0.3, 45)).toBeGreaterThan(
      completionProbability(0.3, 8),
    );
  });

  it("stays a probability at any input, including absurd ones", () => {
    for (const margin of [-Infinity, -50, 0, 50, Infinity]) {
      for (const distance of [0, 25, 500]) {
        const p = completionProbability(margin, distance);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    }
  });

  it("saturates rather than extrapolating past the fitted range", () => {
    // The coefficients were fitted on clipped inputs, so anything beyond the
    // clip must return the value at the clip rather than run off the curve.
    expect(completionProbability(2.0, 20)).toBeCloseTo(completionProbability(9.9, 20), 12);
  });
});

describe("valueOfPass", () => {
  it("prices a safe square ball in your own half as a loss", () => {
    /*
     * The result that makes the valuation worth having. This pass is nearly
     * certain to arrive, so every geometric measure likes it, and it is still
     * not worth playing: it gives up the position it started from and buys
     * nothing. `progressionM` alone calls it neutral.
     */
    const value = valueOfPass({ x: -30, y: 5 }, { x: -32, y: -10 }, 1.2, 15, RIGHT);
    expect(value.completion).toBeGreaterThan(0.9);
    expect(value.expectedXT).toBeLessThan(0);
  });

  it("prefers a contested pass into the box over a safe one in front of it", () => {
    const risky = valueOfPass({ x: 20, y: 0 }, { x: 44, y: 6 }, 0.05, 25, RIGHT);
    const safe = valueOfPass({ x: 20, y: 0 }, { x: 16, y: 20 }, 1.5, 21, RIGHT);

    expect(risky.completion).toBeLessThan(safe.completion);
    expect(risky.expectedXT).toBeGreaterThan(safe.expectedXT);
  });

  it("discounts the reward by exactly the completion probability", () => {
    const v = valueOfPass({ x: 0, y: 0 }, { x: 40, y: 0 }, 0.2, 40, RIGHT);
    expect(v.expectedXT).toBeCloseTo(v.completion * v.toXT - v.fromXT, 12);
    expect(v.rewardXT).toBeCloseTo(v.toXT - v.fromXT, 12);
  });

  it("values a blocked pass into a great position below a certain one into the same place", () => {
    const blocked = valueOfPass({ x: 10, y: 0 }, { x: 44, y: 0 }, -1.2, 34, RIGHT);
    const clear = valueOfPass({ x: 10, y: 0 }, { x: 44, y: 0 }, 1.5, 34, RIGHT);
    expect(blocked.expectedXT).toBeLessThan(clear.expectedXT);
    expect(blocked.rewardXT).toBeCloseTo(clear.rewardXT, 12);
  });
});

describe("offBallThreat", () => {
  const players: PlayerState[] = [
    { id: 1, team: "team_a", x: -30, y: 0 },
    { id: 2, team: "team_a", x: 40, y: 4 },
    { id: 3, team: "team_a", x: 5, y: -20 },
    { id: 9, team: "team_b", x: 42, y: 0 },
  ];

  it("ranks the player in the most valuable space first", () => {
    const ranked = offBallThreat(players, "team_a", { x: -30, y: 0 }, 1, RIGHT);
    expect(ranked[0].playerId).toBe(2);
  });

  it("leaves out the carrier and the opposition", () => {
    const ranked = offBallThreat(players, "team_a", { x: -30, y: 0 }, 1, RIGHT);
    expect(ranked.map((r) => r.playerId).sort()).toEqual([2, 3]);
  });

  it("measures the gain against the ball, so it can be negative", () => {
    // Ball already at the edge of the box; a team mate in his own half is
    // standing somewhere worth less than where the ball already is.
    const ranked = offBallThreat(players, "team_a", { x: 40, y: 0 }, 2, RIGHT);
    const deep = ranked.find((r) => r.playerId === 1)!;
    expect(deep.gainOverBall).toBeLessThan(0);
  });
});
