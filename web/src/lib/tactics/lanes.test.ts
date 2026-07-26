/**
 * Tests for the passing lane solver.
 *
 * The cases below are chosen to pin the behaviour that separates this model
 * from a distance heuristic. Two of them (`a distant defender still shuts a
 * long lane` and `a very close defender cannot stop a short sharp pass`) are
 * situations where perpendicular distance gives the opposite answer to the
 * interception race, and the race is right.
 */

import { describe, expect, it } from "vitest";

import { analysePassingLanes, interceptionMargin, scoreLane } from "./lanes";
import { DEFAULT_MOTION, type PassingLane, type PlayerState } from "./types";

function player(id: number, x: number, y: number, team: PlayerState["team"] = "team_a"): PlayerState {
  return { id, team, x, y, vx: 0, vy: 0, label: String(id) };
}

describe("interceptionMargin", () => {
  it("reports a large margin when no defender is anywhere near", () => {
    const { marginS } = interceptionMargin(
      { x: 0, y: 0 },
      { x: 20, y: 0 },
      player(9, 0, 45, "team_b"),
    );
    expect(marginS).toBeGreaterThan(1);
  });

  it("reports a negative margin when a defender sits on the line with time to spare", () => {
    // 40m pass at 16 m/s takes 2.5s. A defender 8m off the midpoint needs
    // roughly 0.28s reaction plus ~1s of running, so the ball loses.
    const { marginS } = interceptionMargin(
      { x: -20, y: 0 },
      { x: 20, y: 0 },
      player(9, 0, 8, "team_b"),
    );
    expect(marginS).toBeLessThan(0);
  });

  it("locates the pinch point where the defender is most dangerous", () => {
    const { atDistanceM } = interceptionMargin(
      { x: 0, y: 0 },
      { x: 40, y: 0 },
      player(9, 30, 3, "team_b"),
    );
    // The defender stands beside x = 30, which is 30m along a 40m lane.
    expect(atDistanceM).toBeGreaterThan(20);
    expect(atDistanceM).toBeLessThan(40);
  });

  it("a distant defender still shuts a long lane", () => {
    // 12m of perpendicular clearance looks safe by any distance heuristic, but
    // over a 60m pass the defender has ~3.75s to cover it and gets there easily.
    const from = { x: -30, y: 0 };
    const to = { x: 30, y: 0 };
    const defender = player(9, 0, 12, "team_b");

    const { marginS } = interceptionMargin(from, to, defender);
    expect(marginS).toBeLessThan(0);
  });

  it("a very close defender cannot stop a short sharp pass", () => {
    // 2m of clearance looks fatal, but the ball covers this 8m pass in 0.5s and
    // the defender is still inside their reaction time for most of it.
    const { marginS } = interceptionMargin(
      { x: 0, y: 0 },
      { x: 8, y: 0 },
      player(9, 4, 2.4, "team_b"),
    );
    expect(marginS).toBeGreaterThan(0);
  });

  it("accounts for momentum: a defender sprinting away arrives later", () => {
    const from = { x: 0, y: 0 };
    const to = { x: 25, y: 0 };
    const stationary: PlayerState = { ...player(9, 12, 5, "team_b"), vx: 0, vy: 0 };
    const retreating: PlayerState = { ...stationary, vy: 7 };

    const a = interceptionMargin(from, to, stationary).marginS;
    const b = interceptionMargin(from, to, retreating).marginS;
    expect(b).toBeGreaterThan(a);
  });

  it("treats a zero length pass as uncontestable rather than dividing by zero", () => {
    const { marginS } = interceptionMargin(
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      player(9, 5, 5, "team_b"),
    );
    expect(Number.isFinite(marginS) || marginS === Infinity).toBe(true);
  });
});

describe("analysePassingLanes", () => {
  const carrier = player(1, 0, 0);

  it("classifies a completely unmarked short option as open", () => {
    const lanes = analysePassingLanes(carrier, [carrier, player(2, 12, 0)], [
      player(9, -40, 30, "team_b"),
    ]);
    expect(lanes).toHaveLength(1);
    expect(lanes[0].verdict).toBe("open");
  });

  it("classifies a lane with a defender parked on it as blocked", () => {
    const lanes = analysePassingLanes(carrier, [carrier, player(2, 40, 0)], [
      player(9, 20, 0.5, "team_b"),
    ]);
    expect(lanes[0].verdict).toBe("blocked");
    expect(lanes[0].safetyMarginS).toBeLessThan(0);
  });

  it("skips teammates who are effectively on top of the carrier", () => {
    const lanes = analysePassingLanes(carrier, [carrier, player(2, 0.5, 0.2)], []);
    expect(lanes).toHaveLength(0);
  });

  it("counts defenders the pass plays beyond", () => {
    const lanes = analysePassingLanes(
      carrier,
      [carrier, player(2, 40, 0)],
      [player(9, 10, 20, "team_b"), player(10, 20, -20, "team_b"), player(11, 60, 0, "team_b")],
      DEFAULT_MOTION,
      52.5,
    );
    // The two defenders between carrier and receiver are bypassed; the one
    // beyond the receiver is not.
    expect(lanes[0].defendersBypassed).toBe(2);
  });

  it("ranks a progressive option above a safe backward one", () => {
    const lanes = analysePassingLanes(
      carrier,
      [carrier, player(2, 22, 2), player(3, -25, 0)],
      [player(9, 5, 25, "team_b")],
      DEFAULT_MOTION,
      52.5,
    );
    expect(lanes[0].targetId).toBe(2);
  });

  it("returns no lanes when there is nobody to pass to", () => {
    expect(analysePassingLanes(carrier, [carrier], [player(9, 5, 5, "team_b")])).toHaveLength(0);
  });

  it("handles an empty defence without producing NaN", () => {
    const lanes = analysePassingLanes(carrier, [carrier, player(2, 20, 0)], []);
    expect(lanes[0].verdict).toBe("open");
    expect(Number.isNaN(lanes[0].score)).toBe(false);
  });
});

describe("scoreLane", () => {
  const base: PassingLane = {
    targetId: 2,
    from: { x: 0, y: 0 },
    to: { x: 20, y: 0 },
    distanceM: 20,
    safetyMarginS: 0.8,
    threatId: 9,
    threatPointM: 10,
    minClearanceM: 6,
    progressionM: 20,
    defendersBypassed: 2,
    receiverPressureM: 10,
    verdict: "open",
    score: 0,
  };

  it("scores an open progressive lane highly", () => {
    expect(scoreLane(base)).toBeGreaterThan(0.6);
  });

  it("collapses the score for a blocked lane however attractive the destination", () => {
    const blocked = { ...base, safetyMarginS: -0.5, verdict: "blocked" as const };
    expect(scoreLane(blocked)).toBeLessThan(0.15);
  });

  it("never leaves the unit interval", () => {
    const extreme = {
      ...base,
      safetyMarginS: Infinity,
      progressionM: 500,
      defendersBypassed: 50,
      receiverPressureM: Infinity,
    };
    const score = scoreLane(extreme);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});
