/**
 * Tests for team shape, focused on the offside line.
 *
 * The offside line is the metric most sensitive to who counts as a defender.
 * Including the goalkeeper puts it roughly the length of the penalty area too
 * deep, which silently changes who is judged to be in behind.
 */

import { describe, expect, it } from "vitest";

import { analyseShape, playersBeyondLine } from "./shape";
import type { PlayerState } from "./types";

function player(id: number, x: number, y: number, team: PlayerState["team"] = "team_b"): PlayerState {
  return { id, team, x, y, vx: 0, vy: 0, label: String(id) };
}

// A back four at x=38 defending the right-hand goal, plus a midfield band.
const backFour = [-15, -5, 5, 15].map((y, i) => player(i, 38, y));
const midFour = [-18, -6, 6, 18].map((y, i) => player(10 + i, 28, y));
const defenders = [...backFour, ...midFour];

describe("offside line", () => {
  it("uses the deepest outfield defender when a keeper is tracked", () => {
    const keeper = player(99, 49, 0, "keeper");
    const shape = analyseShape(defenders, "team_b", 52.5, [keeper]);
    expect(shape).not.toBeNull();
    expect(shape!.offsideLineUsesKeeper).toBe(true);
    // The back four sit at 38, and the keeper must not drag the line to 49.
    expect(shape!.offsideLineX).toBeCloseTo(38, 5);
  });

  it("falls back to the second deepest when no keeper is identified", () => {
    // Without a keeper label the deepest player might BE the keeper, so the
    // second deepest is the safer guess.
    const withKeeperUnlabelled = [...defenders, player(99, 49, 0, "team_b")];
    const shape = analyseShape(withKeeperUnlabelled, "team_b", 52.5, []);
    expect(shape!.offsideLineUsesKeeper).toBe(false);
    expect(shape!.offsideLineX).toBeCloseTo(38, 5);
  });

  it("does not let a keeper at the far end count as this team's keeper", () => {
    // A keeper defending the opposite goal is irrelevant to this block, and
    // treating them as ours would take the line from the back four.
    const farKeeper = player(98, -49, 0, "keeper");
    const shape = analyseShape(defenders, "team_b", 52.5, [farKeeper]);
    expect(shape!.offsideLineUsesKeeper).toBe(false);
  });

  it("changes who is judged to be in behind", () => {
    const keeper = player(99, 49, 0, "keeper");
    const runner = player(50, 42, 0, "team_a");

    const withKeeper = analyseShape(defenders, "team_b", 52.5, [keeper])!;
    expect(playersBeyondLine([runner], withKeeper, 52.5)).toEqual([50]);

    // If the keeper had been counted as a defender the line would sit at 49,
    // and a runner at 42 would wrongly be judged onside.
    const keeperAsDefender = analyseShape(
      [...defenders, player(99, 49, 0, "team_b")],
      "team_b",
      52.5,
      [],
    )!;
    expect(keeperAsDefender.offsideLineX).toBeLessThan(49);
  });
});

describe("block shape", () => {
  it("excludes nobody it was given and reports a sane depth", () => {
    const shape = analyseShape(defenders, "team_b", 52.5, [])!;
    expect(shape.playerCount).toBe(8);
    expect(shape.blockDepthM).toBeCloseTo(10, 0);
    expect(shape.widthM).toBeCloseTo(36, 0);
  });

  it("returns null for too few players to have a shape", () => {
    expect(analyseShape(defenders.slice(0, 2), "team_b", 52.5)).toBeNull();
  });
});
