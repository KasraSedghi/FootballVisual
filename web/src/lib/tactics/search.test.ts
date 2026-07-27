import { describe, expect, it } from "vitest";

import {
  describeQuery,
  groupIntoSequences,
  searchFrames,
  type SearchHit,
} from "./search";
import type { Snapshot } from "./types";

/**
 * A frame with a carrier, a target, and `defenders` spread across the pitch.
 *
 * Positions are chosen so the engine has a real answer rather than a degenerate
 * one: the carrier holds the ball, the target is ahead of them, and defenders
 * sit between or beside depending on what the test needs.
 */
function snapshot(
  frame: number,
  opts: { defenderY?: number[]; targetX?: number } = {},
): Snapshot {
  const { defenderY = [-20, 0, 20], targetX = 20 } = opts;
  return {
    frame,
    timeS: frame / 25,
    ball: { x: -20, y: 0 },
    players: [
      { id: 1, team: "team_a", x: -20, y: 0, label: "1" },
      { id: 2, team: "team_a", x: targetX, y: 0, label: "2" },
      { id: 3, team: "team_a", x: 0, y: 25, label: "3" },
      ...defenderY.map((y, i) => ({
        id: 10 + i,
        team: "team_b" as const,
        x: 30,
        y,
        label: String(10 + i),
      })),
    ],
  };
}

function hit(index: number, frame: number, openLanes = 1): SearchHit {
  return {
    frame,
    timeS: frame / 25,
    index,
    measurements: {
      openLanes,
      bestMarginS: 1,
      blockWidthM: 40,
      blockDepthM: 10,
      backLineGapM: 15,
      playersBetweenLines: 0,
      playersBeyondLine: 0,
      attackingTeam: "team_a",
    },
  };
}

describe("searchFrames", () => {
  it("returns every frame when no filters are set", () => {
    const frames = [snapshot(0), snapshot(1), snapshot(2)];
    expect(searchFrames(frames, {}).length).toBe(3);
  });

  it("applies filters as AND, not OR", () => {
    const frames = [snapshot(0)];
    const measured = searchFrames(frames, {})[0].measurements;

    // One satisfiable filter and one that cannot be met. If these were ORed,
    // the frame would come back anyway, and a search would return moments that
    // do not have the property the analyst asked about.
    const impossible = (measured.blockWidthM ?? 0) + 1000;
    expect(
      searchFrames(frames, { minOpenLanes: 0, minBlockWidthM: impossible }),
    ).toHaveLength(0);
  });

  it("fails a filter whose measurement could not be computed", () => {
    /*
     * A frame with one team has no defensive block, so block width is null.
     * Treating that as "passes" would offer the frame as evidence of a width
     * that was never established on it.
     */
    const lonely: Snapshot = {
      frame: 0,
      timeS: 0,
      ball: { x: 0, y: 0 },
      players: [{ id: 1, team: "team_a", x: 0, y: 0, label: "1" }],
    };
    expect(searchFrames([lonely], { minBlockWidthM: 1 })).toHaveLength(0);
  });

  it("filters on the team in possession", () => {
    const frames = [snapshot(0)];
    expect(searchFrames(frames, { attackingTeam: "team_a" })).toHaveLength(1);
    expect(searchFrames(frames, { attackingTeam: "team_b" })).toHaveLength(0);
  });

  it("reports the measurements that justified each hit", () => {
    const hits = searchFrames([snapshot(0)], {});
    expect(hits[0].measurements.blockWidthM).toBeGreaterThan(0);
    expect(hits[0].index).toBe(0);
    expect(hits[0].frame).toBe(0);
  });
});

describe("groupIntoSequences", () => {
  it("collapses consecutive frames into one passage", () => {
    /*
     * The behaviour that makes results readable. At 25fps a two second passage
     * is fifty hits, and returning them individually buries every other moment
     * in the clip under one of them.
     */
    const hits = [0, 1, 2, 3, 4].map((i) => hit(i, i));
    const sequences = groupIntoSequences(hits);

    expect(sequences).toHaveLength(1);
    expect(sequences[0].frameCount).toBe(5);
    expect(sequences[0].startFrame).toBe(0);
    expect(sequences[0].endFrame).toBe(4);
  });

  it("bridges a short gap rather than splitting the passage", () => {
    // Tracking flickers: a player lost for two frames can drop the open-lane
    // count under the threshold. That is detection noise, not a new moment.
    const hits = [hit(0, 0), hit(1, 1), hit(4, 4), hit(5, 5)];
    expect(groupIntoSequences(hits, 5)).toHaveLength(1);
  });

  it("splits on a gap longer than the tolerance", () => {
    const hits = [hit(0, 0), hit(1, 1), hit(40, 40), hit(41, 41)];
    expect(groupIntoSequences(hits, 5)).toHaveLength(2);
  });

  it("picks the strongest frame in a run as the peak", () => {
    const hits = [hit(0, 0, 1), hit(1, 1, 5), hit(2, 2, 2)];
    expect(groupIntoSequences(hits)[0].peak.frame).toBe(1);
  });

  it("ranks passages so the most interesting one is first", () => {
    const weak = [hit(0, 0, 1), hit(1, 1, 1)];
    const strong = [hit(40, 40, 6), hit(41, 41, 6)];
    const sequences = groupIntoSequences([...weak, ...strong]);
    expect(sequences[0].peak.measurements.openLanes).toBe(6);
  });

  it("returns nothing for no hits", () => {
    expect(groupIntoSequences([])).toEqual([]);
  });
});

describe("describeQuery", () => {
  it("explains a query without a language model", () => {
    /*
     * A returned passage has to be able to explain itself with no API key, and
     * the explanation has to come from the structure that did the filtering
     * rather than from a model's account of it.
     */
    const text = describeQuery({ minOpenLanes: 2, maxBlockWidthM: 30 });
    expect(text).toContain("2 open passing lanes");
    expect(text).toContain("no wider than 30m");
  });

  it("says plainly when a query filters nothing", () => {
    expect(describeQuery({})).toContain("no filters");
  });

  it("uses the singular for one lane", () => {
    const text = describeQuery({ minOpenLanes: 1 });
    expect(text).toContain("1 open passing lane");
    expect(text).not.toContain("lanes");
  });
});
