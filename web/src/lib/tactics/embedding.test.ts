import { describe, expect, it } from "vitest";

import { analyseSnapshot } from "./index";
import {
  canonicalise,
  embedFrame,
  findSimilar,
  similarity,
  type FrameEmbedding,
} from "./embedding";
import type { Snapshot } from "./types";

/**
 * A build-up: attackers left of centre, a defensive block ahead of them.
 *
 * `mirror` reflects the whole thing about the halfway line, which is the same
 * situation played toward the other goal. `flip` reflects it across the pitch's
 * long axis, which is the same situation down the other wing. Both must embed
 * to the same place.
 */
function scene(frame = 0, mirror = false, flip = false): Snapshot {
  const mx = mirror ? -1 : 1;
  const fy = flip ? -1 : 1;
  const at = (x: number, y: number) => ({ x: x * mx, y: y * fy });

  const attackers = [
    { id: 1, ...at(-20, 5) },
    { id: 2, ...at(-10, 18) },
    { id: 3, ...at(-5, -12) },
    { id: 4, ...at(5, 8) },
  ];
  const defenders = [
    { id: 10, ...at(20, -15) },
    { id: 11, ...at(22, -3) },
    { id: 12, ...at(21, 9) },
    { id: 13, ...at(24, 20) },
  ];

  return {
    frame,
    timeS: frame / 25,
    ball: at(-20, 5),
    players: [
      ...attackers.map((p) => ({ ...p, team: "team_a" as const, label: String(p.id) })),
      ...defenders.map((p) => ({ ...p, team: "team_b" as const, label: String(p.id) })),
    ],
  };
}

/**
 * Embed a scene. The direction of play comes off the report, so these tests
 * exercise the same path the app does rather than a hand-passed goal side.
 */
function embed(s: Snapshot): number[] {
  const report = analyseSnapshot(s, { computeSpace: false });
  if (!report) throw new Error("scene did not analyse");
  return embedFrame(report);
}

describe("canonicalise", () => {
  it("reflects, so distances between players are unchanged", () => {
    /*
     * The whole approach rests on this. A reflection is an isometry, so
     * canonicalising cannot distort the shape it is about to measure, only
     * relabel where it sits. If this ever fails, every embedding is measuring a
     * warped version of the frame.
     */
    const s = scene();
    const c = canonicalise(s, -52.5);

    const gap = (snap: Snapshot, a: number, b: number) => {
      const pa = snap.players.find((p) => p.id === a)!;
      const pb = snap.players.find((p) => p.id === b)!;
      return Math.hypot(pa.x - pb.x, pa.y - pb.y);
    };

    expect(gap(c, 1, 10)).toBeCloseTo(gap(s, 1, 10), 6);
    expect(gap(c, 2, 13)).toBeCloseTo(gap(s, 2, 13), 6);
  });

  it("puts the ball in the positive y half whichever wing it started on", () => {
    expect(canonicalise(scene(0, false, false), 52.5).ball!.y).toBeGreaterThan(0);
    expect(canonicalise(scene(0, false, true), 52.5).ball!.y).toBeGreaterThan(0);
  });

  it("leaves a frame with no ball alone rather than throwing", () => {
    const noBall: Snapshot = { ...scene(), ball: null };
    expect(canonicalise(noBall, 52.5).ball).toBeNull();
  });
});

describe("embedFrame", () => {
  it("returns a unit vector", () => {
    const v = embed(scene());
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it("gives the same situation at the other end nearly the same embedding", () => {
    /*
     * The equivariance that makes retrieval useful. Without canonicalisation
     * these two are far apart in the vector space and an analyst asking for
     * "moments like this" never sees the second half of the match.
     */
    const first = embed(scene(0, false));
    const other = embed(scene(0, true));
    expect(similarity(first, other)).toBeGreaterThan(0.97);
  });

  it("gives the same situation on the other wing nearly the same embedding", () => {
    const left = embed(scene(0, false, false));
    const right = embed(scene(0, false, true));
    expect(similarity(left, right)).toBeGreaterThan(0.97);
  });

  it("separates a compact block from a stretched one", () => {
    /*
     * The other half of the requirement: invariant to orientation, but not
     * invariant to tactics. An embedding that made everything similar would
     * pass the two tests above and be worthless.
     */
    const compact = scene();
    const stretched: Snapshot = {
      ...compact,
      players: compact.players.map((p) =>
        p.team === "team_b" ? { ...p, y: p.y * 2.5 } : p,
      ),
    };

    const s = similarity(embed(compact), embed(stretched));
    expect(s).toBeLessThan(0.97);
  });
});

describe("findSimilar", () => {
  function embedding(index: number, vector: number[]): FrameEmbedding {
    return { frame: index, timeS: index / 25, index, vector };
  }

  const a = [1, 0, 0];
  const b = [0.9, 0.436, 0];
  const c = [0, 1, 0];

  it("ranks by similarity", () => {
    const results = findSimilar(
      a,
      [embedding(0, c), embedding(100, b), embedding(200, a)],
      { minSeparationFrames: 1 },
    );
    expect(results[0].index).toBe(200);
    expect(results[1].index).toBe(100);
  });

  it("does not return the frames either side of the one you picked", () => {
    /*
     * The nearest neighbours of frame 120 are frames 119 and 121. They are the
     * same moment, they are trivially similar, and returning them makes the
     * feature useless. This spacing is what turns the answer into "other times
     * this happened".
     */
    const embeddings = Array.from({ length: 60 }, (_, i) => embedding(i, a));
    const results = findSimilar(a, embeddings, {
      excludeIndex: 30,
      minSeparationFrames: 25,
      limit: 5,
    });

    for (const r of results) {
      expect(Math.abs(r.index - 30)).toBeGreaterThanOrEqual(25);
    }
  });

  it("spaces results apart from each other, not just from the query", () => {
    const embeddings = Array.from({ length: 200 }, (_, i) => embedding(i, a));
    const results = findSimilar(a, embeddings, { minSeparationFrames: 25, limit: 5 });

    for (let i = 1; i < results.length; i += 1) {
      expect(Math.abs(results[i].index - results[i - 1].index)).toBeGreaterThanOrEqual(25);
    }
  });

  it("returns nothing when there is nothing to compare against", () => {
    expect(findSimilar(a, [])).toEqual([]);
  });
});

describe("similarity", () => {
  it("is 1 for identical vectors and 0 for orthogonal ones", () => {
    expect(similarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(similarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("refuses to compare vectors of different lengths", () => {
    expect(similarity([1, 0], [1, 0, 0])).toBe(0);
  });
});
