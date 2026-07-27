import { describe, expect, it } from "vitest";

import { parseQueryHeuristically } from "./route";

/**
 * The no-API-key path.
 *
 * This is the default configuration of the repository, so it is not a fallback
 * in any meaningful sense: it is what most people running this will actually
 * get. It deserves the same testing as the model path would.
 */
describe("parseQueryHeuristically", () => {
  it("reads a compact block as an upper bound on width", () => {
    expect(parseQueryHeuristically("a compact block")).toEqual({
      maxBlockWidthM: 32,
    });
  });

  it("reads a stretched block as a lower bound", () => {
    const q = parseQueryHeuristically("when were they stretched");
    expect(q.minBlockWidthM).toBeGreaterThan(0);
    expect(q.maxBlockWidthM).toBeUndefined();
  });

  it("prefers a distance the analyst actually gave over the default", () => {
    // "wider than 40m" has to mean 40, not the generic 42. Silently substituting
    // a different threshold answers a question that was not asked.
    expect(parseQueryHeuristically("stretched wider than 40m").minBlockWidthM).toBe(40);
    expect(parseQueryHeuristically("a gap of 18m in the last line").minBackLineGapM).toBe(18);
  });

  it("recognises the tactical phrases that name a measurement", () => {
    expect(parseQueryHeuristically("players between the lines")).toEqual({
      minPlayersBetweenLines: 1,
    });
    expect(parseQueryHeuristically("someone in behind").minPlayersBeyondLine).toBe(1);
    expect(parseQueryHeuristically("a gap in the back line").minBackLineGapM).toBe(12);
  });

  it("combines several phrases into one query", () => {
    const q = parseQueryHeuristically("a compact block with an open lane");
    expect(q.maxBlockWidthM).toBe(32);
    expect(q.minOpenLanes).toBe(1);
  });

  it("never returns a query with no filters", () => {
    /*
     * An empty query matches every frame, which surfaces as "here is the whole
     * clip" and reads as a broken search rather than as an unparsed question.
     * Falling back to something narrow is the honest failure.
     */
    const q = parseQueryHeuristically("hjkl qwerty zxcv");
    expect(Object.keys(q).length).toBeGreaterThan(0);
  });

  it("is case insensitive", () => {
    expect(parseQueryHeuristically("A COMPACT BLOCK")).toEqual(
      parseQueryHeuristically("a compact block"),
    );
  });

  it("does not set contradictory width bounds from one phrase", () => {
    // "compact" and "wide" are opposites; a query carrying both can never match
    // and would report an empty clip rather than a bad parse.
    const q = parseQueryHeuristically("a compact block");
    expect(q.minBlockWidthM === undefined || q.maxBlockWidthM === undefined).toBe(true);
  });
});
