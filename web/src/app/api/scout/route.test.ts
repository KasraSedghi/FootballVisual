import { describe, expect, it } from "vitest";

import { MAX_TURNS, runTool } from "./route";
import type { Snapshot } from "@/lib/tactics";

/**
 * The scout's tools are the whole safety story.
 *
 * The agent loop is free to call whatever it likes in whatever order, and that
 * is fine precisely because every tool is a call into the deterministic engine.
 * So what is worth testing is not the loop, it is that a tool cannot be talked
 * into returning something the engine did not compute.
 */

function snapshot(frame: number, defenderY: number[] = [-20, 0, 20]): Snapshot {
  return {
    frame,
    timeS: frame / 25,
    ball: { x: -20, y: 0 },
    players: [
      { id: 1, team: "team_a", x: -20, y: 0, label: "1" },
      { id: 2, team: "team_a", x: 20, y: 0, label: "2" },
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

const clip = [0, 1, 2, 3, 4].map((f) => snapshot(f));

describe("runTool", () => {
  it("refuses a tool name it does not implement", () => {
    /*
     * The model picks tool names. A typo, or a name it invented because it
     * seemed plausible, must come back as an error it can read and recover
     * from, never as a silent no-op that looks like an empty result.
     */
    const result = runTool("delete_everything", {}, clip) as { error?: string };
    expect(result.error).toContain("no such tool");
  });

  it("reports a missing frame rather than guessing a nearby one", () => {
    /*
     * Returning the closest frame would be worse than an error: the model would
     * cite frame 999 in its report while the measurements came from frame 4.
     */
    const result = runTool("inspect_frame", { frame: 999 }, clip) as { error?: string };
    expect(result.error).toContain("999");
  });

  it("returns measurements, not prose, from inspect_frame", () => {
    const result = runTool("inspect_frame", { frame: 2 }, clip) as {
      frame: number;
      block: { widthM: number } | null;
      lanes: unknown[];
    };
    expect(result.frame).toBe(2);
    expect(result.block?.widthM).toBeGreaterThan(0);
    expect(Array.isArray(result.lanes)).toBe(true);
  });

  it("caps how many passages one search can return", () => {
    /*
     * An unfiltered search over a long clip would otherwise fill the context
     * window with near-identical passages and leave no room for the
     * investigation that was supposed to follow it.
     */
    const long = Array.from({ length: 400 }, (_, f) => snapshot(f * 10));
    const result = runTool("search_moments", {}, long) as { passages: unknown[] };
    expect(result.passages.length).toBeLessThanOrEqual(6);
  });

  it("tells the model how its query was interpreted", () => {
    // The model set thresholds; echoing back the reading of them is what lets
    // it notice it asked for something other than what it meant.
    const result = runTool("search_moments", { minBackLineGapM: 12 }, clip) as {
      interpretedAs: string;
    };
    expect(result.interpretedAs).toContain("12m");
  });

  it("reports an empty search honestly rather than relaxing the query", () => {
    /*
     * Silently loosening an over-specified query would hand back passages that
     * do not have the property asked for, which is the one thing this whole
     * design exists to prevent.
     */
    const result = runTool("search_moments", { minBlockWidthM: 999 }, clip) as {
      matchCount: number;
      passages: unknown[];
    };
    expect(result.matchCount).toBe(0);
    expect(result.passages).toEqual([]);
  });

  it("summarises the clip without needing a query", () => {
    const result = runTool("clip_summary", {}, clip) as {
      frames: number;
      framesInPossession: Record<string, number>;
    };
    expect(result.frames).toBe(5);
    expect(Object.keys(result.framesInPossession).length).toBeGreaterThan(0);
  });

  it("bounds the loop so a wandering investigation cannot run forever", () => {
    expect(MAX_TURNS).toBeGreaterThan(1);
    expect(MAX_TURNS).toBeLessThanOrEqual(10);
  });
});
