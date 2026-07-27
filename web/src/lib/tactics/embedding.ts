/**
 * Turning a frame into a vector, so "find moments like this one" works.
 *
 * The threshold search in `search.ts` answers a question you already know how to
 * ask: wider than 40m, a gap over 12m. An analyst watching a clip usually has
 * the opposite problem. They are looking at a shape on screen, they know it
 * matters, and they cannot name the three numbers that would find it again. So
 * this maps each frame to a point in a vector space where tactically similar
 * frames land near each other, and retrieval becomes nearest neighbour.
 *
 * This is the retrieval layer of the architecture Arsenal's research group
 * described, built the honest way round. Theirs learns the representation with a
 * spatiotemporal transformer trained on thousands of match hours. This one
 * computes the features by hand, because ten matches of open tracking data is
 * nowhere near enough to learn one, and a handcrafted embedding that works today
 * beats a learned one that needs a dataset this project does not have.
 *
 * What is genuinely the same idea is everything around the representation: the
 * canonicalisation below, cosine similarity over a shape descriptor, and
 * retrieval by nearest neighbour rather than by threshold. Swapping the
 * handcrafted vector for a learned one later changes `embedFrame` and nothing
 * else.
 */

import type { PlayerState, Snapshot, TacticalReport, Vec2 } from "./types";
import { HALF_LENGTH, HALF_WIDTH } from "../pitch";

/**
 * Occupancy grid resolution, in cells along the pitch and across it.
 *
 * Coarse on purpose. The point is to capture the *shape* a team is holding, and
 * a fine grid records exactly which cell each player stood in, so two frames of
 * the same block with one player a metre over stop matching. Six by four is
 * roughly thirds and channels, which is how the shape gets described out loud.
 */
const GRID_X = 6;
const GRID_Y = 4;

/**
 * Weight on the derived tactical scalars relative to the occupancy grid.
 *
 * Without it the grid dominates by sheer count: 48 occupancy numbers against a
 * dozen scalars means two frames match on where bodies are while disagreeing
 * about whether there is a gap to play through. The scalars carry the tactical
 * reading, so they are scaled up to weigh comparably.
 */
const SCALAR_WEIGHT = 2.5;

export interface FrameEmbedding {
  frame: number;
  timeS: number;
  index: number;
  /** Unit-length, so cosine similarity is a dot product. */
  vector: number[];
}

export interface SimilarFrame {
  frame: number;
  timeS: number;
  index: number;
  /** Cosine similarity in [-1, 1]; 1 is identical. */
  similarity: number;
}

/**
 * Put a frame into a canonical orientation before measuring anything.
 *
 * This is the geometric-equivariance step, and without it retrieval mostly does
 * not work. The same tactical situation played toward the other goal, or mirrored
 * across the halfway line, is the same situation, and an analyst asking for
 * "moments like this" means those too. Raw coordinates say otherwise: a build-up
 * on the left at one end and its mirror at the other end share almost no
 * numbers, so they land far apart in any space built directly on positions.
 *
 * Two reflections fix it, and both are exact rather than approximations:
 *
 * 1. Flip x so the team in possession always attacks toward +x. Direction of
 *    play is a property of the fixture, not of the tactics.
 * 2. Flip y so the ball is always in the +y half. Left and right wing versions
 *    of the same overload are the same overload.
 *
 * Reflection preserves every distance and angle, so nothing measured downstream
 * is distorted, only relabelled.
 */
export function canonicalise(snapshot: Snapshot, attackingGoalX: number): Snapshot {
  const flipX = attackingGoalX < 0 ? -1 : 1;
  // Decided after the x flip, which does not move y, so the two are independent.
  const flipY = (snapshot.ball?.y ?? 0) < 0 ? -1 : 1;

  const move = (p: Vec2): Vec2 => ({ x: p.x * flipX, y: p.y * flipY });

  return {
    ...snapshot,
    players: snapshot.players.map(
      (p): PlayerState => ({ ...p, x: p.x * flipX, y: p.y * flipY }),
    ),
    ball: snapshot.ball ? move(snapshot.ball) : null,
  };
}

/** Normalised occupancy grid for one set of players, summing to 1. */
function occupancy(players: PlayerState[]): number[] {
  const cells = new Array<number>(GRID_X * GRID_Y).fill(0);
  if (!players.length) return cells;

  for (const p of players) {
    // Clamp rather than drop: a player a metre off the touchline in the
    // tracking data is still in the widest channel, not absent from the shape.
    const fx = Math.min(0.999, Math.max(0, (p.x + HALF_LENGTH) / (2 * HALF_LENGTH)));
    const fy = Math.min(0.999, Math.max(0, (p.y + HALF_WIDTH) / (2 * HALF_WIDTH)));
    const cx = Math.floor(fx * GRID_X);
    const cy = Math.floor(fy * GRID_Y);
    cells[cy * GRID_X + cx] += 1;
  }

  return cells.map((c) => c / players.length);
}

/**
 * The tactical scalars, each scaled to roughly [0, 1] so none dominates.
 *
 * Raw metres would let hull area, which runs to the thousands, swamp a lane
 * margin measured in tenths of a second. The divisors are the plausible maximum
 * for each quantity rather than the observed maximum in this clip, so an
 * embedding computed on one clip is comparable to one computed on another.
 */
function scalars(report: TacticalReport): number[] {
  const block = report.block;
  const attack = report.attackingShape;
  const openLanes = report.lanes.filter((l) => l.verdict === "open").length;
  const bestMargin = report.lanes
    .map((l) => l.safetyMarginS)
    .filter((m) => Number.isFinite(m))
    .reduce((a, b) => Math.max(a, b), 0);

  return [
    (block?.widthM ?? 0) / (2 * HALF_WIDTH),
    (block?.blockDepthM ?? 0) / 60,
    (block?.hullAreaM2 ?? 0) / 3000,
    ((block?.defensiveLineX ?? 0) + HALF_LENGTH) / (2 * HALF_LENGTH),
    (block?.largestBackLineGapM ?? 0) / (2 * HALF_WIDTH),
    ((block?.backLineGapCentreY ?? 0) + HALF_WIDTH) / (2 * HALF_WIDTH),
    (attack?.widthM ?? 0) / (2 * HALF_WIDTH),
    (attack?.blockDepthM ?? 0) / 60,
    Math.min(1, openLanes / 5),
    Math.min(1, Math.max(0, bestMargin) / 2),
    Math.min(1, report.playersBetweenLines.length / 4),
    Math.min(1, report.playersBeyondLine.length / 4),
    report.space?.teamAShare ?? 0.5,
  ];
}

/**
 * Build the vector for one already-analysed frame.
 *
 * Takes the report rather than the snapshot because every tactical scalar in it
 * is already computed there, and recomputing them would risk the embedding and
 * the displayed numbers drifting apart.
 */
export function embedFrame(report: TacticalReport): number[] {
  const canonical = canonicalise(report.snapshot, report.attackingGoalX);

  const attackers = canonical.players.filter((p) => p.team === report.attackingTeam);
  const defenders = canonical.players.filter((p) => p.team === report.defendingTeam);

  const ball = canonical.ball;
  const ballFeatures = ball
    ? [
        (ball.x + HALF_LENGTH) / (2 * HALF_LENGTH),
        (ball.y + HALF_WIDTH) / (2 * HALF_WIDTH),
      ]
    : [0.5, 0.5];

  const raw = [
    ...occupancy(attackers),
    ...occupancy(defenders),
    ...ballFeatures.map((v) => v * SCALAR_WEIGHT),
    ...scalars(report).map((v) => v * SCALAR_WEIGHT),
  ];

  // Unit length, so similarity is direction only. Two frames with the same
  // shape should match whether or not one happens to have larger magnitudes,
  // and normalising here means the comparison is a plain dot product.
  const norm = Math.sqrt(raw.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? raw.map((v) => v / norm) : raw;
}

/** Cosine similarity of two unit vectors, which is their dot product. */
export function similarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i += 1) dot += a[i] * b[i];
  return dot;
}

/**
 * Find the frames most like a target frame.
 *
 * `minSeparationFrames` is what makes the result readable rather than correct
 * but useless. The nearest neighbours of frame 120 are frames 119 and 121, which
 * are the same moment and tell an analyst nothing. Requiring returned frames to
 * be spaced apart turns the answer into "other times this happened" rather than
 * "the frames on either side of the one you picked".
 */
export function findSimilar(
  target: number[],
  embeddings: FrameEmbedding[],
  options: { limit?: number; minSeparationFrames?: number; excludeIndex?: number } = {},
): SimilarFrame[] {
  const { limit = 5, minSeparationFrames = 25, excludeIndex } = options;

  const scored = embeddings
    .map((e) => ({
      frame: e.frame,
      timeS: e.timeS,
      index: e.index,
      similarity: similarity(target, e.vector),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  const chosen: SimilarFrame[] = [];
  for (const candidate of scored) {
    if (chosen.length >= limit) break;
    if (excludeIndex !== undefined && Math.abs(candidate.index - excludeIndex) < minSeparationFrames) {
      continue;
    }
    if (chosen.some((c) => Math.abs(c.index - candidate.index) < minSeparationFrames)) {
      continue;
    }
    chosen.push(candidate);
  }

  return chosen;
}
