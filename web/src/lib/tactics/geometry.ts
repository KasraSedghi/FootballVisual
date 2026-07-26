/**
 * Geometry primitives used by the tactical engine.
 *
 * Kept separate from the tactical reasoning so each piece can be tested on its
 * own terms: a convex hull is either right or wrong regardless of what a
 * defensive block is.
 */

import type { Vec2 } from "./types";

export const PITCH_LENGTH = 105;
export const PITCH_WIDTH = 68;
export const HALF_LENGTH = PITCH_LENGTH / 2;
export const HALF_WIDTH = PITCH_WIDTH / 2;

export function dist(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function scale(a: Vec2, k: number): Vec2 {
  return { x: a.x * k, y: a.y * k };
}

export function length(a: Vec2): number {
  return Math.hypot(a.x, a.y);
}

export function normalise(a: Vec2): Vec2 {
  const l = length(a);
  return l < 1e-9 ? { x: 0, y: 0 } : { x: a.x / l, y: a.y / l };
}

/**
 * Perpendicular distance from `p` to the segment `a`->`b`.
 *
 * Segment rather than infinite line, because a defender standing well behind
 * the passer is not in the lane no matter how well aligned they are with it.
 */
export function pointToSegment(p: Vec2, a: Vec2, b: Vec2): number {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const lenSq = abx * abx + aby * aby;
  if (lenSq < 1e-12) return dist(p, a);
  let t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby));
}

/**
 * Convex hull by Andrew's monotone chain, counter-clockwise, no collinear points.
 *
 * Used for team shape area. Duplicated points are common when two tracked
 * players briefly share a position, and the strict cross-product test below
 * discards them rather than producing a degenerate hull edge.
 */
export function convexHull(points: Vec2[]): Vec2[] {
  if (points.length < 3) return [...points];
  const pts = [...points].sort((p, q) => (p.x === q.x ? p.y - q.y : p.x - q.x));

  const cross = (o: Vec2, a: Vec2, b: Vec2) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Vec2[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Vec2[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Shoelace area of a polygon, always non-negative. */
export function polygonArea(poly: Vec2[]): number {
  if (poly.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

export function clampToPitch(p: Vec2): Vec2 {
  return {
    x: Math.max(-HALF_LENGTH, Math.min(HALF_LENGTH, p.x)),
    y: Math.max(-HALF_WIDTH, Math.min(HALF_WIDTH, p.y)),
  };
}

export function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}
