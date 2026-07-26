/**
 * Pitch drawing geometry and the pitch-to-screen transform.
 *
 * These numbers mirror `pipeline/footballvisual/pitch.py`. They are duplicated
 * rather than imported because the two runtimes cannot share a module, and a
 * generated-constants build step would be more machinery than a dozen numbers
 * from the Laws of the Game deserve. The Python side is the source of truth;
 * `pitch.test.ts` pins these against the values it emits so the copies cannot
 * silently drift.
 */

export const PITCH_LENGTH = 105;
export const PITCH_WIDTH = 68;
export const HALF_LENGTH = PITCH_LENGTH / 2;
export const HALF_WIDTH = PITCH_WIDTH / 2;

export const CENTRE_CIRCLE_RADIUS = 9.15;
export const PENALTY_AREA_LENGTH = 16.5;
export const PENALTY_AREA_WIDTH = 40.32;
export const GOAL_AREA_LENGTH = 5.5;
export const GOAL_AREA_WIDTH = 18.32;
export const PENALTY_SPOT_DISTANCE = 11;
export const GOAL_WIDTH = 7.32;

export interface Transform {
  /** Pitch metres to screen pixels. */
  toScreen: (x: number, y: number) => { sx: number; sy: number };
  /** Screen pixels back to pitch metres, for hit testing and dragging. */
  toPitch: (sx: number, sy: number) => { x: number; y: number };
  scale: number;
  width: number;
  height: number;
}

/**
 * Build a transform that fits the pitch into a box with padding.
 *
 * The y axis is flipped. Pitch +y points at the far touchline, screen +y points
 * down, so without the flip every tactical map would be mirrored top to bottom
 * and every "left" in the analysis would appear on the right.
 */
export function makeTransform(width: number, height: number, padding = 16): Transform {
  const usableW = Math.max(1, width - padding * 2);
  const usableH = Math.max(1, height - padding * 2);
  const scale = Math.min(usableW / PITCH_LENGTH, usableH / PITCH_WIDTH);

  const offsetX = width / 2;
  const offsetY = height / 2;

  return {
    scale,
    width,
    height,
    toScreen: (x: number, y: number) => ({
      sx: offsetX + x * scale,
      sy: offsetY - y * scale,
    }),
    toPitch: (sx: number, sy: number) => ({
      x: (sx - offsetX) / scale,
      y: -(sy - offsetY) / scale,
    }),
  };
}

/** Pitch markings as polylines in metres, ready to stroke. */
export function pitchLines(): Array<Array<[number, number]>> {
  const lines: Array<Array<[number, number]>> = [];

  lines.push([
    [-HALF_LENGTH, -HALF_WIDTH],
    [HALF_LENGTH, -HALF_WIDTH],
    [HALF_LENGTH, HALF_WIDTH],
    [-HALF_LENGTH, HALF_WIDTH],
    [-HALF_LENGTH, -HALF_WIDTH],
  ]);
  lines.push([
    [0, -HALF_WIDTH],
    [0, HALF_WIDTH],
  ]);

  const circle: Array<[number, number]> = [];
  for (let i = 0; i <= 72; i++) {
    const t = (i / 72) * Math.PI * 2;
    circle.push([CENTRE_CIRCLE_RADIUS * Math.cos(t), CENTRE_CIRCLE_RADIUS * Math.sin(t)]);
  }
  lines.push(circle);

  for (const sign of [-1, 1] as const) {
    const goalLineX = sign * HALF_LENGTH;
    const penX = sign * (HALF_LENGTH - PENALTY_AREA_LENGTH);
    const goalAreaX = sign * (HALF_LENGTH - GOAL_AREA_LENGTH);
    const halfPen = PENALTY_AREA_WIDTH / 2;
    const halfGoalArea = GOAL_AREA_WIDTH / 2;
    const halfGoal = GOAL_WIDTH / 2;

    lines.push([
      [goalLineX, -halfPen],
      [penX, -halfPen],
      [penX, halfPen],
      [goalLineX, halfPen],
    ]);
    lines.push([
      [goalLineX, -halfGoalArea],
      [goalAreaX, -halfGoalArea],
      [goalAreaX, halfGoalArea],
      [goalLineX, halfGoalArea],
    ]);
    lines.push([
      [goalLineX, -halfGoal],
      [goalLineX + sign * 2, -halfGoal],
      [goalLineX + sign * 2, halfGoal],
      [goalLineX, halfGoal],
    ]);

    // The D, drawn only where it falls outside the penalty area.
    const spotX = sign * (HALF_LENGTH - PENALTY_SPOT_DISTANCE);
    let arc: Array<[number, number]> = [];
    for (let i = 0; i <= 120; i++) {
      const t = (i / 120) * Math.PI * 2;
      const px = spotX + CENTRE_CIRCLE_RADIUS * Math.cos(t);
      const py = CENTRE_CIRCLE_RADIUS * Math.sin(t);
      const outside = sign > 0 ? px < penX : px > penX;
      if (outside) {
        arc.push([px, py]);
      } else if (arc.length > 1) {
        lines.push(arc);
        arc = [];
      } else {
        arc = [];
      }
    }
    if (arc.length > 1) lines.push(arc);
  }

  return lines;
}

/** Penalty spots and the centre spot, drawn as filled dots. */
export function pitchSpots(): Array<[number, number]> {
  return [
    [0, 0],
    [-(HALF_LENGTH - PENALTY_SPOT_DISTANCE), 0],
    [HALF_LENGTH - PENALTY_SPOT_DISTANCE, 0],
  ];
}
