/**
 * Shared types for the tactical analysis engine.
 *
 * Everything in this engine works in pitch coordinates: metres, origin at the
 * centre spot, +x toward the right-hand goal, +y toward the far touchline. The
 * vision pipeline emits these directly, and the sandbox never converts to
 * screen space until the moment of drawing, so no analysis result ever depends
 * on the size of the window it happened to be computed in.
 */

export type TeamId = "team_a" | "team_b" | "other" | "unknown";

export interface Vec2 {
  x: number;
  y: number;
}

/** A player at one instant, optionally with a velocity recovered from tracking. */
export interface PlayerState {
  id: number;
  team: TeamId;
  x: number;
  y: number;
  /** Metres per second. Absent when there is not enough history to estimate it. */
  vx?: number;
  vy?: number;
  /** Shirt number or role, when known. Display only, never used in the maths. */
  label?: string;
}

/** One frame of the match, in pitch coordinates. */
export interface Snapshot {
  frame: number;
  timeS: number;
  players: PlayerState[];
  ball: Vec2 | null;
}

/** Physical constants for the interception model. */
export interface MotionModel {
  /** Speed of a driven ground pass, m/s. */
  passSpeed: number;
  /** Top sprint speed of an outfield player, m/s. */
  playerMaxSpeed: number;
  /**
   * Seconds before a defender starts moving toward a pass. This single number
   * decides more outcomes than any other parameter: at 0 the model claims
   * defenders react instantly and almost nothing is open, and above about 0.5
   * it will call lanes open that any competent defender closes.
   */
  reactionTimeS: number;
  /** How close a defender must get to the ball's path to touch it, metres. */
  interceptRadius: number;
}

export const DEFAULT_MOTION: MotionModel = {
  // A firmly struck ground pass over a short distance. Real passes range from
  // about 8 m/s for a rolled ball to well over 25 m/s for a driven one; this is
  // the middle of the band that matters for breaking a block.
  passSpeed: 16,
  playerMaxSpeed: 7.5,
  reactionTimeS: 0.28,
  interceptRadius: 0.9,
};

export type LaneVerdict = "open" | "contested" | "blocked";

export interface PassingLane {
  targetId: number;
  from: Vec2;
  to: Vec2;
  distanceM: number;
  /**
   * Seconds by which the ball beats the best-placed defender to the most
   * dangerous point on its path. Negative means that defender arrives first,
   * so the pass can be cut out.
   */
  safetyMarginS: number;
  /** Which defender is the threat, and where along the lane they get to it. */
  threatId: number | null;
  threatPointM: number;
  /** Closest a defender ever is to the lane, metres. Cosmetic, not the verdict. */
  minClearanceM: number;
  /** Metres of ground gained toward the opponent's goal. */
  progressionM: number;
  /** Defenders the pass plays beyond, i.e. how many it takes out of the game. */
  defendersBypassed: number;
  /** Distance from the receiver to their nearest opponent, metres. */
  receiverPressureM: number;
  verdict: LaneVerdict;
  /** Combined 0..1 desirability, used only for ranking the display. */
  score: number;
}

export interface BlockShape {
  team: TeamId;
  playerCount: number;
  centroid: Vec2;
  /** Area of the convex hull of the team's outfield players, m^2. */
  hullAreaM2: number;
  hull: Vec2[];
  /** Spread across the pitch and along it, metres. */
  widthM: number;
  depthM: number;
  /** Mean x of the deepest four outfield players: the defensive line. */
  defensiveLineX: number;
  /** Mean x of the most advanced two: where the press starts. */
  forwardLineX: number;
  /** Distance between those, i.e. how compact the block is front to back. */
  blockDepthM: number;
  /** Largest lateral gap between adjacent members of the defensive line. */
  largestBackLineGapM: number;
  backLineGapCentreY: number;
  /** Offside line for this team when defending, in pitch x. */
  offsideLineX: number;
}

export interface SpaceControl {
  /** Grid resolution in metres. */
  cellM: number;
  /** Fraction of the pitch each team is nearest to controlling. */
  teamAShare: number;
  teamBShare: number;
  /** Row-major control values in [-1, 1]: -1 fully team_a, +1 fully team_b. */
  grid: Float32Array;
  cols: number;
  rows: number;
}

export interface TacticalReport {
  snapshot: Snapshot;
  /** Team judged to be in possession, from proximity to the ball. */
  attackingTeam: TeamId;
  defendingTeam: TeamId;
  carrierId: number | null;
  lanes: PassingLane[];
  block: BlockShape | null;
  attackingShape: BlockShape | null;
  space: SpaceControl | null;
  /** Attackers standing beyond the defending team's last line. */
  playersBeyondLine: number[];
  /** Attackers positioned between the opponent's midfield and defensive lines. */
  playersBetweenLines: number[];
}
