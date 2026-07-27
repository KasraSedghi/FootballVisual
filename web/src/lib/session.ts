/**
 * Loading track data and holding the sandbox's editable state.
 *
 * The central idea: the pipeline's output is immutable, and anything the user
 * does in the sandbox is an *override* layered on top of it. Editing the loaded
 * frames in place would mean a drag silently destroys the measured data, with
 * no way back and no way to tell what was observed from what was imagined. With
 * overrides, clearing them restores the truth exactly, and the analysis can
 * always say which positions came from the video.
 */

import type { PlayerState, Snapshot, TeamId, Vec2 } from "./tactics";

export interface Arrow {
  id: string;
  from: Vec2;
  to: Vec2;
  note?: string;
}

export interface DragOverride {
  id: number;
  x: number;
  y: number;
}

export interface TrackMeta {
  id: number;
  team: TeamId;
  teamConfidence: number;
  frames: number;
}

export interface TracksFile {
  meta: {
    source: string;
    fps: number;
    width: number;
    height: number;
    pitch: { length: number; width: number };
    stats?: Record<string, unknown>;
  };
  tracks: TrackMeta[];
  frames: Array<{
    frame: number;
    timeS: number;
    players: Array<{ id: number; team: TeamId; x: number; y: number }>;
    ball: [number, number] | null;
  }>;
}

export interface LoadedSession {
  meta: TracksFile["meta"];
  tracks: TrackMeta[];
  snapshots: Snapshot[];
  /** Frame number to index in `snapshots`, since frames may be strided. */
  indexOfFrame: Map<number, number>;
}

export async function loadTracks(url: string): Promise<LoadedSession> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `could not load ${url} (${response.status}). Run the pipeline first: ` +
        `make demo`,
    );
  }
  const data = (await response.json()) as TracksFile;
  return buildSession(data);
}

/**
 * Build a session from a `tracks.json` the user picked off their own disk.
 *
 * The shape is validated before anything is built, because the failure this
 * guards against is not a corrupt file, it is the *wrong* file. Handing this a
 * `ground_truth.json`, which sits in the same directory and looks similar at a
 * glance, would otherwise produce an empty pitch and no explanation.
 */
export async function loadTracksFromFile(file: File): Promise<LoadedSession> {
  let data: TracksFile;
  try {
    data = JSON.parse(await file.text()) as TracksFile;
  } catch {
    throw new Error(`${file.name} is not valid JSON.`);
  }

  if (!data || typeof data !== "object" || !Array.isArray(data.frames)) {
    throw new Error(
      `${file.name} has no "frames" array, so it is not a tracks.json. ` +
        `The pipeline writes one with "make demo" or the track command.`,
    );
  }
  if (!data.frames.length) {
    throw new Error(`${file.name} contains no frames.`);
  }
  if (!data.meta?.fps) {
    throw new Error(`${file.name} has no meta.fps, so playback cannot be timed.`);
  }

  return buildSession(data);
}

export function buildSession(data: TracksFile): LoadedSession {
  const snapshots: Snapshot[] = data.frames.map((f) => ({
    frame: f.frame,
    timeS: f.timeS,
    players: f.players.map((p) => ({
      id: p.id,
      team: p.team,
      x: p.x,
      y: p.y,
      label: String(p.id),
    })),
    ball: f.ball ? { x: f.ball[0], y: f.ball[1] } : null,
  }));

  const indexOfFrame = new Map<number, number>();
  snapshots.forEach((s, i) => indexOfFrame.set(s.frame, i));

  return { meta: data.meta, tracks: data.tracks, snapshots, indexOfFrame };
}

/**
 * Apply the user's drags to a snapshot.
 *
 * Overrides that name a player who is not on the pitch this frame are ignored
 * rather than added, because a track can disappear when it is lost and
 * resurrecting it from a stale drag would invent a player the video never saw.
 */
export function applyOverrides(
  snapshot: Snapshot,
  overrides: Map<number, DragOverride>,
  ballOverride: Vec2 | null,
): Snapshot {
  if (!overrides.size && !ballOverride) return snapshot;

  const players: PlayerState[] = snapshot.players.map((p) => {
    const o = overrides.get(p.id);
    return o ? { ...p, x: o.x, y: o.y } : p;
  });

  return { ...snapshot, players, ball: ballOverride ?? snapshot.ball };
}

/** Human-readable team names, since the clustering only knows "a" and "b". */
export function teamLabel(team: TeamId): string {
  switch (team) {
    case "team_a":
      return "Team A";
    case "team_b":
      return "Team B";
    case "keeper":
      return "Goalkeeper";
    case "referee":
      return "Referee";
    case "other":
      return "Unclassified";
    default:
      return "Unassigned";
  }
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
