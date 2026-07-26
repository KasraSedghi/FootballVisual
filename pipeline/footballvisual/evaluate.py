"""Score pipeline output against synthetic ground truth.

An untested vision pipeline is a demo, not a system. Because the synthetic clip
knows where every player really was, the recovered tracks can be scored on the
questions that decide whether the tactical layer on top is trustworthy:

* **Position error.** How many metres off is a recovered player? This bounds
  every distance the tactics engine computes, so a passing-lane margin smaller
  than the position error is noise.
* **Detection coverage.** What fraction of the players on screen got tracked at
  all? Missing players silently distort team shape metrics.
* **Identity switches.** How often does a track jump between real players? One
  switch corrupts a trajectory permanently.
* **Team accuracy.** Are the two clusters actually the two teams?

Matching recovered tracks to ground-truth players uses a global assignment on
mean distance over the whole clip, not per frame. A per-frame greedy match
would hide identity switches by silently re-matching after each one, which is
precisely the failure being measured.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Evaluation:
    position_mae_m: float
    position_p95_m: float
    coverage: float
    id_switches: int
    team_accuracy: float
    matched_tracks: int
    truth_players: int
    ball_mae_m: float | None
    ball_coverage: float

    def to_dict(self) -> dict:
        return {
            "positionMaeM": round(self.position_mae_m, 3),
            "positionP95M": round(self.position_p95_m, 3),
            "coverage": round(self.coverage, 3),
            "idSwitches": self.id_switches,
            "teamAccuracy": round(self.team_accuracy, 3),
            "matchedTracks": self.matched_tracks,
            "truthPlayers": self.truth_players,
            "ballMaeM": round(self.ball_mae_m, 3) if self.ball_mae_m is not None else None,
            "ballCoverage": round(self.ball_coverage, 3),
        }

    def format(self) -> str:
        lines = [
            "pipeline accuracy vs ground truth",
            f"  position MAE          {self.position_mae_m:.2f} m",
            f"  position p95          {self.position_p95_m:.2f} m",
            f"  detection coverage    {self.coverage * 100:.1f}%  "
            f"({self.matched_tracks} tracks matched to {self.truth_players} players)",
            f"  identity switches     {self.id_switches}",
            f"  team assignment       {self.team_accuracy * 100:.1f}% correct",
            f"  ball coverage         {self.ball_coverage * 100:.1f}%",
        ]
        if self.ball_mae_m is not None:
            lines.append(f"  ball MAE              {self.ball_mae_m:.2f} m")
        return "\n".join(lines)


def _truth_positions(truth: dict) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, str]]:
    """Ground-truth series per shirt number, restricted to on-screen frames.

    A player the camera cannot see is not a detection failure, so only frames
    where the renderer actually drew that player count toward coverage.
    """
    series: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    teams: dict[int, str] = {}
    for rec in truth["frames"]:
        drawn = set(rec.get("boxes", {}).keys())
        for num_str, xy in rec["positions"].items():
            if num_str not in drawn:
                continue
            series[int(num_str)][rec["frame"]] = (float(xy[0]), float(xy[1]))
            teams[int(num_str)] = rec.get("teams", {}).get(num_str, "unknown")
    return series, teams


def _track_positions(tracks_json: dict) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, str]]:
    series: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    teams: dict[int, str] = {}
    for rec in tracks_json["frames"]:
        for p in rec["players"]:
            series[int(p["id"])][rec["frame"]] = (float(p["x"]), float(p["y"]))
            teams[int(p["id"])] = p.get("team", "unknown")
    return series, teams


def _mean_distance(
    a: dict[int, tuple[float, float]], b: dict[int, tuple[float, float]]
) -> tuple[float, int]:
    """Mean distance over frames both series cover."""
    shared = a.keys() & b.keys()
    if not shared:
        return float("inf"), 0
    d = [
        float(np.hypot(a[f][0] - b[f][0], a[f][1] - b[f][1]))
        for f in shared
    ]
    return float(np.mean(d)), len(shared)


def evaluate(tracks_json: dict, truth: dict, gate_m: float = 8.0) -> Evaluation:
    """Score `tracks_json` against `truth`.

    `gate_m` is the largest mean distance at which a track is still considered
    to be following a given player. Beyond it the pairing is treated as no match
    rather than as a very bad one, which keeps a spurious track from being
    charged against a real player it never followed.
    """
    truth_series, truth_teams = _truth_positions(truth)
    track_series, track_teams = _track_positions(tracks_json)

    truth_ids = sorted(truth_series)
    track_ids = sorted(track_series)
    if not truth_ids or not track_ids:
        return Evaluation(float("inf"), float("inf"), 0.0, 0, 0.0, 0, len(truth_ids), None, 0.0)

    cost = np.full((len(track_ids), len(truth_ids)), 1e6)
    overlap = np.zeros_like(cost)
    for i, tid in enumerate(track_ids):
        for j, num in enumerate(truth_ids):
            dist, shared = _mean_distance(track_series[tid], truth_series[num])
            if shared >= 3 and np.isfinite(dist):
                cost[i, j] = dist
                overlap[i, j] = shared

    rows, cols = linear_sum_assignment(cost)
    pairs = [
        (track_ids[r], truth_ids[c])
        for r, c in zip(rows, cols)
        if cost[r, c] <= gate_m
    ]

    # Position error, pooled over every matched frame.
    errors: list[float] = []
    for tid, num in pairs:
        a, b = track_series[tid], truth_series[num]
        for f in a.keys() & b.keys():
            errors.append(float(np.hypot(a[f][0] - b[f][0], a[f][1] - b[f][1])))

    mae = float(np.mean(errors)) if errors else float("inf")
    p95 = float(np.percentile(errors, 95)) if errors else float("inf")

    # Coverage: of all the frames where a player was on screen, in how many did
    # some track follow them?
    total_visible = sum(len(v) for v in truth_series.values())
    covered = sum(
        len(track_series[tid].keys() & truth_series[num].keys()) for tid, num in pairs
    )
    coverage = covered / total_visible if total_visible else 0.0

    # Identity switches: for each track, walk its frames and count how often the
    # nearest ground-truth player changes.
    switches = 0
    for tid in track_ids:
        frames = sorted(track_series[tid])
        previous: int | None = None
        for f in frames:
            px, py = track_series[tid][f]
            best_num, best_d = None, float("inf")
            for num in truth_ids:
                if f not in truth_series[num]:
                    continue
                tx, ty = truth_series[num][f]
                d = float(np.hypot(px - tx, py - ty))
                if d < best_d:
                    best_num, best_d = num, d
            if best_num is None or best_d > gate_m:
                continue
            if previous is not None and best_num != previous:
                switches += 1
            previous = best_num

    # Team accuracy. Cluster labels are arbitrary, so try both orientations and
    # keep the better one; anything else would score a perfect split as zero
    # half the time.
    truth_team_of_track = {tid: truth_teams.get(num, "unknown") for tid, num in pairs}
    predicted = {tid: track_teams.get(tid, "unknown") for tid, _ in pairs}
    best_acc = 0.0
    for mapping in ({"team_a": "blue", "team_b": "red"}, {"team_a": "red", "team_b": "blue"}):
        correct = sum(
            1
            for tid in predicted
            if mapping.get(predicted[tid]) == truth_team_of_track[tid]
        )
        denom = sum(1 for tid in predicted if predicted[tid] in mapping)
        if denom:
            best_acc = max(best_acc, correct / denom)

    # Ball.
    ball_truth = {
        rec["frame"]: (float(rec["ball"][0]), float(rec["ball"][1]))
        for rec in truth["frames"]
    }
    ball_track = {
        rec["frame"]: (float(rec["ball"][0]), float(rec["ball"][1]))
        for rec in tracks_json["frames"]
        if rec.get("ball")
    }
    shared_ball = ball_truth.keys() & ball_track.keys()
    ball_mae = (
        float(
            np.mean(
                [
                    np.hypot(
                        ball_truth[f][0] - ball_track[f][0],
                        ball_truth[f][1] - ball_track[f][1],
                    )
                    for f in shared_ball
                ]
            )
        )
        if shared_ball
        else None
    )
    scored_frames = len(tracks_json["frames"])
    ball_coverage = len(ball_track) / scored_frames if scored_frames else 0.0

    return Evaluation(
        position_mae_m=mae,
        position_p95_m=p95,
        coverage=coverage,
        id_switches=switches,
        team_accuracy=best_acc,
        matched_tracks=len(pairs),
        truth_players=len(truth_ids),
        ball_mae_m=ball_mae,
        ball_coverage=ball_coverage,
    )


def evaluate_files(tracks_path: Path, truth_path: Path) -> Evaluation:
    return evaluate(
        json.loads(Path(tracks_path).read_text()),
        json.loads(Path(truth_path).read_text()),
    )
