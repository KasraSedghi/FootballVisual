"""Assign tracks to teams by jersey colour.

The approach is deliberately unsupervised. Hard-coding "team A is blue" means
the pipeline only works on clips someone has configured first, whereas
clustering the observed shirt colours into two groups works on any fixture. The
labels that come out are arbitrary ("team_a"/"team_b") because nothing in the
footage says which is home.

Three details do most of the work here:

* Sample the torso, not the box. A player's bounding box is mostly grass.
* Drop green pixels before averaging, because even the torso strip contains
  background around the arms, and averaging it in drags every player toward the
  same green and collapses the clusters.
* Vote per track over time, not per frame. A single frame's colour estimate on
  a 30-pixel-tall player is noisy enough to flip; the mode over a track's
  lifetime is stable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from . import pitch


@dataclass
class TeamAssignment:
    """Result of clustering, mapping track id to a team label."""

    labels: dict[int, str]
    centres: dict[str, tuple[float, float, float]]
    keeper_ids: set[int]
    confidence: dict[int, float]

    def team_of(self, track_id: int) -> str:
        return self.labels.get(track_id, "unknown")


def torso_colour(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Mean Lab colour of a player's shirt, or None if unusable.

    Returns Lab rather than BGR or HSV because Lab's Euclidean distance
    approximates perceived colour difference, which is what a two-team split
    should be measuring. HSV's hue wraps at red and would put a red shirt on
    both ends of the axis.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 8:
        return None

    box_h = y2 - y1
    box_w = x2 - x1
    # Upper-middle band: below the head, above the shorts, inside the arms.
    ty1 = y1 + int(0.18 * box_h)
    ty2 = y1 + int(0.55 * box_h)
    tx1 = x1 + int(0.22 * box_w)
    tx2 = x2 - int(0.22 * box_w)
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        tx1, tx2, ty1, ty2 = x1, x2, y1 + int(0.2 * box_h), y1 + int(0.6 * box_h)
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    patch = frame[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    # Reject grass (green hue with real saturation) and near-black shadow.
    is_grass = (hue >= 32) & (hue <= 95) & (sat > 60)
    is_dark = val < 35
    keep = ~(is_grass | is_dark)
    if keep.sum() < max(4, 0.08 * keep.size):
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return lab[keep].reshape(-1, 3).mean(axis=0).astype(np.float64)


def _kmeans_two(samples: np.ndarray, seed: int = 0, iterations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Two-cluster k-means, seeded by the two most separated samples.

    Deterministic seeding matters: random initialisation makes the team labels
    swap between runs, which turns every downstream snapshot into a moving
    target. Picking the farthest-apart pair is also a better start than random
    for exactly two well-separated jersey colours.
    """
    if len(samples) < 2:
        raise ValueError("need at least two samples to cluster")

    dists = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    centres = samples[[i, j]].astype(np.float64).copy()

    labels = np.zeros(len(samples), dtype=int)
    for _ in range(iterations):
        d = np.linalg.norm(samples[:, None, :] - centres[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            if (labels == k).any():
                centres[k] = samples[labels == k].mean(axis=0)
    return labels, centres


def assign_teams(
    observations: dict[int, list[np.ndarray]],
    min_observations: int = 3,
    keeper_margin: float = 1.9,
) -> TeamAssignment:
    """Cluster per-track colour observations into two teams.

    `observations` maps a track id to the Lab colours sampled for it across the
    clip. Tracks with too few usable samples are left unassigned rather than
    guessed, because a wrong team label is worse for the tactical analysis than
    a missing one: it puts a defender in the attacking shape.

    Goalkeepers and referees wear colours belonging to neither cluster, so any
    track whose distance to its nearest centre is far larger than typical is
    flagged rather than forced into a team.
    """
    track_ids = [
        tid for tid, obs in observations.items() if len(obs) >= min_observations
    ]
    if len(track_ids) < 2:
        return TeamAssignment({}, {}, set(), {})

    # The median is the right summary here. A track that was briefly occluded
    # contributes a few frames of the occluder's colour, and the mean would
    # carry that; the median discards it.
    medians = np.array(
        [np.median(np.stack(observations[tid]), axis=0) for tid in track_ids]
    )

    labels, centres = _kmeans_two(medians)

    distances = np.linalg.norm(medians - centres[labels], axis=1)
    typical = float(np.median(distances)) if len(distances) else 0.0
    outlier_cut = max(18.0, typical * keeper_margin)

    names = {0: "team_a", 1: "team_b"}
    out_labels: dict[int, str] = {}
    confidence: dict[int, float] = {}
    keepers: set[int] = set()

    for idx, tid in enumerate(track_ids):
        own = float(np.linalg.norm(medians[idx] - centres[labels[idx]]))
        other = float(np.linalg.norm(medians[idx] - centres[1 - labels[idx]]))
        if own > outlier_cut:
            keepers.add(tid)
            out_labels[tid] = "other"
            confidence[tid] = 0.0
            continue
        out_labels[tid] = names[int(labels[idx])]
        # Margin-based confidence: how much closer the chosen centre is.
        total = own + other
        confidence[tid] = float(1.0 - (own / total)) * 2.0 - 1.0 if total > 0 else 0.0
        confidence[tid] = float(np.clip(confidence[tid], 0.0, 1.0))

    centre_map = {
        names[k]: tuple(float(v) for v in centres[k]) for k in (0, 1)
    }
    return TeamAssignment(out_labels, centre_map, keepers, confidence)


class TeamVoter:
    """Accumulates per-frame colour samples, then resolves teams once.

    Deferring the decision to the end of the clip is what makes assignment
    stable. Clustering frame by frame would let a player flip teams whenever
    they turned or passed through shadow, and the tactical metrics would
    recompute the defensive shape around a different set of players every
    frame.
    """

    def __init__(self) -> None:
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)

    def observe(self, track_id: int, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
        colour = torso_colour(frame, bbox)
        if colour is None:
            return
        # Repeat the sample in proportion to how tall the detection is. A player
        # 100 pixels high yields a torso patch of a few hundred pixels and a
        # trustworthy mean; one 25 pixels high yields a dozen pixels, half of
        # them contaminated by the background. Both would otherwise carry equal
        # weight in the median, letting the noisiest observations on the far
        # touchline drag a cluster around. Repetition rather than a weight
        # keeps the downstream median a plain median.
        height = bbox[3] - bbox[1]
        weight = 1 if height < 40 else (2 if height < 70 else 3)
        self._samples[track_id].extend([colour] * weight)

    def resolve(self, **kwargs) -> TeamAssignment:
        return assign_teams(dict(self._samples), **kwargs)

    def sample_counts(self) -> dict[int, int]:
        return {tid: len(v) for tid, v in self._samples.items()}


def majority_team(labels: list[str]) -> str:
    """Most common label, used to collapse a track's per-frame votes."""
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


@dataclass
class OfficialsResult:
    """Which of the colour outliers are keepers, and which are officials."""

    roles: dict[int, str]
    evidence: dict[int, dict]

    def role_of(self, track_id: int) -> str:
        return self.roles.get(track_id, "unknown")


def classify_officials(
    candidate_ids: set[int],
    trajectories: dict[int, list[tuple[float, float]]],
    ball: list[tuple[float, float]] | None = None,
    goal_band_m: float = 18.0,
    max_keeper_range_m: float = 34.0,
) -> OfficialsResult:
    """Split colour outliers into goalkeepers and match officials.

    Colour alone cannot do this. Both wear kit unlike either team, which is
    exactly why they end up in the same bucket, and a referee's black is not
    reliably more or less like a keeper's green than the two teams are like each
    other. What separates them is where they spend their time, so this runs on
    pitch coordinates rather than pixels.

    A goalkeeper lives in a band in front of one goal and essentially never
    leaves it, so their x is extreme and its range is small. A referee follows
    play across the whole pitch, so their x range is large and they are usually
    near the ball. Those two signals disagree strongly enough that a simple rule
    separates them, and the evidence is returned so a wrong call can be
    understood rather than just observed.

    Anything with too little history to judge is left `unknown`, because
    mislabelling a defender as a keeper would corrupt the offside line, which is
    worse than declining to label.
    """
    roles: dict[int, str] = {}
    evidence: dict[int, dict] = {}

    ball_by_index = ball or []

    for tid in sorted(candidate_ids):
        points = trajectories.get(tid, [])
        if len(points) < 8:
            roles[tid] = "unknown"
            evidence[tid] = {"reason": "too few observations", "frames": len(points)}
            continue

        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)

        median_x = float(np.median(xs))
        x_range = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        distance_to_goal = float(pitch.HALF_LENGTH - abs(median_x))

        mean_ball_distance = None
        if ball_by_index and len(ball_by_index) >= len(points):
            paired = [
                float(np.hypot(px - bx, py - by))
                for (px, py), (bx, by) in zip(points, ball_by_index)
            ]
            if paired:
                mean_ball_distance = float(np.mean(paired))

        is_keeper = distance_to_goal <= goal_band_m and x_range <= max_keeper_range_m

        roles[tid] = "keeper" if is_keeper else "referee"
        evidence[tid] = {
            "medianX": round(median_x, 1),
            "xRange": round(x_range, 1),
            "distanceToGoalM": round(distance_to_goal, 1),
            "meanBallDistanceM": (
                round(mean_ball_distance, 1) if mean_ball_distance is not None else None
            ),
            "medianY": round(float(np.median(ys)), 1),
        }

    return OfficialsResult(roles=roles, evidence=evidence)
