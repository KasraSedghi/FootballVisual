"""Multi-object tracking, a from-scratch ByteTrack.

ByteTrack's insight is that the low-confidence detections everyone throws away
are mostly real objects that happen to be occluded, and that they can be
recovered by matching them against tracks that the high-confidence pass failed
to explain. That matters enormously in football, where players occlude each
other constantly and the detector's confidence collapses exactly when two
players cross, which is exactly the moment an identity switch would happen.

Implemented here rather than imported so the association logic and the motion
model are inspectable and testable, and because the pipeline needs to report
identity-switch counts against ground truth, which means owning the track
lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


def _to_xyah(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Box to (centre x, centre y, aspect ratio, height).

    Tracking in this parameterisation rather than in corner coordinates keeps
    the filter's notion of "the object got bigger" separate from "the object
    moved", which is what lets a player walking toward the camera be modelled
    as smooth motion instead of as four independently drifting corners.
    """
    x1, y1, x2, y2 = bbox
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def _to_bbox(xyah: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, a, h = xyah[:4]
    h = max(1e-6, h)
    w = max(1e-6, a * h)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over (x, y, aspect, height).

    State is eight-dimensional: the four measured quantities and their rates.
    Process and measurement noise are scaled by the box height, following the
    SORT/DeepSORT convention, because positional uncertainty in pixels grows
    with apparent size and a fixed noise floor would over-trust distant players
    and under-trust near ones.
    """

    def __init__(self) -> None:
        self.ndim = 4
        dt = 1.0
        self._motion = np.eye(8)
        for i in range(self.ndim):
            self._motion[i, self.ndim + i] = dt
        self._update = np.eye(self.ndim, 8)
        self._std_position = 1.0 / 20.0
        self._std_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate([measurement, np.zeros(4)])
        h = measurement[3]
        std = np.array(
            [
                2 * self._std_position * h,
                2 * self._std_position * h,
                1e-2,
                2 * self._std_position * h,
                10 * self._std_velocity * h,
                10 * self._std_velocity * h,
                1e-5,
                10 * self._std_velocity * h,
            ]
        )
        return mean, np.diag(std**2)

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_position * h,
                self._std_position * h,
                1e-2,
                self._std_position * h,
                self._std_velocity * h,
                self._std_velocity * h,
                1e-5,
                self._std_velocity * h,
            ]
        )
        motion_cov = np.diag(std**2)
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + motion_cov
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_position * h,
                self._std_position * h,
                1e-1,
                self._std_position * h,
            ]
        )
        innovation_cov = np.diag(std**2)

        projected_mean = self._update @ mean
        projected_cov = self._update @ cov @ self._update.T + innovation_cov

        try:
            kalman_gain = np.linalg.solve(projected_cov.T, (cov @ self._update.T).T).T
        except np.linalg.LinAlgError:  # pragma: no cover - numerical edge
            return mean, cov

        innovation = measurement - projected_mean
        mean = mean + kalman_gain @ innovation
        cov = cov - kalman_gain @ projected_cov @ kalman_gain.T
        return mean, cov


@dataclass
class Track:
    """One tracked object and its history."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    score: float = 0.0
    history: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _to_bbox(self.mean)

    @property
    def is_active(self) -> bool:
        """Whether this track should be offered a detection this frame.

        Lost tracks are included deliberately. Recovering a track that was
        occluded for a few frames is the main thing this tracker exists to do,
        and it can only happen if the track is still a candidate for matching
        while it is lost.
        """
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)


def iou_matrix(tracks: list[Track], detections: list[Detection]) -> np.ndarray:
    """Pairwise IoU between predicted track boxes and detections."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))

    t_boxes = np.array([t.bbox for t in tracks], dtype=np.float64)
    d_boxes = np.array([d.bbox for d in detections], dtype=np.float64)

    t_area = np.maximum(0, t_boxes[:, 2] - t_boxes[:, 0]) * np.maximum(
        0, t_boxes[:, 3] - t_boxes[:, 1]
    )
    d_area = np.maximum(0, d_boxes[:, 2] - d_boxes[:, 0]) * np.maximum(
        0, d_boxes[:, 3] - d_boxes[:, 1]
    )

    xx1 = np.maximum(t_boxes[:, None, 0], d_boxes[None, :, 0])
    yy1 = np.maximum(t_boxes[:, None, 1], d_boxes[None, :, 1])
    xx2 = np.minimum(t_boxes[:, None, 2], d_boxes[None, :, 2])
    yy2 = np.minimum(t_boxes[:, None, 3], d_boxes[None, :, 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    union = t_area[:, None] + d_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _associate(
    tracks: list[Track], detections: list[Detection], iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment on IoU, gated by a minimum overlap.

    The gate matters more than the optimiser: without it, the Hungarian
    algorithm will happily pair a track with a detection on the far side of the
    frame simply because that pairing minimises total cost.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou = iou_matrix(tracks, detections)
    rows, cols = linear_sum_assignment(-iou)

    matches: list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


class ByteTracker:
    """ByteTrack: two-stage association against high- and low-score detections.

    Parameters are tuned for football at broadcast framing, where boxes are
    small and IoU between adjacent frames is lower than in the pedestrian
    benchmarks ByteTrack was designed on. In particular `match_threshold` is
    looser than the usual 0.8 because a 30-pixel-tall player moving 4 pixels
    between frames already loses a lot of overlap.

    `high_threshold` is set from measurement rather than convention. Only
    high-score detections may start a new track, so that threshold decides which
    players ever enter the system at all. Sweeping it against ground truth on
    this project's clip showed detector precision holding at 0.98 all the way
    down to 0.25, while the usual 0.45 discarded roughly a third of genuinely
    visible players. Distant players simply do not detect confidently, and
    refusing to track anyone the network is less than half sure about means
    never tracking the far side of the pitch.
    """

    def __init__(
        self,
        high_threshold: float = 0.25,
        low_threshold: float = 0.10,
        match_threshold: float = 0.25,
        second_match_threshold: float = 0.15,
        max_age: int = 40,
        min_hits: int = 3,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frame = -1

    def _new_track(self, det: Detection) -> Track:
        mean, cov = self.kf.initiate(_to_xyah(det.bbox))
        track = Track(
            track_id=self._next_id, mean=mean, covariance=cov, score=det.score
        )
        self._next_id += 1
        return track

    def update(self, detections: list[Detection], frame_index: int | None = None) -> list[Track]:
        """Advance the tracker by one frame and return the active tracks."""
        self._frame = frame_index if frame_index is not None else self._frame + 1

        for track in self.tracks:
            track.mean, track.covariance = self.kf.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1

        high = [d for d in detections if d.score >= self.high_threshold]
        low = [
            d for d in detections if self.low_threshold <= d.score < self.high_threshold
        ]

        active = [t for t in self.tracks if t.is_active]

        # Stage one: confident detections against every active track, lost ones
        # included so an occluded player can be picked straight back up.
        matches, unmatched_t, unmatched_d = _associate(active, high, self.match_threshold)
        for ti, di in matches:
            self._apply(active[ti], high[di])

        # Stage two: the leftovers. Only tracks with an established history get
        # a shot at a low-confidence detection; letting a one-frame tentative
        # track latch onto a weak blob is how phantom tracks are born.
        second_chance = [
            active[i]
            for i in unmatched_t
            if active[i].state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        matched_second: set[int] = set()
        if second_chance and low:
            m2, _, _ = _associate(second_chance, low, self.second_match_threshold)
            for ti, di in m2:
                self._apply(second_chance[ti], low[di])
                matched_second.add(id(second_chance[ti]))

        for i in unmatched_t:
            track = active[i]
            if id(track) in matched_second:
                continue
            if track.state is TrackState.TENTATIVE:
                # A tentative track that misses even one frame was probably a
                # false positive. Killing it immediately keeps the ID space from
                # filling with noise.
                track.state = TrackState.REMOVED
            elif track.time_since_update > self.max_age:
                # Applies to lost tracks too, which is what stops them
                # accumulating forever as candidates that can never match.
                track.state = TrackState.REMOVED
            else:
                track.state = TrackState.LOST

        for di in unmatched_d:
            self.tracks.append(self._new_track(high[di]))

        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        out: list[Track] = []
        for track in self.tracks:
            if track.state is TrackState.CONFIRMED and track.time_since_update == 0:
                track.history.append((self._frame, track.bbox))
                out.append(track)
        return out

    def _apply(self, track: Track, det: Detection) -> None:
        track.mean, track.covariance = self.kf.update(
            track.mean, track.covariance, _to_xyah(det.bbox)
        )
        track.hits += 1
        track.time_since_update = 0
        track.score = det.score
        if track.state is TrackState.LOST:
            track.state = TrackState.CONFIRMED
        elif track.state is TrackState.TENTATIVE and track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED


class BallTracker:
    """Single-target ball filter with coasting through missed frames.

    The ball is missed often, so a plain nearest-detection tracker produces a
    position series full of holes. A constant-velocity filter that keeps
    predicting through gaps produces a continuous trajectory, which is what the
    possession logic and the sandbox timeline need. Coasting is capped because
    an unbounded prediction sails off the pitch and looks confident doing it.
    """

    def __init__(self, max_coast: int = 12, max_jump_px: float = 55.0) -> None:
        self.max_coast = max_coast
        # Largest believable movement in *one* frame. A driven pass at 16 m/s
        # covers 0.64m per frame, which is a few tens of pixels at broadcast
        # scale, so a gate of hundreds of pixels accepts essentially any white
        # blob on the pitch. Measured on this project's clip, a loose gate left
        # around 10% of frames locked onto the wrong blob and roughly 15m out,
        # which dragged ball error from a 1.8m median to a 3.0m mean.
        self.max_jump_px = max_jump_px
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2)
        self.missed = 0

    def update(self, detection: Detection | None) -> tuple[float, float] | None:
        if detection is not None:
            # The ball's ground contact point, not its centre. The homography
            # maps the ground plane, so projecting the centre of a sphere is the
            # same mistake as projecting the centre of a player's bounding box,
            # and it biases the result the same way: up the pitch, growing with
            # distance from the camera.
            measured = np.array(detection.foot_point, dtype=np.float64)
            if self.position is None:
                self.position = measured
                self.velocity = np.zeros(2)
            else:
                jump = float(np.linalg.norm(measured - self.position))
                # The budget scales with how long the ball has been unseen,
                # because it really can have travelled further in that time.
                # A fixed gate would either reject legitimate re-acquisitions
                # after an occlusion or accept nonsense on consecutive frames.
                allowed = self.max_jump_px * (1 + self.missed)
                if jump > allowed and self.missed < self.max_coast:
                    # Too far to be the same ball this soon. Coast instead of
                    # teleporting, and let a second consistent sighting win.
                    return self._coast()
                self.velocity = 0.6 * self.velocity + 0.4 * (measured - self.position)
                self.position = measured
            self.missed = 0
            return (float(self.position[0]), float(self.position[1]))
        return self._coast()

    def _coast(self) -> tuple[float, float] | None:
        if self.position is None:
            return None
        self.missed += 1
        if self.missed > self.max_coast:
            self.position = None
            self.velocity = np.zeros(2)
            return None
        # Damp the extrapolation so a long gap decays toward a stop rather than
        # accelerating away.
        self.velocity *= 0.85
        self.position = self.position + self.velocity
        return (float(self.position[0]), float(self.position[1]))
