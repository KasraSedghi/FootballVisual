"""Keeping the pitch-to-image homography valid while the camera moves.

Calibrating one frame is easy. Staying calibrated is the actual problem: a
broadcast camera pans, tilts and zooms continuously, so a homography fitted on
frame zero is wrong by frame thirty.

The approach here is frame-to-frame propagation. Track sparse features on the
*pitch surface* between consecutive frames, fit the inter-frame image-to-image
homography those correspondences imply, and compose it onto the running
pitch-to-image estimate:

    H_pitch->img(t+1) = H_img(t)->img(t+1) @ H_pitch->img(t)

Two things make this work rather than fall apart. Features are sampled only
from the grass, with detected players masked out, because a feature on a moving
player describes the player's motion and not the camera's. And composition
accumulates error, so drift is measured and reported rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import pitch
from .homography import HomographyError, project


@dataclass
class DriftReport:
    """Per-frame agreement between the tracked homography and a reference."""

    frame: int
    mean_error_m: float
    max_error_m: float
    inlier_features: int


@dataclass
class HomographyTracker:
    """Propagates a pitch-to-image homography across frames via optical flow.

    `min_features` guards the failure mode that matters: when too few pitch
    features survive the flow, the inter-frame fit is unreliable, and it is
    better to hold the previous homography for a frame than to accept a bad
    update and corrupt the running estimate permanently.
    """

    h: np.ndarray
    min_features: int = 25
    max_features: int = 600
    ransac_threshold: float = 2.5
    _prev_gray: np.ndarray | None = field(default=None, repr=False)
    _prev_points: np.ndarray | None = field(default=None, repr=False)
    frames_held: int = 0

    def _pitch_mask(self, frame: np.ndarray, player_boxes) -> np.ndarray:
        """Pixels that are grass and not covered by a player.

        Keying on green is crude but exactly right for the job: it selects the
        painted lines' surroundings and the mow-stripe texture, which is the
        static structure the camera motion should be measured against.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([100, 255, 255]))
        # Include white pitch lines, which sit on grass and are the strongest
        # trackable features available.
        white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 80, 255]))
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(white, cv2.dilate(mask, np.ones((9, 9), np.uint8))))

        for x1, y1, x2, y2 in player_boxes or []:
            # Generous padding. A feature clinging to a player's edge is worse
            # than no feature, because it moves with the player.
            pad_x = 0.35 * (x2 - x1)
            pad_y = 0.2 * (y2 - y1)
            ix1 = int(max(0, x1 - pad_x))
            iy1 = int(max(0, y1 - pad_y))
            ix2 = int(min(frame.shape[1], x2 + pad_x))
            iy2 = int(min(frame.shape[0], y2 + pad_y))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 0
        return mask

    def start(self, frame: np.ndarray, player_boxes=None) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._pitch_mask(frame, player_boxes)
        self._prev_gray = gray
        self._prev_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=8, mask=mask
        )
        self.frames_held = 0

    def step(self, frame: np.ndarray, player_boxes=None) -> np.ndarray:
        """Advance to `frame`, returning the updated pitch-to-image homography."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_points is None or len(self._prev_points) < self.min_features:
            self.start(frame, player_boxes)
            return self.h

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None:
            self.start(frame, player_boxes)
            return self.h

        good_old = self._prev_points[status.ravel() == 1].reshape(-1, 2)
        good_new = nxt[status.ravel() == 1].reshape(-1, 2)

        if len(good_old) < self.min_features:
            self.frames_held += 1
            self.start(frame, player_boxes)
            return self.h

        delta, inliers = cv2.findHomography(
            good_old, good_new, cv2.RANSAC, self.ransac_threshold
        )
        if delta is None or not np.all(np.isfinite(delta)):
            self.frames_held += 1
            self.start(frame, player_boxes)
            return self.h

        num_inliers = int(inliers.sum()) if inliers is not None else 0
        if num_inliers < self.min_features:
            self.frames_held += 1
            self.start(frame, player_boxes)
            return self.h

        self.h = delta @ self.h
        if abs(self.h[2, 2]) > 1e-12:
            self.h = self.h / self.h[2, 2]

        # Re-seed features every frame from the current image. Re-detecting is
        # cheap next to the alternative, which is watching the tracked set decay
        # until it drops below the threshold and forces a hard restart.
        self._prev_gray = gray
        mask = self._pitch_mask(frame, player_boxes)
        self._prev_points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=8, mask=mask
        )
        self.last_inliers = num_inliers
        return self.h


def landmark_correspondences(
    h_true: np.ndarray,
    width: int,
    height: int,
    noise_px: float = 0.0,
    seed: int = 0,
    margin: int = 12,
) -> dict[str, tuple[float, float]]:
    """Which pitch landmarks are visible in a frame, and where.

    This stands in for the human who clicks pitch landmarks on the first frame,
    or for an automatic line-intersection detector. `noise_px` simulates how
    imprecise that click is, so the pipeline is never handed a perfect
    calibration it would not get in practice.
    """
    rng = np.random.default_rng(seed)
    names = sorted(pitch.LANDMARKS)
    pts = np.array([pitch.landmark(n).xy for n in names], dtype=np.float64)
    projected = project(h_true, pts)

    out: dict[str, tuple[float, float]] = {}
    for name, (x, y) in zip(names, projected):
        if not (margin <= x < width - margin and margin <= y < height - margin):
            continue
        if noise_px > 0:
            x = x + rng.normal(0.0, noise_px)
            y = y + rng.normal(0.0, noise_px)
        out[name] = (float(x), float(y))
    return out


def pitch_error(
    h_est: np.ndarray,
    h_true: np.ndarray,
    samples: int = 400,
    seed: int = 0,
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """How wrong an estimated homography is, measured in metres on the pitch.

    Pixel reprojection error is the wrong unit to report to anyone who cares
    about tactics: two pixels of error near the far touchline is several metres
    on the ground, and two pixels in the foreground is centimetres. This samples
    points across the pitch, pushes them through the truth into the image and
    back through the estimate, and reports how far off the round trip lands in
    metres.

    `image_size` restricts scoring to pitch points the camera can actually see.
    Without it the average is dominated by parts of the pitch that are outside
    the frame, where both homographies are extrapolating far beyond any evidence
    and the disagreement between them says nothing about how well a *player*
    would be located. Since players only exist where the camera is looking, the
    visible region is the only region whose accuracy matters.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-pitch.HALF_LENGTH, pitch.HALF_LENGTH, samples)
    ys = rng.uniform(-pitch.HALF_WIDTH, pitch.HALF_WIDTH, samples)
    pts = np.column_stack([xs, ys])

    try:
        img = project(h_true, pts)
        back = project(np.linalg.inv(h_est), img)
    except (np.linalg.LinAlgError, HomographyError):
        return float("inf"), float("inf")

    keep = np.ones(len(pts), dtype=bool)
    if image_size is not None:
        width, height = image_size
        keep = (
            (img[:, 0] >= 0) & (img[:, 0] < width)
            & (img[:, 1] >= 0) & (img[:, 1] < height)
        )
        if keep.sum() < 8:
            keep = np.ones(len(pts), dtype=bool)

    err = np.sqrt(((back - pts) ** 2).sum(axis=1))[keep]
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("inf"), float("inf")
    return float(err.mean()), float(err.max())
