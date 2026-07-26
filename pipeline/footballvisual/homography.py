"""Planar homography between the pitch plane and the image plane.

A football pitch is flat, so the map from pitch coordinates to broadcast image
coordinates is a plane-to-plane projective transform: an eight degree of
freedom 3x3 matrix applied to homogeneous coordinates. That is the entire
reason this project can turn a camera view into a top-down tactical map without
ever recovering full 3D camera pose.

The estimator here is a normalised DLT wrapped in RANSAC. It is written in
plain numpy rather than delegating to `cv2.findHomography` so that the maths is
inspectable and unit-testable without pulling OpenCV into the test path, and so
the failure modes (degenerate point configurations especially) are ours to
handle explicitly.

Coordinate conventions
----------------------
Pitch coordinates are metres, as defined in `pitch.py`. Image coordinates are
pixels, origin top-left, +y downward, which is what every video decoder hands
us. `H` always maps *pitch to image*; the inverse direction goes through
`invert()` so callers never have to remember which way a given matrix points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CORRESPONDENCES = 4


class HomographyError(RuntimeError):
    """Raised when a homography cannot be estimated from the given points."""


def _as_points(points: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {arr.shape}")
    return arr


def _normalisation_matrix(points: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centroid at origin, mean distance sqrt(2).

    Skipping this step is the single most common reason a hand-rolled DLT
    produces a numerically garbage homography. Pitch coordinates are order 50
    and image coordinates are order 1000, so the unnormalised design matrix has
    a condition number in the millions.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = float(np.sqrt((centred**2).sum(axis=1)).mean())
    if mean_dist < 1e-12:
        # All points coincident. Caller will fail the degeneracy check anyway,
        # but avoid dividing by zero on the way there.
        scale = 1.0
    else:
        scale = float(np.sqrt(2.0) / mean_dist)
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _collinear(points: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether every point lies on a single line.

    Four collinear correspondences satisfy the DLT constraints but leave the
    homography underdetermined, which shows up downstream as a matrix that
    reprojects the sample points perfectly and everything else nowhere near.
    Cheaper to reject here than to debug there.
    """
    if len(points) < 3:
        return True
    centred = points - points.mean(axis=0)
    # Smallest singular value near zero means no spread off the principal axis.
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[-1] < tol * max(singular[0], 1.0))


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least squares homography mapping `src` onto `dst` via normalised DLT.

    Each correspondence contributes two rows to a 2N x 9 design matrix; the
    solution is the right singular vector of the smallest singular value, i.e.
    the null space direction. With exactly four points the fit is exact; with
    more it is the algebraic least squares fit, which is what RANSAC refines.
    """
    src = _as_points(src)
    dst = _as_points(dst)
    if len(src) != len(dst):
        raise ValueError(f"point count mismatch: {len(src)} source vs {len(dst)} dest")
    if len(src) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"need at least {MIN_CORRESPONDENCES} correspondences, got {len(src)}"
        )
    if _collinear(src) or _collinear(dst):
        raise HomographyError("degenerate correspondences: points are collinear")

    t_src = _normalisation_matrix(src)
    t_dst = _normalisation_matrix(dst)
    src_n = _apply(t_src, src)
    dst_n = _apply(t_dst, dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)

    try:
        _, _, vt = np.linalg.svd(design)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical edge
        raise HomographyError(f"SVD failed to converge: {exc}") from exc

    h_normalised = vt[-1].reshape(3, 3)
    # Undo the normalisation: H = T_dst^-1 @ H_n @ T_src
    h = np.linalg.inv(t_dst) @ h_normalised @ t_src

    if abs(h[2, 2]) < 1e-12:
        raise HomographyError("estimated homography is degenerate (h33 ~ 0)")
    h = h / h[2, 2]

    if not np.all(np.isfinite(h)):
        raise HomographyError("estimated homography contains non-finite values")
    return h


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 projective matrix to (N, 2) points, returning (N, 2)."""
    pts = _as_points(points)
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    # Points on or behind the horizon divide by ~zero. Rather than emitting inf
    # and letting it poison a mean somewhere, clamp the magnitude while keeping
    # the sign, so the result stays a huge finite coordinate on the correct side
    # and the caller's plausibility filter can reject it. Clamping through
    # `sign` alone would send w = -1e-13 to exactly zero, which is the very
    # division this guard exists to prevent.
    eps = 1e-12
    safe_w = np.where(np.abs(w) < eps, np.where(w < 0.0, -eps, eps), w)
    return projected[:, :2] / safe_w


def project(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project points through homography `h`."""
    return _apply(np.asarray(h, dtype=np.float64), points)


def invert(h: np.ndarray) -> np.ndarray:
    """Invert a homography, normalised so h33 == 1."""
    h = np.asarray(h, dtype=np.float64)
    try:
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError as exc:
        raise HomographyError(f"homography is singular and cannot be inverted: {exc}") from exc
    if abs(inv[2, 2]) < 1e-12:
        raise HomographyError("inverted homography is degenerate")
    return inv / inv[2, 2]


def reprojection_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence Euclidean reprojection error, in `dst` units."""
    projected = project(h, src)
    return np.sqrt(((projected - _as_points(dst)) ** 2).sum(axis=1))


@dataclass
class RansacResult:
    """Outcome of a robust homography fit.

    `inliers` is a boolean mask over the input correspondences. Keeping the mask
    rather than the filtered points means a caller can report *which* named
    landmark was rejected, which is the difference between a usable calibration
    error message and "calibration failed".
    """

    h: np.ndarray
    inliers: np.ndarray
    error: np.ndarray
    iterations: int

    @property
    def num_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def mean_inlier_error(self) -> float:
        if self.num_inliers == 0:
            return float("inf")
        return float(self.error[self.inliers].mean())


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 6.0,
    max_iterations: int = 2000,
    confidence: float = 0.995,
    seed: int | None = 0,
) -> RansacResult:
    """Robustly fit a homography, tolerating mismatched correspondences.

    `threshold` is in `dst` units (pixels, for a pitch-to-image fit). Six pixels
    is deliberately loose: a human clicking a line intersection on a 1280-wide
    frame is routinely three or four pixels off, and an automatic line detector
    is worse, so a tight threshold rejects good points and starves the fit.

    Iteration count adapts to the inlier ratio found so far, so a clean set of
    correspondences terminates in a handful of rounds rather than always paying
    for `max_iterations`.
    """
    src = _as_points(src)
    dst = _as_points(dst)
    n = len(src)
    if n < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"need at least {MIN_CORRESPONDENCES} correspondences, got {n}"
        )

    rng = np.random.default_rng(seed)

    # With the minimum sample there is nothing to be robust about; fit directly.
    if n == MIN_CORRESPONDENCES:
        h = estimate_homography(src, dst)
        err = reprojection_error(h, src, dst)
        return RansacResult(h, np.ones(n, dtype=bool), err, 1)

    best_inliers = np.zeros(n, dtype=bool)
    best_h: np.ndarray | None = None
    iterations_run = 0
    budget = max_iterations

    while iterations_run < budget:
        iterations_run += 1
        idx = rng.choice(n, size=MIN_CORRESPONDENCES, replace=False)
        try:
            candidate = estimate_homography(src[idx], dst[idx])
        except HomographyError:
            # Degenerate sample (three collinear points is common on a pitch,
            # where landmarks share lines by construction). Draw again.
            continue

        err = reprojection_error(candidate, src, dst)
        inliers = err < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_h = candidate

            ratio = inliers.sum() / n
            if ratio > 0.0:
                # Standard adaptive stopping rule: how many draws until we are
                # `confidence` sure of having sampled one all-inlier set.
                denom = np.log(max(1e-12, 1.0 - ratio**MIN_CORRESPONDENCES))
                needed = np.log(max(1e-12, 1.0 - confidence)) / denom
                budget = int(min(max_iterations, max(needed, MIN_CORRESPONDENCES)))

    if best_h is None or best_inliers.sum() < MIN_CORRESPONDENCES:
        raise HomographyError(
            "RANSAC found no consensus set; correspondences are likely mismatched "
            f"(best support {int(best_inliers.sum())} of {n})"
        )

    # Refit on the full consensus set. The minimal-sample model got us the
    # right inliers; the least squares refit on all of them is the actual
    # answer and typically halves the error.
    refined = estimate_homography(src[best_inliers], dst[best_inliers])
    refined_err = reprojection_error(refined, src, dst)
    refined_inliers = refined_err < threshold
    if refined_inliers.sum() >= best_inliers.sum():
        return RansacResult(refined, refined_inliers, refined_err, iterations_run)

    err = reprojection_error(best_h, src, dst)
    return RansacResult(best_h, best_inliers, err, iterations_run)


@dataclass
class Calibration:
    """A homography plus the evidence behind it.

    Carrying the landmark names and residuals alongside the matrix is what lets
    the pipeline emit a calibration report a human can sanity check, instead of
    nine opaque floats.
    """

    h: np.ndarray
    landmark_names: list[str]
    image_points: np.ndarray
    pitch_points: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray

    @property
    def mean_error_px(self) -> float:
        if not self.inliers.any():
            return float("inf")
        return float(self.errors[self.inliers].mean())

    @property
    def max_error_px(self) -> float:
        if not self.inliers.any():
            return float("inf")
        return float(self.errors[self.inliers].max())

    def rejected(self) -> list[str]:
        return [n for n, ok in zip(self.landmark_names, self.inliers) if not ok]

    def image_to_pitch(self, points: np.ndarray) -> np.ndarray:
        return project(invert(self.h), points)

    def pitch_to_image(self, points: np.ndarray) -> np.ndarray:
        return project(self.h, points)

    def to_dict(self) -> dict:
        return {
            "h": self.h.tolist(),
            "landmarks": self.landmark_names,
            "meanErrorPx": round(self.mean_error_px, 3),
            "maxErrorPx": round(self.max_error_px, 3),
            "rejected": self.rejected(),
        }


def calibrate_from_landmarks(
    correspondences: dict[str, tuple[float, float]],
    threshold: float = 6.0,
) -> Calibration:
    """Fit a pitch-to-image homography from named landmark clicks.

    `correspondences` maps a name from `pitch.LANDMARKS` to where that point
    appears in the image, in pixels. This is the entry point for both a manual
    calibration file and an automatic line-intersection detector: both produce
    the same named-point dictionary, so only one estimator has to exist.
    """
    from .pitch import landmark

    if len(correspondences) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"need at least {MIN_CORRESPONDENCES} landmark correspondences, "
            f"got {len(correspondences)}"
        )

    names = sorted(correspondences)
    pitch_pts = np.asarray([landmark(n).xy for n in names], dtype=np.float64)
    image_pts = np.asarray([correspondences[n] for n in names], dtype=np.float64)

    result = ransac_homography(pitch_pts, image_pts, threshold=threshold)
    return Calibration(
        h=result.h,
        landmark_names=names,
        image_points=image_pts,
        pitch_points=pitch_pts,
        inliers=result.inliers,
        errors=result.error,
    )


def image_point_for_player(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """The image point to project for a player, given their bounding box.

    Projecting the box centre is the obvious thing and it is wrong: the
    homography maps the *ground plane*, and a player's centre floats roughly a
    metre above it, which on a shallow broadcast angle throws the position
    several metres up the pitch. The bottom-centre of the box is where the
    player meets the grass, so that is the point that actually lives on the
    plane the homography describes.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
