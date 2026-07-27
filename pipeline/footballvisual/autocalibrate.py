"""Calibrate a frame from its pitch markings, with nobody clicking anything.

The hard part of automatic calibration is not finding the lines. It is deciding
*which* line each detected one is. A pitch is a small set of long white lines,
many of them parallel and evenly spaced, and from a single frame a stretch of
white could be the halfway line or the edge of a penalty area. Get the
assignment wrong and you still obtain a perfectly self-consistent homography,
one that puts every player in the wrong half.

The approach here is hypothesis and verify.

1. Segment the line pixels: bright, unsaturated, surrounded by grass.
2. Fit straight segments and merge collinear ones into long lines.
3. Split the lines into two families by *vanishing point*. On a pitch every
   marking is parallel or perpendicular to the touchlines, so in the image they
   form two pencils, each converging on its own vanishing point. Grouping by
   angle instead does not work: perspective makes lines that are parallel on the
   grass span tens of degrees in the image.
4. Enumerate assignments of detected lines to model lines, take two from each
   family, and solve for the homography from their four intersections.
5. Score each candidate by reprojecting the *whole* pitch model and measuring
   how well it lands on the detected line pixels, using a distance transform.

Step 5 is what makes step 4 safe to be greedy about. A wrong assignment still
explains the four lines it was fitted to, and then puts the centre circle, the
D, and the six yard boxes somewhere with no white pixels at all, so it scores
badly. Fitting on a few lines and verifying against every marking on the pitch
is the whole idea.

The search is kept tractable by two constraints. Lines in a family preserve
their order under projection, so a detected line above another must map to a
model line on the same side; and a candidate homography must map the pitch to a
plausibly sized region of the image with the correct orientation, which is far
cheaper to check than a full score and rejects almost everything.

What this cannot do
-------------------
A pitch is symmetric, and no amount of line finding escapes that. Mirroring it
about the halfway line, or rotating it 180 degrees, maps every marking onto a
marking, so those fits explain the image exactly as well as the true one while
placing players on the wrong side or in the wrong half.

The mirror is resolved by `camera_side`, since knowing which touchline the
camera is behind fixes the orientation. The rotation is not resolvable from
geometry at all, and is reported through `rotation_ambiguous` rather than
guessed at. Resolving it needs information from outside the markings, which in a
real system means the direction of play, known kit colours, or an operator
confirming it once.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import pitch
from .homography import HomographyError, estimate_homography, project

_MODEL_SAMPLE_CACHE: dict[float, np.ndarray] = {}

# Sample spacing along pitch markings, in metres, for the two scoring passes.
COARSE_SPACING_M = 6.0
FINE_SPACING_M = 1.5

# A hypothesis that puts essentially the whole pitch model within a few pixels
# of a detected marking is not going to be beaten by anything meaningful, so the
# sweep stops rather than enumerating the remaining tens of thousands.
EARLY_EXIT_INLIERS = 0.995
EARLY_EXIT_SCORE_PX = 1.0

# Model lines, grouped by which family they belong to.
#
# `LINES_CONST_Y` run the length of the pitch (parallel to the touchlines) and
# are identified by their y. `LINES_CONST_X` run across it and are identified by
# their x. Only long, reliably visible markings are listed: the goal-area lines
# are included because they are strong edges, the arcs are not because they are
# not straight.
LINES_CONST_Y: tuple[float, ...] = (
    -pitch.HALF_WIDTH,
    -pitch.PENALTY_AREA_WIDTH / 2.0,
    -pitch.GOAL_AREA_WIDTH / 2.0,
    pitch.GOAL_AREA_WIDTH / 2.0,
    pitch.PENALTY_AREA_WIDTH / 2.0,
    pitch.HALF_WIDTH,
)

LINES_CONST_X: tuple[float, ...] = (
    -pitch.HALF_LENGTH,
    -(pitch.HALF_LENGTH - pitch.GOAL_AREA_LENGTH),
    -(pitch.HALF_LENGTH - pitch.PENALTY_AREA_LENGTH),
    0.0,
    pitch.HALF_LENGTH - pitch.PENALTY_AREA_LENGTH,
    pitch.HALF_LENGTH - pitch.GOAL_AREA_LENGTH,
    pitch.HALF_LENGTH,
)


@dataclass
class DetectedLine:
    """An infinite line in the image, in homogeneous form, with its support."""

    # (a, b, c) with a*x + b*y + c = 0, normalised so hypot(a, b) == 1.
    coeffs: np.ndarray
    # Total length of the segments that voted for it, in pixels. Used as a
    # confidence: a line supported by 400px of pitch marking is far more
    # trustworthy than one supported by 40px of a boot.
    support: float
    angle: float

    def intersect(self, other: "DetectedLine") -> tuple[float, float] | None:
        point = np.cross(self.coeffs, other.coeffs)
        if abs(point[2]) < 1e-9:
            return None  # parallel in the image
        return (float(point[0] / point[2]), float(point[1] / point[2]))


@dataclass
class AutoCalibration:
    """Result of an automatic fit."""

    h: np.ndarray
    score: float
    inlier_fraction: float
    num_lines: int
    hypotheses_scored: int
    # Fraction of the detected line pixels this fit accounts for. The other
    # direction of `score`, and the only one of the three that notices a pitch
    # shrunk onto a corner of the mask. See `_explained_fraction`.
    explained_fraction: float = 0.0
    assignment: dict = field(default_factory=dict)
    # The equally-good fit with the pitch rotated 180 degrees, when one exists.
    # See `rotation_ambiguous`.
    h_rotated: np.ndarray | None = None
    # True when a caller-supplied prior was used to settle that ambiguity.
    resolved_by_prior: bool = False

    @property
    def rotation_ambiguous(self) -> bool:
        """Whether the markings alone cannot say which goal is which.

        A pitch is symmetric under a 180 degree rotation: swap the two halves
        and every line lands back on a line. That rotation also preserves
        orientation, so unlike the mirror ambiguity it cannot be resolved by
        knowing which touchline the camera is behind, and the two fits score
        identically by construction.

        This is a property of the problem, not a weakness of this
        implementation. Line markings simply do not encode which end is which,
        and a real system resolves it from something outside the geometry: the
        scoreboard, the direction of play, known kit colours, or an operator
        confirming it once at kickoff.
        """
        return self.h_rotated is not None

    @property
    def is_confident(self) -> bool:
        """Whether this fit should be trusted without a human looking at it.

        All three terms matter, and each was added because the ones before it
        were not enough. A low mean distance says the reprojected model sits on
        white pixels. A high inlier fraction says *most* of the model does, not
        just the few lines it was fitted to, which a wrong assignment rarely
        achieves.

        Both of those only ever look outward from the model, and a fit that
        shrinks the pitch onto a dense patch of the mask passes them both while
        being 94 metres wrong: score 0.85, inlier fraction 0.98, and confident.
        That fit explains 5% of the detected line pixels, against 81% for the
        truth. So the third term asks the question from the other side, and the
        threshold sits far below any correct fit measured rather than tuned
        close to that one counterexample.
        """
        return (
            self.score < 4.0
            and self.inlier_fraction > 0.55
            and self.explained_fraction > 0.35
        )


def _odd(value: float, minimum: int = 3) -> int:
    """Nearest odd integer at least `minimum`, for morphology kernels."""
    k = int(round(value))
    if k % 2 == 0:
        k += 1
    return max(minimum, k)


def pitch_region(frame: np.ndarray, erode_frac: float = 0.012) -> np.ndarray:
    """The playing surface: the largest connected patch of grass, pulled in.

    Taking the *largest connected component* rather than all green matters,
    because a stadium has green in the stands, on advertising, and on kit. And
    eroding matters even more: the advertising hoardings run directly along the
    touchline, so a mask that merely asks for white "near grass" accepts the
    whole boarding as pitch marking. On a real Premier League frame that single
    difference was the bulk of the problem, with only 0.52% of the frame being
    white-inside-the-pitch against 4.21% white overall.

    The colour bounds are deliberately loose, and tightening them was tried and
    reverted. They are loose enough to admit a sky blue crowd under floodlights,
    which the closing step then bridges into the pitch component, so the returned
    surface reaches row 0 on every input tested and quietly contains the stands.
    Tightening to hue 32 to 88 at saturation 40 fixes that cleanly, cutting the
    surface to 70% of the frame and stopping it at the horizon, and it makes
    calibration much *worse*: on the demo clip the fit goes from 0.09m to 94m
    out, because the erosion that follows also pulls the boundary in off the far
    touchline, and losing that one line costs five of the twenty-two detected
    lines and the constraint that pins the far side of the pitch.

    So the stands stay in. What keeps them out of the answer is `line_mask`,
    whose top-hat is unmoved by anything broad, rather than this region. The
    region's real job is connectivity, which is a reason to prefer it generous.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([100, 255, 255]))

    # Close small holes so players standing on the grass do not fragment it.
    height, width = frame.shape[:2]
    scale = min(height, width)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((_odd(scale / 40), _odd(scale / 40)), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(grass, 8)
    if count <= 1:
        return np.zeros_like(grass)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8) * 255

    erode = _odd(scale * erode_frac)
    return cv2.erode(region, np.ones((erode, erode), np.uint8))


def line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of pitch markings.

    Two things make this work on real broadcast rather than only on a clean
    render, and both were found by running it on a real frame.

    The mask is restricted to the playing surface rather than to anything near
    grass. Advertising hoardings sit right on the touchline and the score bug
    sits over the crowd, both bright and unsaturated, so a proximity test lets
    them straight in and they then dominate the line fitting. Note that
    `pitch_region` is generous and does leak into the stands, deliberately: see
    its docstring for why tightening it made calibration far worse. It is the
    top-hat below, not the region, that keeps the crowd out of the answer.

    Markings are found as *local* brightness, through a white top-hat, rather
    than by any global threshold on the value channel. This is the difference
    between working on one clip and working on both. A synthetic render puts its
    lines at value 231 over grass at 94, so almost any global rule finds them; a
    floodlit broadcast pitch has mow stripes and a lighting gradient that between
    them span a wider range than the gap between paint and grass, so a global
    rule either takes half the pitch or none of it. Two attempts failed here
    before this one. A high percentile of the surface's brightness lands on lit
    grass rather than paint on the render (96th percentile is 119, the lines are
    at 231), and Otsu on the value channel splits floodlit grass into lit and
    shaded halves instead of grass from paint, taking 77% of a real frame. The
    top-hat asks the only question that is actually true of a pitch marking on
    any of these inputs: is this pixel brighter than its own surroundings, on the
    scale of a painted line.

    Morphology kernels scale with the frame throughout. A structuring element
    sized for a line at 1280x720 is twice too large at 640x360, which is the
    resolution real broadcast clips actually arrive at.
    """
    height, width = frame.shape[:2]
    scale = min(height, width)

    interior = pitch_region(frame)
    if not interior.any():
        return np.zeros((height, width), dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    inside = interior > 0

    # The structuring element must be comfortably wider than a line and narrower
    # than the gaps between them, so the response peaks on paint and flattens on
    # everything broad, including a mow stripe or a floodlight gradient.
    top_hat = cv2.morphologyEx(
        value,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(scale / 40), _odd(scale / 40))),
    )

    # Otsu is safe *here*, on the top-hat response, in a way it is not on raw
    # brightness: the response really is bimodal, near zero on grass and high on
    # paint, whatever the exposure. The floor stops it from manufacturing a split
    # in a frame that contains no markings at all, where the response is noise.
    otsu, _ = cv2.threshold(
        top_hat[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = (inside & (top_hat >= max(float(otsu), 12.0))).astype(np.uint8) * 255

    close_k = _odd(scale / 180)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))


def _normalise_line(x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    a = y2 - y1
    b = x1 - x2
    norm = float(np.hypot(a, b))
    if norm < 1e-9:
        return None
    a, b = a / norm, b / norm
    c = -(a * x1 + b * y1)
    # Fix the sign so two representations of the same line compare equal.
    if a < 0 or (abs(a) < 1e-9 and b < 0):
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float64)


def extract_lines(
    mask: np.ndarray,
    min_length: int = 55,
    merge_angle_deg: float = 2.5,
    merge_offset_px: float = 14.0,
    max_lines: int = 40,
) -> list[DetectedLine]:
    """Find long straight markings, merging collinear fragments.

    Players standing on a line break it into pieces, so the raw segments from a
    Hough transform are fragments of the real markings. Merging by angle and
    perpendicular offset reassembles them, which matters because the ordering
    constraint in the search assumes one entry per physical line.
    """
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360.0,
        threshold=55,
        minLineLength=min_length,
        maxLineGap=22,
    )
    if segments is None:
        return []

    # OpenCV 4 returns (N, 1, 4) and OpenCV 5 returns (N, 4). Normalise rather
    # than indexing one shape, so this works across both.
    segments = np.asarray(segments).reshape(-1, 4)

    clusters: list[dict] = []
    for seg in segments:
        x1, y1, x2, y2 = (float(v) for v in seg)
        coeffs = _normalise_line(x1, y1, x2, y2)
        if coeffs is None:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(-coeffs[0], coeffs[1])) % 180.0)

        for cluster in clusters:
            delta = abs(cluster["angle"] - angle)
            delta = min(delta, 180.0 - delta)
            if delta > merge_angle_deg:
                continue
            # Perpendicular distance between the two lines, measured at the
            # cluster's midpoint so near-parallel lines far apart are not merged.
            mid = cluster["mid"]
            offset = abs(float(coeffs[0] * mid[0] + coeffs[1] * mid[1] + coeffs[2]))
            if offset > merge_offset_px:
                continue
            cluster["points"].extend([(x1, y1), (x2, y2)])
            cluster["support"] += length
            pts = np.array(cluster["points"])
            cluster["mid"] = pts.mean(axis=0)
            break
        else:
            clusters.append(
                {
                    "points": [(x1, y1), (x2, y2)],
                    "support": length,
                    "angle": angle,
                    "mid": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]),
                }
            )

    lines: list[DetectedLine] = []
    for cluster in clusters:
        pts = np.array(cluster["points"], dtype=np.float64)
        if len(pts) < 2:
            continue
        # Total least squares through the cluster's endpoints: the principal
        # direction is the line, which is more accurate than any single segment.
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        direction = vt[0]
        coeffs = _normalise_line(
            centre[0], centre[1], centre[0] + direction[0], centre[1] + direction[1]
        )
        if coeffs is None:
            continue
        angle = float(np.degrees(np.arctan2(-coeffs[0], coeffs[1])) % 180.0)
        lines.append(DetectedLine(coeffs=coeffs, support=cluster["support"], angle=angle))

    lines.sort(key=lambda l: l.support, reverse=True)
    return lines[:max_lines]


def _normalised_coeffs(lines: list[DetectedLine], width: int, height: int) -> np.ndarray:
    """Line coefficients expressed in image coordinates scaled to about [-1, 1].

    Vanishing points of a pitch are routinely thousands of pixels outside the
    frame, and in raw pixels the algebraic error used to test whether a line
    passes through one is badly scaled. Normalising first makes a single
    tolerance meaningful for both near and distant vanishing points.
    """
    scale = max(width, height) / 2.0
    cx, cy = width / 2.0, height / 2.0
    out = np.empty((len(lines), 3), dtype=np.float64)
    for i, line in enumerate(lines):
        a, b, c = line.coeffs
        out[i] = (a, b, (a * cx + b * cy + c) / scale)
    return out


def split_families(
    lines: list[DetectedLine],
    width: int,
    height: int,
    tolerance: float = 0.03,
) -> tuple[list[DetectedLine], list[DetectedLine]]:
    """Split lines into the two pencils a pitch produces.

    Grouped by vanishing point, not by angle. Lines that are parallel on the
    pitch are *not* parallel in the image: perspective makes the touchlines
    converge, so in a typical broadcast frame the members of one family can span
    tens of degrees while a member of the other family sits right between them.
    Any angle-based split therefore mixes the families on exactly the wide-angle
    views where calibration matters most.

    What does hold is that each family passes through a common vanishing point,
    so the split is a two-model RANSAC: find the point most lines agree on,
    take those as one family, and repeat on what is left.
    """
    if len(lines) < 4:
        return [], []

    coeffs = _normalised_coeffs(lines, width, height)
    supports = np.array([l.support for l in lines])

    def best_vanishing_point(indices: np.ndarray) -> tuple[np.ndarray, float]:
        """Highest-support consensus set among `indices`."""
        best_mask = np.zeros(len(indices), dtype=bool)
        best_weight = -1.0
        subset = coeffs[indices]
        for i, j in itertools.combinations(range(len(indices)), 2):
            v = np.cross(subset[i], subset[j])
            norm = float(np.linalg.norm(v))
            if norm < 1e-9:
                continue
            v = v / norm
            residual = np.abs(subset @ v)
            mask = residual < tolerance
            weight = float(supports[indices][mask].sum())
            if weight > best_weight:
                best_weight = weight
                best_mask = mask
        return best_mask, best_weight

    all_idx = np.arange(len(lines))
    mask_a, _ = best_vanishing_point(all_idx)
    idx_a = all_idx[mask_a]
    idx_rest = all_idx[~mask_a]

    if len(idx_rest) < 2:
        return [lines[i] for i in idx_a], []

    mask_b, _ = best_vanishing_point(idx_rest)
    idx_b = idx_rest[mask_b]

    family_a = [lines[i] for i in idx_a]
    family_b = [lines[i] for i in idx_b]

    # Order them so the larger family is first only for determinism; the caller
    # tries both role assignments anyway.
    if len(family_b) > len(family_a):
        family_a, family_b = family_b, family_a
    return family_a, family_b


def _order_key(line: DetectedLine, width: int, height: int) -> float:
    """Where a line sits, for the monotonic ordering constraint.

    Evaluated at the image centre: for a family of near-parallel lines this
    orders them consistently along the direction they are stacked in.
    """
    a, b, c = line.coeffs
    cx, cy = width / 2.0, height / 2.0
    if abs(b) > abs(a):
        return -(a * cx + c) / b  # y at the centre column
    return -(b * cy + c) / a  # x at the centre row


def _signed_area(h: np.ndarray) -> float | None:
    """Shoelace area of the pitch's four corners projected into the image.

    The *sign* is the useful part. It says which way round the pitch has been
    mapped, which is equivalent to saying which touchline the camera is behind.
    """
    corners = np.array(
        [
            [-pitch.HALF_LENGTH, -pitch.HALF_WIDTH],
            [pitch.HALF_LENGTH, -pitch.HALF_WIDTH],
            [pitch.HALF_LENGTH, pitch.HALF_WIDTH],
            [-pitch.HALF_LENGTH, pitch.HALF_WIDTH],
        ]
    )
    try:
        image = project(h, corners)
    except (HomographyError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(image)):
        return None

    area = 0.0
    for i in range(4):
        x1, y1 = image[i]
        x2, y2 = image[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return area


def _plausible(h: np.ndarray, width: int, height: int, orientation: float) -> bool:
    """Cheap rejection of nonsense homographies before paying to score them.

    Almost every hypothesis in the search is wrong, and most are wrong in ways
    that are obvious from where they put the pitch: inside out, collapsed to a
    sliver, or a thousand frames off screen. Catching those here is what keeps
    the search affordable.

    `orientation` is the required sign of the projected pitch's shoelace area,
    which encodes which touchline the camera sits behind. Note that image y
    points *down*, so a camera on the negative-y touchline produces a
    **negative** area: getting this sign backwards does not fail loudly, it
    quietly selects the pitch mirrored about the halfway line, which reprojects
    onto the real markings perfectly because a pitch is symmetric.
    """
    area = _signed_area(h)
    if area is None:
        return False

    if np.abs(area) < 1e-9:
        return False
    if np.sign(area) != np.sign(orientation):
        return False

    # A real view of a pitch covers a meaningful part of the frame, and cannot
    # legitimately be tens of screens across.
    covered = abs(area) / 2.0
    if covered < 0.25 * width * height:
        return False
    if covered > 4000.0 * width * height:
        return False

    return True


def _model_samples(spacing_m: float) -> np.ndarray:
    """Points along every pitch marking, at roughly `spacing_m` intervals.

    Cached because it is a constant of the pitch, and the search scores tens of
    thousands of hypotheses against it. Rebuilding it per call made scoring the
    dominant cost of calibration.
    """
    cached = _MODEL_SAMPLE_CACHE.get(spacing_m)
    if cached is not None:
        return cached

    samples: list[np.ndarray] = []
    for poly in pitch.pitch_polylines():
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            steps = max(2, int(np.linalg.norm(b - a) / spacing_m))
            t = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
            samples.append(a + (b - a) * t)

    model = np.vstack(samples) if samples else np.zeros((0, 2))
    _MODEL_SAMPLE_CACHE[spacing_m] = model
    return model


def _score_homography(
    h: np.ndarray, distance: np.ndarray, width: int, height: int, spacing_m: float = 1.5
) -> tuple[float, float]:
    """How well the reprojected pitch model lands on detected line pixels.

    Returns `(mean distance in pixels, fraction within tolerance)`. Sampling the
    model's own polylines and reading a distance transform is far more robust
    than trying to match detected lines to model lines one to one, because it
    naturally rewards a homography that explains the curved markings and the
    boxes as well as the straight lines it was fitted to.

    `spacing_m` trades accuracy for speed. The search uses a coarse pass to rank
    candidates and re-scores only the survivors finely, because a hypothesis
    that is wrong is obviously wrong at any resolution.
    """
    model = _model_samples(spacing_m)
    if not len(model):
        return float("inf"), 0.0

    try:
        image = project(h, model)
    except (HomographyError, np.linalg.LinAlgError):
        return float("inf"), 0.0

    inside = (
        np.isfinite(image).all(axis=1)
        & (image[:, 0] >= 0)
        & (image[:, 0] < width)
        & (image[:, 1] >= 0)
        & (image[:, 1] < height)
    )
    # Require a real amount of the model to be visible. Otherwise a homography
    # that squeezes the pitch into one clean corner scores beautifully on the
    # three points that happen to land there.
    if inside.sum() < 0.1 * len(model):
        return float("inf"), 0.0

    xs = image[inside, 0].astype(np.int32)
    ys = image[inside, 1].astype(np.int32)
    d = distance[ys, xs]

    return float(d.mean()), float((d < 4.0).mean())


def _explained_fraction(
    h: np.ndarray, mask: np.ndarray, width: int, height: int, tolerance_px: int = 6
) -> float:
    """Fraction of the detected line pixels that the fitted pitch accounts for.

    This is the other half of `_score_homography`, and without it the scoring is
    one-sided in a way that admits a confidently wrong answer. That function asks
    only whether every model marking lands on a detected line pixel. A homography
    that shrinks the pitch down onto a dense patch of the mask satisfies that
    completely, scoring under a pixel with a 0.98 inlier fraction while sitting
    94 metres from the truth, because nothing ever asks about the detected lines
    it left unexplained. Measured on that exact fit, this returns 5%, against 81%
    for the correct one and 81% for the ground truth homography.

    Computed once for the winner rather than per hypothesis. It needs a dilation
    over the whole frame, which is far too expensive to run tens of thousands of
    times, and it is a check on the answer rather than a way to find it.
    """
    pixels = mask > 0
    if not pixels.any():
        return 0.0

    # Sample finer than the scoring pass: this rasterises the model into an
    # image, so gaps between samples would read as unexplained line pixels.
    try:
        image = project(h, _model_samples(0.75))
    except (HomographyError, np.linalg.LinAlgError):
        return 0.0

    on_screen = (
        np.isfinite(image).all(axis=1)
        & (image[:, 0] >= 0)
        & (image[:, 0] < width)
        & (image[:, 1] >= 0)
        & (image[:, 1] < height)
    )
    points = image[on_screen].astype(np.int32)
    if not len(points):
        return 0.0

    drawn = np.zeros((height, width), dtype=np.uint8)
    drawn[points[:, 1], points[:, 0]] = 255
    kernel = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    near_model = cv2.dilate(drawn, kernel) > 0
    return float(near_model[pixels].mean())


def _rotate180(h: np.ndarray) -> np.ndarray:
    """The same fit with the pitch model turned through 180 degrees."""
    rotated = h @ np.diag([-1.0, -1.0, 1.0])
    if abs(rotated[2, 2]) > 1e-12:
        rotated = rotated / rotated[2, 2]
    return rotated


def _agreement_with_prior(h: np.ndarray, prior: np.ndarray) -> float:
    """Mean pitch-space disagreement between two homographies, in metres."""
    grid = np.array(
        [[x, y] for x in (-40.0, -20.0, 0.0, 20.0, 40.0) for y in (-25.0, 0.0, 25.0)]
    )
    try:
        return float(np.linalg.norm(project(h, grid) - project(prior, grid), axis=1).mean())
    except (HomographyError, np.linalg.LinAlgError):
        return float("inf")


def calibrate_auto(
    frame: np.ndarray,
    camera_side: str = "minus_y",
    prior_h: np.ndarray | None = None,
    max_lines_per_family: int = 5,
    max_hypotheses: int = 40000,
) -> AutoCalibration | None:
    """Fit a pitch-to-image homography from the frame's markings alone.

    `camera_side` says which touchline the camera is behind, and it is required
    rather than inferred because it cannot be inferred. A pitch is mirror
    symmetric about the halfway line, so the mirrored homography explains the
    markings exactly as well as the true one while placing every player on the
    wrong side of the pitch. `"minus_y"` is the broadcast main-camera
    convention used by this project's renderer; pass `"plus_y"` for a camera on
    the far touchline.

    `prior_h` resolves the 180 degree ambiguity. Pass the last known good
    homography and the variant closer to it is chosen, which is what makes
    automatic re-calibration after a camera cut usable: the ambiguity is settled
    once, then carried across cuts. Without a prior the choice between the two
    is arbitrary, and `rotation_ambiguous` says so.

    Returns None when the frame does not contain enough structure to identify.
    Callers should check `is_confident` before trusting the result, and
    `rotation_ambiguous` before trusting which end is which: a low-confidence
    answer is worth showing to a human for approval, not worth silently
    calibrating on.
    """
    if camera_side not in ("minus_y", "plus_y"):
        raise ValueError(f"camera_side must be 'minus_y' or 'plus_y', got {camera_side!r}")
    # Image y points down, so a camera on the negative-y touchline maps the
    # pitch to a negatively-oriented quad.
    orientation = -1.0 if camera_side == "minus_y" else 1.0

    height, width = frame.shape[:2]

    mask = line_mask(frame)
    lines = extract_lines(mask)
    if len(lines) < 4:
        return None

    family_a, family_b = split_families(lines, width, height)
    if len(family_a) < 2 or len(family_b) < 2:
        return None

    family_a = sorted(family_a, key=lambda l: l.support, reverse=True)[:max_lines_per_family]
    family_b = sorted(family_b, key=lambda l: l.support, reverse=True)[:max_lines_per_family]
    family_a.sort(key=lambda l: _order_key(l, width, height))
    family_b.sort(key=lambda l: _order_key(l, width, height))

    # Distance to the nearest detected line pixel, for scoring. Inverted first
    # because distanceTransform measures distance to the nearest zero.
    distance = cv2.distanceTransform(
        cv2.bitwise_not(mask), cv2.DIST_L2, 3
    ).astype(np.float32)

    best: AutoCalibration | None = None
    scored = 0
    considered = 0

    # Both families could be the constant-y set or the constant-x set: the
    # camera might be behind a goal rather than at the halfway line. Trying both
    # assignments costs one extra pass and removes an assumption about camera
    # placement that would otherwise be silently baked in.
    for a_values, b_values in (
        (LINES_CONST_Y, LINES_CONST_X),
        (LINES_CONST_X, LINES_CONST_Y),
    ):
        a_is_const_y = a_values is LINES_CONST_Y

        for (ia, ja) in itertools.combinations(range(len(family_a)), 2):
            for (ka, la) in itertools.combinations(range(len(a_values)), 2):
                # Projection preserves order within a pencil, so the two
                # detected lines map to the two model lines either in order or
                # reversed, never crossed.
                for a_pair in ((ka, la), (la, ka)):
                    for (ib, jb) in itertools.combinations(range(len(family_b)), 2):
                        for (kb, lb) in itertools.combinations(range(len(b_values)), 2):
                            for b_pair in ((kb, lb), (lb, kb)):
                                considered += 1
                                if considered > max_hypotheses:
                                    break
                                if (
                                    best is not None
                                    and best.inlier_fraction >= EARLY_EXIT_INLIERS
                                    and best.score <= EARLY_EXIT_SCORE_PX
                                ):
                                    break

                                pitch_pts = []
                                image_pts = []
                                ok = True
                                for a_idx, a_val in ((ia, a_pair[0]), (ja, a_pair[1])):
                                    for b_idx, b_val in ((ib, b_pair[0]), (jb, b_pair[1])):
                                        point = family_a[a_idx].intersect(family_b[b_idx])
                                        if point is None:
                                            ok = False
                                            break
                                        if a_is_const_y:
                                            pitch_pts.append(
                                                (b_values[b_val], a_values[a_val])
                                            )
                                        else:
                                            pitch_pts.append(
                                                (a_values[a_val], b_values[b_val])
                                            )
                                        image_pts.append(point)
                                    if not ok:
                                        break
                                if not ok:
                                    continue

                                try:
                                    h = estimate_homography(
                                        np.array(pitch_pts), np.array(image_pts)
                                    )
                                except HomographyError:
                                    continue

                                if not _plausible(h, width, height, orientation):
                                    continue

                                # Coarse pass: a wrong hypothesis is obviously
                                # wrong at low resolution, so the expensive fine
                                # score is spent only on the shortlist below.
                                mean_d, inliers = _score_homography(
                                    h, distance, width, height, spacing_m=COARSE_SPACING_M
                                )
                                scored += 1
                                if best is None or (
                                    inliers > best.inlier_fraction + 0.02
                                    or (
                                        abs(inliers - best.inlier_fraction) <= 0.02
                                        and mean_d < best.score
                                    )
                                ):
                                    best = AutoCalibration(
                                        h=h,
                                        score=mean_d,
                                        inlier_fraction=inliers,
                                        num_lines=len(lines),
                                        hypotheses_scored=scored,
                                        assignment={
                                            "familyAIsConstY": a_is_const_y,
                                            "familyA": [a_values[a_pair[0]], a_values[a_pair[1]]],
                                            "familyB": [b_values[b_pair[0]], b_values[b_pair[1]]],
                                        },
                                    )

    if best is None:
        return None

    # Re-score the winner at full resolution so the reported numbers, and the
    # confidence threshold that keys off them, mean what they say.
    best.score, best.inlier_fraction = _score_homography(
        best.h, distance, width, height, spacing_m=FINE_SPACING_M
    )
    best.hypotheses_scored = scored
    best.explained_fraction = _explained_fraction(best.h, mask, width, height)

    # A 180 degree rotation of the pitch maps every marking onto a marking and
    # preserves orientation, so it survives the camera-side constraint and
    # scores identically. Rather than pick one silently, hand the caller both.
    rotated = _rotate180(best.h)
    if _plausible(rotated, width, height, orientation):
        _, rot_inliers = _score_homography(
            rotated, distance, width, height, spacing_m=FINE_SPACING_M
        )
        if abs(rot_inliers - best.inlier_fraction) < 0.05:
            best.h_rotated = rotated

    # With a prior available the ambiguity is decidable: keep whichever variant
    # agrees with where the pitch was last known to be.
    if prior_h is not None and best.h_rotated is not None:
        if _agreement_with_prior(best.h_rotated, prior_h) < _agreement_with_prior(
            best.h, prior_h
        ):
            best.h, best.h_rotated = best.h_rotated, best.h
            best.score, best.inlier_fraction = _score_homography(
                best.h, distance, width, height, spacing_m=FINE_SPACING_M
            )
        best.resolved_by_prior = True

    return best
