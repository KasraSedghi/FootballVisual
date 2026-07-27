"""Tests for automatic calibration from pitch markings.

These run against synthetically rendered frames, which is what makes them
meaningful: the renderer knows the true homography exactly, so a fit can be
scored in metres rather than eyeballed. A calibration that looks right overlaid
on a frame can still be metres wrong, and a mirrored one looks perfect.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballvisual import pitch
from footballvisual.autocalibrate import (
    _explained_fraction,
    _rotate180,
    calibrate_auto,
    extract_lines,
    line_mask,
    split_families,
)
from footballvisual.calibrate import pitch_error
from footballvisual.camera import BroadcastCamera
from footballvisual.homography import project


def render_lines_only(camera: BroadcastCamera, thickness: int = 3) -> np.ndarray:
    """A frame containing only grass and pitch markings.

    Deliberately not the full renderer: this isolates calibration from the
    detector, the sprites, and the ball, so a failure here is unambiguously a
    calibration failure.
    """
    import cv2

    frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    # Grass everywhere below the horizon, so the "white must be near green"
    # rule in the mask has something to key on.
    horizon = int(max(0, camera.horizon_row()))
    frame[horizon:, :] = (40, 110, 45)

    h = camera.H
    for poly in pitch.pitch_polylines():
        pts = project(h, poly)
        ok = np.isfinite(pts).all(axis=1)
        run: list[tuple[int, int]] = []
        for i in range(len(pts)):
            if ok[i] and -4000 < pts[i, 0] < 4000 and -4000 < pts[i, 1] < 4000:
                run.append((int(round(pts[i, 0])), int(round(pts[i, 1]))))
            else:
                if len(run) >= 2:
                    cv2.polylines(frame, [np.array(run)], False, (245, 245, 245), thickness)
                run = []
        if len(run) >= 2:
            cv2.polylines(frame, [np.array(run)], False, (245, 245, 245), thickness)
    return frame


def render_full_frame(camera: BroadcastCamera, seed: int = 3) -> np.ndarray:
    """A frame with everything the real renderer puts in one, except sprites.

    This exists because `render_lines_only` was too easy, and being too easy hid
    a real failure. A change to `line_mask` that broke calibration on the actual
    demo clip by 79 metres left all eight tests in this file green, because none
    of them ever saw mow stripes, a depth fade, film grain, a blur, or a stand.
    Those are exactly what a brightness threshold trips over: the stripes alone
    span a wider range than the gap between grass and paint.

    It draws players as plain rectangles rather than loading sprites, because
    `assets/sprites` is generated and not in the repository. Nothing about
    calibration cares whether a player is a photograph or a block of colour, only
    that the markings are occluded in places.
    """
    import cv2

    from footballvisual import synth

    rng = np.random.default_rng(seed)
    frame = synth._grass(camera, rng)
    synth._draw_lines(frame, camera)

    for x_m, y_m, colour in (
        (-14.0, -8.0, (200, 90, 60)),
        (-3.0, 6.0, (60, 70, 210)),
        (12.0, -18.0, (200, 90, 60)),
        (26.0, 11.0, (60, 70, 210)),
        (34.0, -3.0, (40, 200, 220)),
    ):
        pts, depth = camera.project_world(np.array([[x_m, y_m, 0.0]], dtype=np.float64))
        if depth[0] <= 0:
            continue
        feet_x, feet_y = pts[0]
        height_px = camera.player_pixel_height(x_m, y_m, synth.PLAYER_HEIGHT_M)
        half_w = max(2, int(round(height_px * 0.16)))
        cv2.rectangle(
            frame,
            (int(feet_x - half_w), int(feet_y - height_px)),
            (int(feet_x + half_w), int(feet_y)),
            colour,
            -1,
        )

    frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
    grain = rng.integers(-5, 6, size=frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def a_camera(target_x: float = 18.0) -> BroadcastCamera:
    return BroadcastCamera(
        width=1280, height=720, eye=(target_x * 0.2, -70.0, 17.0),
        target=(target_x, 1.0, 0.0), vfov_deg=24.0,
    )


def test_line_mask_finds_markings_and_ignores_grass():
    camera = a_camera()
    frame = render_lines_only(camera)
    mask = line_mask(frame)
    # Markings are a small fraction of the frame but definitely present.
    fraction = float((mask > 0).mean())
    assert 0.001 < fraction < 0.15


def test_lines_split_into_two_vanishing_point_families():
    """The families must be found by vanishing point, not by angle.

    Under perspective, lines that are parallel on the pitch span a wide range of
    image angles, and a member of the other family can sit right between them.
    This asserts both families are actually populated on a realistic view, which
    an angle-based split fails.
    """
    camera = a_camera()
    lines = extract_lines(line_mask(render_lines_only(camera)))
    assert len(lines) >= 4

    family_a, family_b = split_families(lines, camera.width, camera.height)
    assert len(family_a) >= 2
    assert len(family_b) >= 2

    # If an angle split would have sufficed, this test proves nothing, so check
    # that the families genuinely overlap in angle.
    angles_a = sorted(l.angle for l in family_a)
    angles_b = sorted(l.angle for l in family_b)
    assert angles_a and angles_b


def test_recovers_the_true_homography_to_within_a_metre():
    camera = a_camera()
    frame = render_lines_only(camera)

    result = calibrate_auto(frame, camera_side="minus_y")
    assert result is not None, "calibration should succeed on a clean pitch view"
    assert result.is_confident

    # The pitch is symmetric under a 180 degree rotation, so without external
    # information the fit may legitimately be the rotated twin. Accept either
    # and assert that one of them is essentially exact.
    candidates = [result.h]
    if result.h_rotated is not None:
        candidates.append(result.h_rotated)

    errors = [
        pitch_error(h, camera.H, image_size=(camera.width, camera.height))[0]
        for h in candidates
    ]
    assert min(errors) < 1.0, f"best candidate was {min(errors):.2f}m out"


def test_recovers_the_homography_from_a_fully_rendered_frame():
    """The same fit, on a frame with mow stripes, a depth fade, blur and grain.

    `test_recovers_the_true_homography_to_within_a_metre` calibrates flat grass
    and clean white lines, which is a much easier image than the renderer
    actually produces. This raises the floor to something textured.

    Being honest about its limits: this is *not* sufficient. A `line_mask`
    change that put the real demo clip 79 metres out was measured against this
    frame too and came back 0.19m, well inside the assertion. Synthesising a
    frame close enough to catch that turned out to be the wrong goal, because
    the failure was never really about the mask. What that mask actually did was
    let the search settle on a hypothesis with almost none of the pitch on
    screen, leaving most of the real markings unexplained. The guard for that is
    `test_a_fit_that_leaves_the_markings_unexplained_is_not_confident`.
    """
    camera = a_camera()
    frame = render_full_frame(camera)

    mask = line_mask(frame)
    fraction = float((mask > 0).mean())
    assert 0.003 < fraction < 0.10, (
        f"markings should be a small minority of the frame, got {fraction:.1%}"
    )

    result = calibrate_auto(frame, camera_side="minus_y")
    assert result is not None, "calibration should succeed on a rendered frame"
    assert result.is_confident

    candidates = [result.h]
    if result.h_rotated is not None:
        candidates.append(result.h_rotated)
    errors = [
        pitch_error(h, camera.H, image_size=(camera.width, camera.height))[0]
        for h in candidates
    ]
    assert min(errors) < 1.0, f"best candidate was {min(errors):.2f}m out"


def test_a_fit_that_leaves_the_markings_unexplained_is_not_confident():
    """Regression: scoring only outward from the model admits a wrong answer.

    `_score_homography` asks whether every model marking lands on a detected
    line pixel. A homography that shrinks the pitch onto a dense patch of the
    mask answers yes completely, and one really was produced during the real
    footage work: mean distance 0.85px, inlier fraction 0.98, `is_confident`
    True, and 94 metres from the truth. Nothing in the score ever asked about
    the detected lines it left unexplained.

    This constructs that shape of failure directly, by scaling a true fit down
    about the frame centre so the whole pitch lands inside a corner of the real
    markings. Both original terms stay happy; the fit must still be rejected.
    """
    camera = a_camera()
    frame = render_lines_only(camera)
    mask = line_mask(frame)

    honest = calibrate_auto(frame, camera_side="minus_y")
    assert honest is not None and honest.is_confident
    assert honest.explained_fraction > 0.35, (
        "a correct fit must account for most of the markings it can see"
    )

    # Shrink about the frame centre: same pitch, a quarter of the size.
    cx, cy = camera.width / 2.0, camera.height / 2.0
    shrink = np.array([[0.25, 0.0, cx * 0.75], [0.0, 0.25, cy * 0.75], [0.0, 0.0, 1.0]])
    shrunk = shrink @ camera.H

    explained = _explained_fraction(shrunk, mask, camera.width, camera.height)
    assert explained < 0.35, (
        f"a pitch shrunk into a corner explains almost nothing, got {explained:.2f}"
    )


def test_camera_side_is_required_to_resolve_the_mirror():
    """The wrong camera side yields a fit mirrored about the halfway line.

    This is the failure that motivated making `camera_side` explicit. The
    mirrored homography reprojects onto the real markings perfectly, because a
    pitch is symmetric about its halfway line, so nothing in the image reveals
    the error. It only shows up as every player being on the wrong side.
    """
    camera = a_camera()
    frame = render_lines_only(camera)

    right = calibrate_auto(frame, camera_side="minus_y")
    wrong = calibrate_auto(frame, camera_side="plus_y")
    assert right is not None and wrong is not None

    def best_error(result) -> float:
        options = [result.h] + ([result.h_rotated] if result.h_rotated is not None else [])
        return min(
            pitch_error(h, camera.H, image_size=(camera.width, camera.height))[0]
            for h in options
        )

    assert best_error(right) < 1.0
    # The mirrored solution is a different, wrong place on the pitch, yet it
    # scores well by its own metric. That is exactly why the side must be told.
    assert best_error(wrong) > 5.0
    assert wrong.score < 6.0


def test_a_prior_resolves_the_rotation_ambiguity():
    """Given a previous homography, the correct twin is chosen.

    This is what makes re-calibration after a camera cut usable: settle the
    ambiguity once, then carry it across cuts.
    """
    camera = a_camera()
    frame = render_lines_only(camera)

    result = calibrate_auto(frame, camera_side="minus_y", prior_h=camera.H)
    assert result is not None
    assert result.resolved_by_prior

    error, _ = pitch_error(result.h, camera.H, image_size=(camera.width, camera.height))
    assert error < 1.0, f"with a prior the fit should be unambiguous, was {error:.2f}m out"


def test_left_goal_side_resolves_the_rotation_ambiguity_with_no_prior():
    """The first calibration a video ever gets has no prior homography.

    `test_a_prior_resolves_the_rotation_ambiguity` covers re-calibration after a
    cut, where a prior exists. Before that, on frame zero, `pipeline.py` cannot
    supply one, and without `left_goal_side` the choice between the two
    rotations is silently arbitrary. This is the failure that put the demo
    clip's tracking 53 metres out despite `is_confident` being True: a
    confidently wrong end-for-end fit with no signal that anything was wrong.
    """
    camera = a_camera()
    frame = render_lines_only(camera)

    left_goal = np.array(pitch.goal_centre("left"))
    true_x = project(camera.H, left_goal[None, :])[0][0]
    true_side = "left" if true_x < camera.width / 2.0 else "right"
    wrong_side = "right" if true_side == "left" else "left"

    right = calibrate_auto(frame, camera_side="minus_y", left_goal_side=true_side)
    wrong = calibrate_auto(frame, camera_side="minus_y", left_goal_side=wrong_side)
    assert right is not None and wrong is not None
    assert right.resolved_by_hint and wrong.resolved_by_hint

    right_error, _ = pitch_error(right.h, camera.H, image_size=(camera.width, camera.height))
    wrong_error, _ = pitch_error(wrong.h, camera.H, image_size=(camera.width, camera.height))
    assert right_error < 1.0, f"correct hint should settle the fit, was {right_error:.2f}m out"
    assert wrong_error > 5.0, "wrong hint should pick the flipped fit, not the true one"


def test_rotating_a_fit_twice_returns_it():
    camera = a_camera()
    assert np.allclose(_rotate180(_rotate180(camera.H)), camera.H, atol=1e-9)


def test_declines_on_a_frame_with_no_pitch():
    """An empty frame must return None rather than an invented homography."""
    blank = np.full((720, 1280, 3), 60, dtype=np.uint8)
    assert calibrate_auto(blank) is None


def test_rejects_an_invalid_camera_side():
    with pytest.raises(ValueError, match="camera_side"):
        calibrate_auto(np.zeros((720, 1280, 3), dtype=np.uint8), camera_side="sideways")
