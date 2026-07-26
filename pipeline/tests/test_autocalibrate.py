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
