"""Tests for the homography estimator.

These are the tests that matter most in the project. Everything downstream, the
pitch coordinates, the tactical distances, the passing lane margins, is only as
correct as the homography, and a subtly wrong one produces output that looks
entirely plausible and is silently metres off.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballvisual import pitch
from footballvisual.camera import BroadcastCamera
from footballvisual.homography import (
    HomographyError,
    calibrate_from_landmarks,
    estimate_homography,
    image_point_for_player,
    invert,
    project,
    ransac_homography,
    reprojection_error,
)


def a_camera() -> BroadcastCamera:
    return BroadcastCamera(
        width=1280, height=720, eye=(4.0, -70.0, 17.0), target=(22.0, 1.0, 0.0)
    )


def test_exact_fit_from_four_points():
    src = np.array([[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]])
    dst = np.array([[120.0, 700.0], [1160.0, 700.0], [980.0, 300.0], [300.0, 300.0]])
    h = estimate_homography(src, dst)
    assert reprojection_error(h, src, dst).max() < 1e-6


def test_round_trip_through_inverse_is_identity():
    h = a_camera().H
    points = np.array([[0.0, 0.0], [30.0, -12.0], [-45.0, 25.0], [52.0, 33.0]])
    back = project(invert(h), project(h, points))
    assert np.allclose(back, points, atol=1e-6)


def test_collinear_points_are_rejected():
    """Four points on a line satisfy the DLT constraints but fix nothing.

    Without this check the solver returns a matrix that reprojects the sample
    perfectly and every other point nowhere near, which is far harder to
    diagnose downstream than an explicit failure here.
    """
    src = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    dst = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    with pytest.raises(HomographyError, match="collinear"):
        estimate_homography(src, dst)


def test_too_few_correspondences_are_rejected():
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 5.0]])
    with pytest.raises(HomographyError):
        estimate_homography(src, src)


def test_ransac_rejects_injected_outliers():
    """The whole point of RANSAC here: survive mismatched landmarks.

    An automatic detector confusing one line intersection for another is the
    normal failure mode, so the fit has to tolerate a few gross errors rather
    than being dragged by them.
    """
    h_true = a_camera().H
    rng = np.random.default_rng(4)

    names = sorted(pitch.LANDMARKS)[:22]
    src = np.array([pitch.landmark(n).xy for n in names])
    dst = project(h_true, src) + rng.normal(0.0, 1.0, (len(src), 2))

    bad = [2, 8, 15]
    dst[bad] += np.array([400.0, -350.0])

    result = ransac_homography(src, dst, threshold=6.0)

    for index in bad:
        assert not result.inliers[index], f"outlier {index} was accepted as an inlier"
    assert result.num_inliers >= len(src) - len(bad) - 2
    assert result.mean_inlier_error < 3.0


def test_calibration_recovers_the_true_camera_within_a_metre():
    """End to end: noisy landmark clicks should still localise players well.

    Two pixels of click error is a realistic human, and the acceptance bound is
    stated in metres on the pitch because that is the unit the tactical layer
    actually cares about.
    """
    camera = a_camera()
    h_true = camera.H
    rng = np.random.default_rng(11)

    correspondences = {}
    for name in sorted(pitch.LANDMARKS):
        xy = project(h_true, np.array([pitch.landmark(name).xy]))[0]
        if 12 <= xy[0] < camera.width - 12 and 12 <= xy[1] < camera.height - 12:
            correspondences[name] = tuple(xy + rng.normal(0.0, 2.0, 2))

    assert len(correspondences) >= 4, "camera should see at least four landmarks"

    calibration = calibrate_from_landmarks(correspondences)
    assert calibration.mean_error_px < 6.0

    # Score where the players actually are, not across the whole pitch: the
    # parts of it this camera cannot see are unconstrained by any evidence.
    sample = np.array([[x, y] for x in (0.0, 15.0, 30.0, 40.0) for y in (-20.0, 0.0, 20.0)])
    recovered = calibration.image_to_pitch(project(h_true, sample))
    error = np.linalg.norm(recovered - sample, axis=1)
    assert error.mean() < 1.0, f"mean pitch error {error.mean():.2f}m is too large"


def test_player_ground_point_is_the_feet_not_the_centre():
    """The homography maps the ground plane, so the box centre is the wrong point.

    A player's centre floats about a metre above the grass, and on a shallow
    broadcast angle projecting it throws the position several metres up the
    pitch. This pins the convention that avoids that.
    """
    bbox = (100.0, 200.0, 140.0, 320.0)
    assert image_point_for_player(bbox) == (120.0, 320.0)


def test_camera_homography_matches_full_projection_on_the_ground_plane():
    """`H` must agree with the full 3x4 projection for any z = 0 point.

    If these ever disagree, the synthetic clip's ground truth would describe a
    different camera than the one that drew the frame, and every accuracy number
    the project reports would be meaningless.
    """
    camera = a_camera()
    ground = np.array([[0.0, 0.0], [35.0, -18.0], [-20.0, 28.0]])

    via_homography = project(camera.H, ground)
    via_projection, depth = camera.project_world(
        np.column_stack([ground, np.zeros(len(ground))])
    )

    assert np.all(depth > 0)
    assert np.allclose(via_homography, via_projection, atol=1e-6)


def test_players_further_away_render_smaller():
    """Perspective sanity: the far touchline must not look the same as the near one."""
    camera = a_camera()
    near = camera.player_pixel_height(20.0, -30.0, 1.8)
    far = camera.player_pixel_height(20.0, 30.0, 1.8)
    assert near > far > 0
