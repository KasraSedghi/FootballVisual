"""Tests for shot-change detection.

The failure this guards against is quiet. Across a cut, optical flow still
returns matches and RANSAC still fits a homography to them, so the pipeline
carries on composing transforms and projecting players to confidently wrong
places. Nothing in the output looks broken.
"""

from __future__ import annotations

import numpy as np

from footballvisual.camera import BroadcastCamera, broadcast_pan
from footballvisual.cuts import CutDetector

from .test_autocalibrate import render_lines_only


def a_camera(target_x: float = 18.0) -> BroadcastCamera:
    return BroadcastCamera(
        width=640, height=360, eye=(target_x * 0.2, -70.0, 17.0),
        target=(target_x, 1.0, 0.0), vfov_deg=24.0,
    )


def tinted(frame: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """Recolour a frame, standing in for a different shot.

    A cut in football usually goes to a different camera, a replay, or a crowd
    shot, all of which change the colour distribution. Shifting the palette
    reproduces that without needing a second rendered scene.
    """
    out = frame.astype(np.int16)
    for c in range(3):
        out[:, :, c] += shift[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def test_no_cut_is_reported_during_a_smooth_pan():
    """A pan moves every pixel and must not be mistaken for a cut.

    This is why the detector compares histograms rather than pixels: a
    pixel-difference test fires on exactly this sequence.
    """
    cameras = broadcast_pan(640, 360, 40, start_x=5.0, end_x=45.0)
    detector = CutDetector()

    cuts = [
        detector.update(render_lines_only(camera), i)
        for i, camera in enumerate(cameras)
    ]
    assert not any(cuts), "a continuous pan should contain no shot changes"


def test_a_hard_cut_is_detected_on_the_frame_it_happens():
    frames = [render_lines_only(a_camera(10.0)) for _ in range(6)]
    frames += [tinted(render_lines_only(a_camera(40.0)), (90, -60, 70)) for _ in range(6)]

    detector = CutDetector()
    fired = [i for i, f in enumerate(frames) if detector.update(f, i)]

    assert fired == [6], f"expected a single cut at frame 6, got {fired}"
    assert detector.events[0].frame == 6


def test_repeat_firings_are_suppressed_while_the_shot_settles():
    """One cut should be reported once, not once per settling frame."""
    a = render_lines_only(a_camera(10.0))
    b = tinted(render_lines_only(a_camera(40.0)), (90, -60, 70))

    # Alternate rapidly: without suppression every frame is a "cut".
    frames = [a, a, b, a, b, a, b, a]
    detector = CutDetector(min_gap=8)
    fired = [i for i, f in enumerate(frames) if detector.update(f, i)]

    assert len(fired) == 1, f"min_gap should collapse the burst, got {fired}"


def test_first_frame_is_never_a_cut():
    detector = CutDetector()
    assert detector.update(render_lines_only(a_camera()), 0) is False


def test_reset_clears_history():
    detector = CutDetector()
    detector.update(render_lines_only(a_camera(10.0)), 0)
    detector.reset()
    # After a reset the next frame is a first frame again, so no cut.
    assert detector.update(tinted(render_lines_only(a_camera(40.0)), (90, -60, 70)), 1) is False
