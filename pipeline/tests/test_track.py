"""Tests for the tracker.

The regression test at the bottom pins a real bug found during development, and
is the most valuable test in this file. See its docstring.
"""

from __future__ import annotations

import numpy as np

from footballvisual.detect import Detection, _nms
from footballvisual.track import BallTracker, ByteTracker, TrackState, iou_matrix
from footballvisual.teams import assign_teams


def box(x: float, y: float, w: float = 20.0, h: float = 50.0) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def test_nms_removes_duplicate_boxes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    scores = np.array([0.9, 0.8, 0.7])
    keep = _nms(boxes, scores, iou_threshold=0.45)
    assert sorted(keep.tolist()) == [0, 2]


def test_iou_matrix_is_zero_for_disjoint_boxes():
    tracker = ByteTracker(min_hits=2)
    # Feed enough frames for the track to be confirmed; `update` only returns
    # confirmed tracks, so a single frame yields an empty list.
    for frame in range(4):
        tracks = tracker.update([Detection(box(0, 0), 0.9)], frame_index=frame)
    assert tracks, "expected a confirmed track to compare against"
    assert iou_matrix(tracks, [Detection(box(500, 500), 0.9)]).max() == 0.0


def test_iou_matrix_is_empty_when_either_side_is_empty():
    assert iou_matrix([], [Detection(box(0, 0), 0.9)]).size == 0


def test_track_is_confirmed_after_min_hits_and_keeps_its_id():
    tracker = ByteTracker(min_hits=3)
    ids = []
    for frame in range(6):
        confirmed = tracker.update([Detection(box(100 + frame * 3, 200), 0.9)], frame_index=frame)
        if confirmed:
            ids.append(confirmed[0].track_id)
    assert ids, "a steadily detected object should become a confirmed track"
    assert len(set(ids)) == 1, "a single object must not change identity"


def test_two_separated_objects_get_distinct_ids():
    tracker = ByteTracker(min_hits=2)
    for frame in range(6):
        tracks = tracker.update(
            [Detection(box(100, 200), 0.9), Detection(box(600, 210), 0.9)],
            frame_index=frame,
        )
    assert len({t.track_id for t in tracks}) == 2


def test_lost_track_is_recovered_after_a_gap():
    """Regression: lost tracks must stay eligible for matching.

    `Track.is_active` originally excluded `LOST`, which meant a track that
    missed a single detection was dropped from the association candidates
    permanently. It could then never be recovered, and because the removal check
    also only ran over active tracks it never aged out either, so lost tracks
    accumulated forever as immortal non-candidates. On the project's own clip
    this capped the tracker at roughly 14 simultaneous tracks against 20 visible
    players, and fixing it moved detection coverage from 43% to 76%.

    Recovering through an occlusion is the entire reason to run ByteTrack rather
    than plain IoU matching, so it is worth a dedicated test.
    """
    tracker = ByteTracker(min_hits=2, max_age=30)

    for frame in range(5):
        tracker.update([Detection(box(100 + frame * 4, 200), 0.9)], frame_index=frame)
    original = [t.track_id for t in tracker.tracks if t.state is TrackState.CONFIRMED]
    assert original, "expected a confirmed track before the occlusion"

    # Occluded: no detections at all for several frames.
    for frame in range(5, 11):
        tracker.update([], frame_index=frame)

    assert any(t.state is TrackState.LOST for t in tracker.tracks), "track should be lost"

    # Reappears roughly where the constant-velocity model predicts.
    recovered = []
    for frame in range(11, 15):
        recovered = tracker.update(
            [Detection(box(100 + frame * 4, 200), 0.9)], frame_index=frame
        )

    assert recovered, "track should be picked back up after the gap"
    assert recovered[0].track_id == original[0], "recovery must reuse the original id"


def test_lost_tracks_are_eventually_removed():
    """The other half of the same bug: lost tracks must not accumulate forever."""
    tracker = ByteTracker(min_hits=2, max_age=5)
    for frame in range(4):
        tracker.update([Detection(box(100, 200), 0.9)], frame_index=frame)

    for frame in range(4, 40):
        tracker.update([], frame_index=frame)

    assert not tracker.tracks, "a track lost for far longer than max_age should be dropped"


def test_ball_tracker_coasts_through_gaps_then_gives_up():
    ball = BallTracker(max_coast=5)
    ball.update(Detection((100.0, 100.0, 104.0, 104.0), 0.9, kind="ball"))
    moved = ball.update(Detection((110.0, 100.0, 114.0, 104.0), 0.9, kind="ball"))
    assert moved is not None

    coasted = ball.update(None)
    assert coasted is not None and coasted[0] > moved[0], "should extrapolate forward"

    for _ in range(10):
        result = ball.update(None)
    assert result is None, "coasting must not continue indefinitely"


def test_ball_tracker_ignores_an_implausible_jump():
    ball = BallTracker(max_jump_px=100.0)
    ball.update(Detection((100.0, 100.0, 104.0, 104.0), 0.9, kind="ball"))
    ball.update(Detection((105.0, 100.0, 109.0, 104.0), 0.9, kind="ball"))
    # A white blob on the far side of the frame is not this ball.
    result = ball.update(Detection((900.0, 400.0, 904.0, 404.0), 0.9, kind="ball"))
    assert result is not None and result[0] < 200.0


def test_team_clustering_separates_two_colour_groups():
    rng = np.random.default_rng(0)
    # Lab colours: two well separated clusters plus one outlier for a keeper.
    observations = {
        i: [np.array([50.0, 60.0, 30.0]) + rng.normal(0, 2, 3) for _ in range(6)]
        for i in range(5)
    }
    observations.update(
        {
            i: [np.array([55.0, -40.0, -50.0]) + rng.normal(0, 2, 3) for _ in range(6)]
            for i in range(5, 10)
        }
    )
    observations[99] = [np.array([90.0, 5.0, 95.0]) for _ in range(6)]

    result = assign_teams(observations)

    first = {result.team_of(i) for i in range(5)}
    second = {result.team_of(i) for i in range(5, 10)}
    assert len(first) == 1 and len(second) == 1
    assert first != second, "the two colour groups must land in different teams"
    assert result.team_of(99) == "other", "a colour unlike either team should not be forced"


def test_team_clustering_declines_when_there_is_nothing_to_cluster():
    assert assign_teams({}).labels == {}
    assert assign_teams({1: [np.array([50.0, 0.0, 0.0])] * 5}).labels == {}
