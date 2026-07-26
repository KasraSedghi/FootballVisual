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


def test_keeper_and_referee_are_told_apart_by_where_they_go():
    """Colour cannot separate a keeper from a referee, position can.

    Both wear kit unlike either team, so both land in the same colour outlier
    bucket. What distinguishes them is that a keeper lives in front of one goal
    while a referee follows play across the whole pitch.
    """
    from footballvisual.teams import classify_officials

    keeper = [(48.0 + 0.5 * np.sin(i / 3.0), 2.0 * np.sin(i / 5.0)) for i in range(60)]
    # A referee tracking play from one third to the other.
    referee = [(-30.0 + i, 8.0 * np.sin(i / 7.0)) for i in range(60)]

    result = classify_officials(
        candidate_ids={1, 2},
        trajectories={1: keeper, 2: referee},
    )
    assert result.role_of(1) == "keeper"
    assert result.role_of(2) == "referee"
    # The evidence should be inspectable, not just the verdict.
    assert result.evidence[1]["distanceToGoalM"] < 18.0
    assert result.evidence[2]["xRange"] > 34.0


def test_a_stationary_outlier_is_not_called_a_referee():
    """Referee must be positive evidence, not "failed the keeper test".

    A track sitting mid-pitch with a few metres of range is not an official; it
    is more likely a player whose colour was unreliable. Labelling it referee
    made the label meaningless, which is what this pins.
    """
    from footballvisual.teams import classify_officials

    stationary = [(10.0 + 0.3 * np.sin(i / 4.0), -23.0) for i in range(40)]
    result = classify_officials(candidate_ids={5}, trajectories={5: stationary})
    assert result.role_of(5) == "other"


def test_a_track_with_too_little_history_is_left_unknown():
    """Better to decline than to mislabel a defender as a keeper.

    A wrong keeper label moves the offside line, so an uncertain call here is
    more damaging than a missing one.
    """
    from footballvisual.teams import classify_officials

    result = classify_officials(candidate_ids={7}, trajectories={7: [(50.0, 0.0)] * 3})
    assert result.role_of(7) == "unknown"


def test_a_keeper_at_the_other_end_is_still_a_keeper():
    """The rule must be symmetric in x, not hardcoded to one goal."""
    from footballvisual.teams import classify_officials

    keeper = [(-49.0 + 0.4 * np.sin(i / 3.0), 1.5 * np.sin(i / 4.0)) for i in range(40)]
    result = classify_officials(candidate_ids={3}, trajectories={3: keeper})
    assert result.role_of(3) == "keeper"


def test_ball_jump_gate_scales_with_time_unseen():
    """A gate that ignores elapsed time is wrong in both directions.

    Fixed too loose, it accepts any white blob on the pitch as the ball on the
    very next frame. Fixed too tight, it refuses a legitimate re-acquisition
    after the ball has been occluded for half a second, during which it really
    can have travelled a long way.
    """
    ball = BallTracker(max_jump_px=50.0, max_coast=12)
    ball.update(Detection((100.0, 100.0, 104.0, 104.0), 0.9, kind="ball"))
    ball.update(Detection((110.0, 100.0, 114.0, 104.0), 0.9, kind="ball"))

    # Immediately: a 300px jump is not this ball.
    far = ball.update(Detection((410.0, 100.0, 414.0, 104.0), 0.9, kind="ball"))
    assert far is not None and far[0] < 200.0, "a huge one-frame jump must be rejected"

    # After several missed frames the same jump is plausible again.
    patient = BallTracker(max_jump_px=50.0, max_coast=12)
    patient.update(Detection((100.0, 100.0, 104.0, 104.0), 0.9, kind="ball"))
    for _ in range(6):
        patient.update(None)
    reacquired = patient.update(Detection((410.0, 100.0, 414.0, 104.0), 0.9, kind="ball"))
    assert reacquired is not None and reacquired[0] > 350.0, (
        "after a gap the ball should be allowed to have moved further"
    )
