"""Tests for the professional-tracking comparison.

These build tiny SkillCorner-shaped files rather than reading the real 86MB
match, so the suite still needs no network and no downloads.
"""

from __future__ import annotations

import json

import pytest

from footballvisual.reference import (
    format_comparison,
    load_pipeline_stats,
    load_skillcorner_stats,
)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def frame(n, positioned, detected, ball_detected):
    """One SkillCorner frame with `positioned` players, `detected` of them seen."""
    players = [
        {"x": float(i), "y": 0.0, "player_id": i, "is_detected": i < detected}
        for i in range(positioned)
    ]
    return {
        "frame": n,
        "player_data": players,
        "ball_data": {"x": 0.0, "y": 0.0, "is_detected": ball_detected},
    }


def test_detection_rate_counts_seen_players_not_reported_ones(tmp_path):
    """The whole point of the reference is this distinction.

    A broadcast tracker reports the full squad every frame and extrapolates the
    players it could not see. Reading the reported count as coverage would say a
    commercial system tracks 100% of players from broadcast, which is exactly
    the wrong conclusion to draw.
    """
    rows = [frame(0, 22, 11, True), frame(1, 22, 11, True)]
    stats = load_skillcorner_stats(write_jsonl(tmp_path / "t.jsonl", rows))

    assert stats.players_reported_median == 22
    assert stats.players_detected_median == 11
    assert stats.detection_rate_mean == pytest.approx(0.5)


def test_frames_before_kickoff_are_skipped(tmp_path):
    """Pre-match frames carry a full player list of nulls.

    Counting them would dilute the detection rate with footage that is not of
    the match, which on a real file is thousands of frames of build-up.
    """
    empty = {
        "frame": 0,
        "player_data": [{"x": None, "y": None, "player_id": 1, "is_detected": None}],
        "ball_data": {"x": None, "y": None, "is_detected": None},
    }
    stats = load_skillcorner_stats(write_jsonl(tmp_path / "t.jsonl", [empty, frame(1, 22, 22, True)]))

    assert stats.frames == 1, "only the frame with real positions should count"
    assert stats.detection_rate_mean == pytest.approx(1.0)


def test_a_file_with_no_positioned_frames_is_an_error(tmp_path):
    """Better to fail than to report a detection rate over zero frames."""
    empty = {"frame": 0, "player_data": [], "ball_data": {}}
    with pytest.raises(ValueError, match="no frames"):
        load_skillcorner_stats(write_jsonl(tmp_path / "t.jsonl", [empty]))


def test_ball_rate_tracks_the_detected_flag(tmp_path):
    rows = [frame(i, 22, 22, i % 4 == 0) for i in range(8)]
    stats = load_skillcorner_stats(write_jsonl(tmp_path / "t.jsonl", rows))
    assert stats.ball_detection_rate == pytest.approx(0.25)


def test_pipeline_stats_read_back_from_tracks_json(tmp_path):
    tracks = tmp_path / "tracks.json"
    tracks.write_text(
        json.dumps(
            {
                "meta": {"fps": 25},
                "tracks": [],
                "frames": [
                    {"frame": 0, "timeS": 0.0, "players": [{"id": 1}], "ball": [0, 0]},
                    {"frame": 1, "timeS": 0.04, "players": [{"id": 1}, {"id": 2}], "ball": None},
                ],
            }
        )
    )
    stats = load_pipeline_stats(tracks)
    assert stats.frames == 2
    assert stats.players_tracked_mean == pytest.approx(1.5)
    assert stats.ball_present_rate == pytest.approx(0.5)


def test_comparison_says_plainly_that_it_is_not_a_score(tmp_path):
    """The report must not read as a head to head, because it is not one.

    The reference is a different match on footage this repository does not have.
    A reader skimming two columns of numbers will assume otherwise unless the
    text says so, and a misread here would turn context into a false claim.
    """
    rows = [frame(0, 22, 11, True)]
    reference = load_skillcorner_stats(write_jsonl(tmp_path / "ref.jsonl", rows))
    text = format_comparison(reference, None)
    assert "not as a score" in text
    assert "different match" in text


def test_aggregate_weights_matches_by_frames_not_equally(tmp_path):
    """A longer match carries more evidence, so it should count for more.

    Averaging match means equally would let a 40 frame fixture move the baseline
    as much as a 40,000 frame one. That is a choice made for arithmetic
    convenience rather than for a reason, and the baseline is the whole point of
    this module.
    """
    from footballvisual.reference import aggregate_stats

    short = load_skillcorner_stats(
        write_jsonl(tmp_path / "a.jsonl", [frame(0, 10, 10, True)])
    )
    long = load_skillcorner_stats(
        write_jsonl(tmp_path / "b.jsonl", [frame(i, 10, 0, False) for i in range(9)])
    )

    combined = aggregate_stats([short, long])

    assert combined.frames == 10
    # Frame weighted: one frame at 100% and nine at 0% is 10%, not the 50% an
    # unweighted mean of the two matches would give.
    assert combined.detection_rate_mean == pytest.approx(0.1)
    assert combined.ball_detection_rate == pytest.approx(0.1)


def test_aggregate_reports_how_many_matches_it_read(tmp_path):
    """The source line has to say the baseline is not one fixture."""
    from footballvisual.reference import aggregate_stats

    runs = [
        load_skillcorner_stats(
            write_jsonl(tmp_path / f"{i}.jsonl", [frame(0, 22, 11, True)])
        )
        for i in range(3)
    ]
    combined = aggregate_stats(runs)
    assert "3 matches" in combined.source


def test_aggregate_refuses_an_empty_list():
    from footballvisual.reference import aggregate_stats

    with pytest.raises(ValueError, match="no runs"):
        aggregate_stats([])
