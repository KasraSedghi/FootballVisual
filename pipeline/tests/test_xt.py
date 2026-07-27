"""Tests for the Expected Threat model.

These use synthetic tallies rather than the network. The point is to pin the
solver's behaviour and the coordinate handling, both of which are things a
change could silently break; whether the real grid looks sensible is checked by
`train_xt.py` printing it and by `value.test.ts` asserting its shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballvisual.xt import (
    GRID_X,
    GRID_Y,
    Counts,
    _to_cell,
    count_match,
    solve,
    symmetrise,
)


def _uniform_counts(shot_rate: float = 0.05) -> Counts:
    """Tallies where every cell moves one cell toward the goal, or shoots."""
    counts = Counts()
    for y in range(GRID_Y):
        for x in range(GRID_X):
            counts.moves[y, x] = 100 * (1 - shot_rate)
            counts.shots[y, x] = 100 * shot_rate
            # Conversion rises strictly toward the goal, as it really does.
            # Strictly, not in steps: a flat pair of columns here would make the
            # monotonicity assertion below fail on the fixture rather than on
            # anything the solver did.
            counts.goals[y, x] = counts.shots[y, x] * (0.01 + 0.03 * x)
            target = min(GRID_X - 1, x + 1)
            counts.transitions[y, x, y, target] = counts.moves[y, x]
    return counts


class TestSolve:
    def test_converges_to_a_fixed_point(self):
        xt, deltas = solve(_uniform_counts())
        assert deltas[-1] < 1e-7, "solver stopped while still moving"
        assert len(deltas) < 400, "hit the iteration cap instead of converging"

    def test_value_increases_toward_the_goal(self):
        xt, _ = solve(_uniform_counts())
        row = xt[GRID_Y // 2]
        assert np.all(np.diff(row) > 0), f"not monotonic toward goal: {row}"

    def test_a_dozen_iterations_is_not_enough(self):
        """The reason `solve` runs to a tolerance rather than a fixed count.

        Convergence is geometric at the move share, which is about 0.95 here and
        nearer 0.99 in real data, so an early stop leaves the far end of the
        pitch still climbing. This pins the mistake rather than the fix: if
        someone reinstates a small fixed iteration count, the value in the
        build-up third silently drops.
        """
        counts = _uniform_counts()
        early, _ = solve(counts, max_iterations=12)
        settled, _ = solve(counts)
        assert early[GRID_Y // 2, 0] < settled[GRID_Y // 2, 0] * 0.9

    def test_losing_the_ball_absorbs_value_instead_of_recirculating_it(self):
        """The bug that produced a flat grid, pinned.

        Transitions must be divided by moves *attempted*, so each row sums to
        that cell's completion rate and the missing mass is a turnover that
        absorbs at zero. Normalising rows to sum to 1 instead says every move
        arrives somewhere, which removes the only absorbing state; with a move
        share near 0.99 the value then diffuses until every cell holds the same
        number.

        Here half of all moves are lost, so each step toward the goal should
        cost half the value. Under the broken normalisation the whole pitch
        flattens to roughly the shooting cell's value.
        """
        counts = Counts()
        for x in range(GRID_X):
            counts.moves[0, x] = 100
            # Only half the moves arrive, and the rest are turnovers.
            counts.transitions[0, x, 0, min(GRID_X - 1, x + 1)] = 50
        # One shooting cell at the goal end, converting often.
        counts.shots[0, GRID_X - 1] = 100
        counts.goals[0, GRID_X - 1] = 30

        xt, _ = solve(counts)
        row = xt[0]

        assert row[GRID_X - 1] > row[0] * 5, (
            f"value did not decay away from the goal, so turnovers are being "
            f"recirculated rather than absorbed: {row}"
        )

    def test_a_cell_nobody_plays_from_is_worth_nothing_rather_than_nan(self):
        counts = _uniform_counts()
        counts.moves[0, 0] = 0
        counts.shots[0, 0] = 0
        counts.goals[0, 0] = 0
        counts.transitions[0, 0] = 0

        xt, _ = solve(counts)
        assert np.all(np.isfinite(xt))
        assert xt[0, 0] == pytest.approx(0.0)


class TestSymmetrise:
    def test_folding_makes_the_grid_mirror_symmetric(self):
        counts = Counts()
        counts.moves[0, 3] = 10
        counts.shots[0, 3] = 5
        counts.goals[0, 3] = 1

        folded = symmetrise(counts)
        assert folded.moves[0, 3] == folded.moves[GRID_Y - 1, 3]
        assert folded.goals[0, 3] == folded.goals[GRID_Y - 1, 3]

    def test_folding_conserves_the_totals(self):
        counts = _uniform_counts()
        folded = symmetrise(counts)
        # Each event is counted once in its own cell and once in the mirror, so
        # the total doubles. Anything else means events were lost or duplicated
        # unevenly, which would bias the grid rather than just rescale it.
        assert folded.shots.sum() == pytest.approx(2 * counts.shots.sum())
        assert folded.transitions.sum() == pytest.approx(2 * counts.transitions.sum())

    def test_folding_transitions_mirrors_both_ends(self):
        counts = Counts()
        counts.transitions[1, 2, 6, 9] = 4
        folded = symmetrise(counts)
        # Reflecting a pass reflects where it started *and* where it finished.
        assert folded.transitions[GRID_Y - 2, 2, GRID_Y - 7, 9] == 4


class TestToCell:
    def test_maps_the_corners_to_the_corner_cells(self):
        assert _to_cell(0.0, 0.0) == (0, 0)
        assert _to_cell(119.9, 79.9) == (GRID_Y - 1, GRID_X - 1)

    def test_clamps_a_location_recorded_off_the_pitch(self):
        # StatsBomb occasionally records a location fractionally outside. A ball
        # on the byline is a real position, not a corrupt one.
        assert _to_cell(120.0, 80.0) == (GRID_Y - 1, GRID_X - 1)
        assert _to_cell(-3.0, -1.0) == (0, 0)


class TestCountMatch:
    def test_a_failed_pass_counts_as_an_action_but_moves_no_value(self):
        events = [
            {
                "type": {"name": "Pass"},
                "location": [60.0, 40.0],
                "pass": {"end_location": [90.0, 40.0], "outcome": {"name": "Incomplete"}},
            }
        ]
        counts = count_match(events)
        assert counts.moves.sum() == 1
        assert counts.transitions.sum() == 0, "an incomplete pass moved value"

    def test_set_pieces_are_excluded(self):
        events = [
            {
                "type": {"name": "Pass"},
                "location": [120.0, 80.0],
                "pass": {"end_location": [110.0, 40.0], "type": {"name": "Corner"}},
            },
            {
                "type": {"name": "Shot"},
                "location": [110.0, 40.0],
                "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}},
            },
        ]
        counts = count_match(events)
        assert counts.moves.sum() == 0
        assert counts.shots.sum() == 0, "a penalty was counted as an open-play shot"

    def test_a_completed_carry_records_a_transition(self):
        events = [
            {
                "type": {"name": "Carry"},
                "location": [60.0, 40.0],
                "carry": {"end_location": [70.0, 40.0]},
            }
        ]
        counts = count_match(events)
        assert counts.transitions.sum() == 1
