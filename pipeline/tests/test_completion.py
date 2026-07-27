"""Tests for the pass-completion calibration.

The load-bearing test here is `TestMotionConstants`. The margin this module
computes is only a calibration of the *engine's* margin if the two use the same
physics, and they live in different languages with no shared module. A drift
would not fail anything, it would silently fit a curve for one model and apply
it in another.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from footballvisual import completion
from footballvisual.completion import (
    Sample,
    fit_logistic,
    interception_margin,
    samples_from_match,
    to_metres,
)
from footballvisual.pitch import HALF_LENGTH, HALF_WIDTH

TYPES_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "tactics" / "types.ts"


class TestMotionConstants:
    """Pin the Python copies against the TypeScript source of truth."""

    @staticmethod
    def _read_default_motion() -> dict[str, float]:
        source = TYPES_TS.read_text()
        block = re.search(
            r"export const DEFAULT_MOTION: MotionModel = \{(.*?)\};", source, re.S
        )
        assert block, "DEFAULT_MOTION not found in types.ts"
        return {
            key: float(value)
            for key, value in re.findall(r"(\w+):\s*([0-9.]+)", block.group(1))
        }

    def test_python_matches_typescript(self):
        ts = self._read_default_motion()
        assert completion.PASS_SPEED == ts["passSpeed"]
        assert completion.PLAYER_MAX_SPEED == ts["playerMaxSpeed"]
        assert completion.REACTION_TIME_S == ts["reactionTimeS"]
        assert completion.INTERCEPT_RADIUS == ts["interceptRadius"]


class TestToMetres:
    def test_the_corners_land_on_the_corners(self):
        assert to_metres(0.0, 0.0) == pytest.approx((-HALF_LENGTH, -HALF_WIDTH))
        assert to_metres(120.0, 80.0) == pytest.approx((HALF_LENGTH, HALF_WIDTH))

    def test_the_centre_spot_is_the_origin(self):
        assert to_metres(60.0, 40.0) == pytest.approx((0.0, 0.0))


class TestInterceptionMargin:
    def test_no_defenders_means_the_pass_is_free(self):
        margin = interception_margin((0, 0), (20, 0), np.zeros((0, 2)))
        assert math.isinf(margin)

    def test_a_defender_standing_on_the_lane_cuts_it_out(self):
        margin = interception_margin((0, 0), (30, 0), np.array([[15.0, 0.0]]))
        assert margin < 0

    def test_a_distant_defender_does_not(self):
        margin = interception_margin((0, 0), (10, 0), np.array([[5.0, 30.0]]))
        assert margin > 0

    def test_the_same_defender_is_more_dangerous_to_a_longer_pass(self):
        """Why the race beats a clearance measurement.

        The defender is the same distance off the line in both cases. The long
        pass gives them time to get there and the short one does not, which a
        clearance test cannot express at all.
        """
        near = interception_margin((0, 0), (6, 0), np.array([[3.0, 4.0]]))
        far = interception_margin((0, 0), (60, 0), np.array([[30.0, 4.0]]))
        assert near > far

    def test_the_worst_defender_decides(self):
        harmless = np.array([[0.0, 30.0]])
        both = np.array([[0.0, 30.0], [15.0, 0.0]])
        assert interception_margin((0, 0), (30, 0), both) < interception_margin(
            (0, 0), (30, 0), harmless
        )


class TestSamplesFromMatch:
    @staticmethod
    def _frame(uuid: str, opponents: int = 10):
        return {
            "event_uuid": uuid,
            "freeze_frame": [
                {"teammate": False, "keeper": False, "location": [50.0 + i, 30.0]}
                for i in range(opponents)
            ],
        }

    @staticmethod
    def _pass(uuid: str, outcome=None, kind=None):
        detail = {"end_location": [80.0, 40.0]}
        if outcome:
            detail["outcome"] = {"name": outcome}
        if kind:
            detail["type"] = {"name": kind}
        return {
            "id": uuid,
            "type": {"name": "Pass"},
            "location": [40.0, 40.0],
            "pass": detail,
        }

    def test_reads_the_outcome_off_the_event(self):
        got = samples_from_match(
            [self._pass("a"), self._pass("b", outcome="Incomplete")],
            [self._frame("a"), self._frame("b")],
        )
        assert [s.completed for s in got] == [True, False]

    def test_drops_a_pass_with_too_few_visible_opponents(self):
        """A freeze frame only holds what was on camera.

        A pass whose nearest defender was out of shot looks safer than it was,
        so it would drag the fitted curve toward optimism.
        """
        got = samples_from_match([self._pass("a")], [self._frame("a", opponents=3)])
        assert got == []

    def test_drops_set_pieces(self):
        got = samples_from_match([self._pass("a", kind="Corner")], [self._frame("a")])
        assert got == []

    def test_drops_a_pass_with_no_freeze_frame(self):
        assert samples_from_match([self._pass("a")], []) == []


class TestFitLogistic:
    @staticmethod
    def _synthetic(n: int = 4000) -> list[Sample]:
        """Passes generated from a known logistic, to check the fit recovers it."""
        rng = np.random.default_rng(7)
        margins = rng.uniform(-1.5, 2.0, n)
        probability = 1 / (1 + np.exp(-(0.5 + 2.5 * margins)))
        outcomes = rng.random(n) < probability
        return [
            Sample(margin_s=float(m), completed=bool(o), distance_m=20.0)
            for m, o in zip(margins, outcomes)
        ]

    def test_recovers_the_coefficients_it_was_generated_from(self):
        beta, _ = fit_logistic(self._synthetic(), use_distance=False)
        assert beta[0] == pytest.approx(0.5, abs=0.15)
        assert beta[1] == pytest.approx(2.5, abs=0.25)

    def test_completion_rises_with_margin(self):
        beta, _ = fit_logistic(self._synthetic(), use_distance=False)
        assert beta[1] > 0

    def test_reports_positive_skill_when_the_margin_predicts(self):
        _, diagnostics = fit_logistic(self._synthetic(), use_distance=False)
        assert diagnostics["brier_skill"] > 0.1

    def test_reports_no_skill_when_the_margin_is_noise(self):
        """The negative case, without which the skill number means nothing."""
        rng = np.random.default_rng(3)
        samples = [
            Sample(
                margin_s=float(rng.uniform(-1.5, 2.0)),
                completed=bool(rng.random() < 0.8),
                distance_m=20.0,
            )
            for _ in range(4000)
        ]
        _, diagnostics = fit_logistic(samples, use_distance=False)
        assert diagnostics["brier_skill"] < 0.01
