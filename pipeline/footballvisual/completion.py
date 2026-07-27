"""Calibrating pass completion against real outcomes.

Valuing an action needs two things: what it gains if it works, and how often it
works. Expected Threat supplies the first. This module supplies the second, and
it measures it rather than assuming it.

The engine already computes a safety margin for every lane, the seconds by which
the ball beats the best placed defender to the most dangerous point on its path.
That number is on a physically meaningful scale but it is not a probability, and
the tempting move is to run it through a logistic with hand-chosen constants.
That would be an invented number wearing the costume of a measured one, in a
project whose entire premise is that tactical claims are checkable.

So it is fitted instead. StatsBomb's 360 data records a freeze frame of every
visible player at the moment of each event, which is enough to run the identical
interception race the engine runs, on a real pass whose outcome is known. Tens
of thousands of those give an empirical curve of completion rate against margin,
and the logistic is fitted to that.

Two honest limitations, both recorded in the fitted model:

* Freeze frames carry no velocities, so defenders are treated as stationary
  here while the engine credits them with momentum. Momentum shifts a margin by
  roughly a defender's speed times the reaction window, well under a metre in
  most cases, but it means the fit is calibrated on a slightly optimistic margin.
* A freeze frame only contains players inside the broadcast frame. A pass whose
  nearest defender was off camera looks safer than it was. Passes with fewer
  than eight visible opponents are dropped for that reason.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass

import numpy as np

from .xt import OPEN_DATA, SB_LENGTH, SB_WIDTH, _fetch
from .pitch import HALF_LENGTH, HALF_WIDTH, PITCH_LENGTH, PITCH_WIDTH

# Mirrors DEFAULT_MOTION in web/src/lib/tactics/types.ts. Duplicated for the
# same reason pitch.ts duplicates pitch.py: the two runtimes cannot share a
# module. A drift here silently miscalibrates, so `test_completion.py` pins
# these against the TypeScript source.
PASS_SPEED = 16.0
PLAYER_MAX_SPEED = 7.5
REACTION_TIME_S = 0.28
INTERCEPT_RADIUS = 0.9
PATH_SAMPLES = 48

# Below this many visible opponents the freeze frame is too incomplete to trust:
# the margin is computed against whoever happened to be on camera.
MIN_VISIBLE_OPPONENTS = 8


def to_metres(x: float, y: float) -> tuple[float, float]:
    """StatsBomb's 120x80 grid to this project's centred metres.

    A linear rescale, which is the accepted convention for open data. It is not
    exact, because StatsBomb's grid is nominal rather than a survey of each
    ground, but the alternative is having no comparable coordinates at all.
    """
    return (
        x / SB_LENGTH * PITCH_LENGTH - HALF_LENGTH,
        y / SB_WIDTH * PITCH_WIDTH - HALF_WIDTH,
    )


def interception_margin(
    start: tuple[float, float],
    end: tuple[float, float],
    defenders: np.ndarray,
) -> float:
    """The engine's lane margin, in Python, for a whole set of defenders at once.

    Identical in form to `interceptionMargin` in lanes.ts: sample the path, ask
    when the ball arrives and when a defender could, and take the worst case.
    Vectorised over defenders because this runs tens of thousands of times.
    """
    if len(defenders) == 0:
        return math.inf

    from_x, from_y = start
    to_x, to_y = end
    total = math.hypot(to_x - from_x, to_y - from_y)
    if total < 1e-6:
        return math.inf

    steps = np.linspace(0.0, total, PATH_SAMPLES + 1)
    dir_x = (to_x - from_x) / total
    dir_y = (to_y - from_y) / total

    path_x = from_x + dir_x * steps
    path_y = from_y + dir_y * steps

    t_ball = steps / PASS_SPEED

    # (defenders, samples)
    dx = defenders[:, 0:1] - path_x[None, :]
    dy = defenders[:, 1:2] - path_y[None, :]
    need = np.maximum(0.0, np.hypot(dx, dy) - INTERCEPT_RADIUS)
    t_def = REACTION_TIME_S + need / PLAYER_MAX_SPEED

    return float((t_def - t_ball[None, :]).min())


@dataclass
class Sample:
    margin_s: float
    completed: bool
    distance_m: float


def samples_from_match(events: list[dict], frames: list[dict]) -> list[Sample]:
    """Every open-play pass in one match that has a usable freeze frame."""
    by_uuid = {f["event_uuid"]: f for f in frames}
    out: list[Sample] = []

    for event in events:
        if event.get("type", {}).get("name") != "Pass":
            continue
        detail = event.get("pass", {})
        # Set pieces are not contested the same way and would skew the fit.
        if detail.get("type", {}).get("name") in {
            "Corner", "Free Kick", "Throw-in", "Kick Off", "Goal Kick", "Penalty",
        }:
            continue
        frame = by_uuid.get(event.get("id"))
        if frame is None:
            continue

        start = event.get("location")
        end = detail.get("end_location")
        if not start or not end:
            continue

        opponents = [
            p["location"]
            for p in frame["freeze_frame"]
            if not p.get("teammate") and not p.get("keeper")
        ]
        if len(opponents) < MIN_VISIBLE_OPPONENTS:
            continue

        start_m = to_metres(start[0], start[1])
        end_m = to_metres(end[0], end[1])
        defenders = np.array([to_metres(p[0], p[1]) for p in opponents])

        margin = interception_margin(start_m, end_m, defenders)
        if not math.isfinite(margin):
            continue

        out.append(
            Sample(
                margin_s=margin,
                completed=detail.get("outcome") is None,
                distance_m=math.dist(start_m, end_m),
            )
        )

    return out


def fit_logistic(samples: list[Sample], use_distance: bool = True) -> tuple[list[float], dict]:
    """Fit P(complete) = sigmoid(a + b*margin + c*distance) by Newton's method.

    Plain Newton rather than a scikit-learn dependency: two features and a few
    dozen iterations of arithmetic, and the pipeline already treats numpy as its
    only numerical dependency.

    Distance earns its place empirically rather than by assumption. The margin
    alone conflates two things, because a long pass has more path for a defender
    to reach and is also harder to strike accurately, and only the first of those
    is in the interception race. `use_distance=False` fits the one-feature model
    so the two can be compared on skill.
    """
    x = np.array([s.margin_s for s in samples])
    y = np.array([1.0 if s.completed else 0.0 for s in samples])
    d = np.array([s.distance_m for s in samples])

    # Margins saturate: a pass the ball wins by three seconds is not meaningfully
    # safer than one it wins by two, and leaving the tail unbounded lets a
    # handful of enormous margins dominate the slope.
    x = np.clip(x, -1.5, 2.0)
    # Scaled to roughly the same range as the margin so Newton is well
    # conditioned, and so the two coefficients are readable side by side.
    d = np.clip(d, 0.0, 60.0) / 30.0

    columns = [np.ones_like(x), x] + ([d] if use_distance else [])
    design = np.column_stack(columns)
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        gradient = design.T @ (y - p)
        w = np.clip(p * (1 - p), 1e-6, None)
        hessian = design.T @ (design * w[:, None])
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-9:
            break

    p_all = 1.0 / (1.0 + np.exp(-design @ beta))

    # Reliability: observed completion rate against predicted, by margin bin.
    # This is the check that matters. A model can have good aggregate skill and
    # still be wrong in the band that decides whether a lane is worth playing.
    edges = np.array([-1.5, -0.75, -0.4, -0.2, -0.05, 0.1, 0.3, 0.6, 1.0, 2.0])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 30:
            continue
        predicted = p_all[mask]
        bins.append(
            {
                "margin_from": round(float(lo), 2),
                "margin_to": round(float(hi), 2),
                "passes": int(mask.sum()),
                "observed": round(float(y[mask].mean()), 4),
                "predicted": round(float(predicted.mean()), 4),
            }
        )

    diagnostics = {
        "samples": len(samples),
        "base_rate": round(float(y.mean()), 4),
        "brier": round(float(np.mean((p_all - y) ** 2)), 5),
        # Against always predicting the base rate. Positive means the margin
        # carries information; near zero would mean the race explains nothing.
        "brier_skill": round(
            float(1 - np.mean((p_all - y) ** 2) / np.mean((y.mean() - y) ** 2)), 4
        ),
        "uses_distance": use_distance,
        "reliability": bins,
    }
    return [float(v) for v in beta], diagnostics


def load_match(match_id: int) -> tuple[list[dict], list[dict]] | None:
    try:
        events = _fetch(f"{OPEN_DATA}/events/{match_id}.json")
        frames = _fetch(f"{OPEN_DATA}/three-sixty/{match_id}.json")
    except Exception:
        return None
    return events, frames  # type: ignore[return-value]
