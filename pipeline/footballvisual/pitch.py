"""IFAB standard pitch geometry.

Everything downstream speaks *pitch coordinates*: metres, origin at the centre
spot, +x toward the right-hand goal, +y toward the far touchline. A pitch is
105m x 68m, so x runs [-52.5, 52.5] and y runs [-34, 34].

This module is the single source of truth for the pitch. The homography solver
matches against the landmarks defined here, the synthetic renderer draws from
here, and the web sandbox draws the same geometry from the JSON emitted by
`landmark_table()`. Change a number here and it changes everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# IFAB Laws of the Game, Law 1. Professional dimensions.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
GOAL_WIDTH = 7.32
CORNER_ARC_RADIUS = 1.0

HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0


@dataclass(frozen=True)
class Landmark:
    """A named point on the pitch that a calibrator can key on.

    `name` is stable and is what a calibration file refers to, so renaming one
    is a breaking change to any saved calibration.
    """

    name: str
    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def _landmarks() -> list[Landmark]:
    """Build the canonical landmark set.

    These are the points a human (or a line detector) can actually identify in
    a broadcast frame: line intersections and box corners. Points that are
    ambiguous from a single view, like the centre of the centre circle when the
    halfway line is out of frame, are still included because they are useful
    when they *are* visible.
    """
    marks: list[Landmark] = []

    def add(name: str, x: float, y: float) -> None:
        marks.append(Landmark(name, x, y))

    # Pitch corners.
    add("corner_bl", -HALF_LENGTH, -HALF_WIDTH)
    add("corner_tl", -HALF_LENGTH, HALF_WIDTH)
    add("corner_br", HALF_LENGTH, -HALF_WIDTH)
    add("corner_tr", HALF_LENGTH, HALF_WIDTH)

    # Halfway line meeting the touchlines, and the centre spot.
    add("halfway_bottom", 0.0, -HALF_WIDTH)
    add("halfway_top", 0.0, HALF_WIDTH)
    add("centre_spot", 0.0, 0.0)

    # Centre circle extremities. Useful when the halfway line is visible but
    # the touchline intersections are outside a zoomed broadcast frame.
    add("centre_circle_top", 0.0, CENTRE_CIRCLE_RADIUS)
    add("centre_circle_bottom", 0.0, -CENTRE_CIRCLE_RADIUS)

    for side, sign in (("left", -1.0), ("right", 1.0)):
        goal_line_x = sign * HALF_LENGTH
        pen_x = sign * (HALF_LENGTH - PENALTY_AREA_LENGTH)
        goal_area_x = sign * (HALF_LENGTH - GOAL_AREA_LENGTH)
        spot_x = sign * (HALF_LENGTH - PENALTY_SPOT_DISTANCE)

        half_pen_w = PENALTY_AREA_WIDTH / 2.0
        half_goal_w = GOAL_AREA_WIDTH / 2.0
        half_goal_mouth = GOAL_WIDTH / 2.0

        # Penalty area: two corners on the goal line, two out on the D side.
        add(f"{side}_pen_goalline_top", goal_line_x, half_pen_w)
        add(f"{side}_pen_goalline_bottom", goal_line_x, -half_pen_w)
        add(f"{side}_pen_corner_top", pen_x, half_pen_w)
        add(f"{side}_pen_corner_bottom", pen_x, -half_pen_w)

        # Six yard box.
        add(f"{side}_goalarea_goalline_top", goal_line_x, half_goal_w)
        add(f"{side}_goalarea_goalline_bottom", goal_line_x, -half_goal_w)
        add(f"{side}_goalarea_corner_top", goal_area_x, half_goal_w)
        add(f"{side}_goalarea_corner_bottom", goal_area_x, -half_goal_w)

        add(f"{side}_penalty_spot", spot_x, 0.0)
        add(f"{side}_goal_top", goal_line_x, half_goal_mouth)
        add(f"{side}_goal_bottom", goal_line_x, -half_goal_mouth)

    return marks


LANDMARKS: dict[str, Landmark] = {m.name: m for m in _landmarks()}


def landmark(name: str) -> Landmark:
    try:
        return LANDMARKS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"unknown pitch landmark {name!r}; known names: "
            + ", ".join(sorted(LANDMARKS))
        ) from exc


def landmark_table() -> dict[str, list[float]]:
    """Serialisable landmark map, consumed by the web sandbox."""
    return {name: [m.x, m.y] for name, m in LANDMARKS.items()}


def is_inside(x: float, y: float, margin: float = 0.0) -> bool:
    """Whether a pitch coordinate lies within the field of play.

    `margin` widens the acceptance region. Tracking noise routinely pushes a
    touchline player a few centimetres outside the true boundary, and throwing
    those away would be worse than keeping them, so callers that filter for
    plausibility should pass a margin of a metre or two rather than zero.
    """
    return (
        -HALF_LENGTH - margin <= x <= HALF_LENGTH + margin
        and -HALF_WIDTH - margin <= y <= HALF_WIDTH + margin
    )


def pitch_polylines() -> list[np.ndarray]:
    """The pitch markings as polylines in pitch coordinates.

    Used by the synthetic renderer to draw lines, and by the calibration
    overlay to reproject the model onto a real frame so a human can eyeball
    whether the homography is right. Curves are pre-tessellated because
    everything consuming this only knows how to draw straight segments.
    """
    lines: list[np.ndarray] = []

    def add(points: list[tuple[float, float]]) -> None:
        lines.append(np.asarray(points, dtype=np.float64))

    # Touchlines and goal lines.
    add(
        [
            (-HALF_LENGTH, -HALF_WIDTH),
            (HALF_LENGTH, -HALF_WIDTH),
            (HALF_LENGTH, HALF_WIDTH),
            (-HALF_LENGTH, HALF_WIDTH),
            (-HALF_LENGTH, -HALF_WIDTH),
        ]
    )
    # Halfway line.
    add([(0.0, -HALF_WIDTH), (0.0, HALF_WIDTH)])

    # Centre circle.
    theta = np.linspace(0.0, 2.0 * np.pi, 96)
    add(
        [
            (float(CENTRE_CIRCLE_RADIUS * np.cos(t)), float(CENTRE_CIRCLE_RADIUS * np.sin(t)))
            for t in theta
        ]
    )

    for sign in (-1.0, 1.0):
        goal_line_x = sign * HALF_LENGTH
        pen_x = sign * (HALF_LENGTH - PENALTY_AREA_LENGTH)
        goal_area_x = sign * (HALF_LENGTH - GOAL_AREA_LENGTH)
        half_pen_w = PENALTY_AREA_WIDTH / 2.0
        half_goal_w = GOAL_AREA_WIDTH / 2.0

        add(
            [
                (goal_line_x, -half_pen_w),
                (pen_x, -half_pen_w),
                (pen_x, half_pen_w),
                (goal_line_x, half_pen_w),
            ]
        )
        add(
            [
                (goal_line_x, -half_goal_w),
                (goal_area_x, -half_goal_w),
                (goal_area_x, half_goal_w),
                (goal_line_x, half_goal_w),
            ]
        )

        # The D: the arc of the centre-circle radius around the penalty spot
        # that falls outside the penalty area.
        spot_x = sign * (HALF_LENGTH - PENALTY_SPOT_DISTANCE)
        arc = []
        for t in np.linspace(0.0, 2.0 * np.pi, 160):
            px = spot_x + CENTRE_CIRCLE_RADIUS * np.cos(t)
            py = CENTRE_CIRCLE_RADIUS * np.sin(t)
            # Keep only the portion beyond the penalty area edge.
            if (sign > 0 and px < pen_x) or (sign < 0 and px > pen_x):
                arc.append((float(px), float(py)))
            elif arc:
                add(arc)
                arc = []
        if arc:
            add(arc)

    return lines


def goal_centre(side: str) -> tuple[float, float]:
    """Centre of the goal mouth for `side` in ("left", "right")."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return (-HALF_LENGTH if side == "left" else HALF_LENGTH, 0.0)
