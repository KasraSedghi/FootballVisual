"""Ground-truth motion model for the synthetic clip.

This generates *what actually happens on the pitch*, in metres, independent of
any camera. The renderer in `synth.py` then photographs it, and the vision
pipeline tries to recover it. Because this module is the ground truth, the
recovered positions can be scored against it, which is the whole reason the
synthetic path exists.

The scenario is not random motion. It is a specific, tactically legible
situation: a team building up against a 4-4-2 mid block, ending with a pass
into the space between the opponent's midfield and defensive lines. That gives
the tactical analysis engine a real answer to find rather than noise to
describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import pitch

# Physically plausible constants, used both here and by the tactics engine.
PLAYER_HEIGHT_M = 1.80
BALL_RADIUS_M = 0.11
PASS_SPEED_MPS = 16.0


@dataclass(frozen=True)
class PlayerSpec:
    """A player's identity and tactical home position.

    `role` is kept because it makes the emitted ground truth readable when
    debugging, and because the sandbox displays it. `home` is the formation
    position the player returns to when nothing is pulling them elsewhere.
    """

    number: int
    team: str
    role: str
    home: tuple[float, float]
    is_keeper: bool = False


@dataclass
class Pass:
    """A single pass event: leaves `frm` at `start_s`, arrives at `to`."""

    start_s: float
    frm: int
    to: int


@dataclass
class ScenarioFrame:
    """Ground truth for one instant."""

    frame: int
    time_s: float
    positions: dict[int, tuple[float, float]]
    ball: tuple[float, float]
    ball_height: float
    possessor: int | None


# Blue attacks toward +x (the right-hand goal) in a 4-3-3. Numbers are shirt
# numbers, and they are the stable identity the ground truth is keyed on.
BLUE: list[PlayerSpec] = [
    PlayerSpec(1, "blue", "GK", (-46.0, 0.0), is_keeper=True),
    PlayerSpec(5, "blue", "CB", (2.0, -8.0)),
    PlayerSpec(6, "blue", "CB", (2.0, 8.0)),
    PlayerSpec(3, "blue", "LB", (9.0, -26.0)),
    PlayerSpec(2, "blue", "RB", (9.0, 26.0)),
    PlayerSpec(4, "blue", "DM", (13.0, 0.0)),
    PlayerSpec(8, "blue", "CM", (19.0, -11.0)),
    PlayerSpec(10, "blue", "CM", (20.0, 10.0)),
    PlayerSpec(11, "blue", "LW", (30.0, -27.0)),
    PlayerSpec(7, "blue", "RW", (30.0, 27.0)),
    PlayerSpec(9, "blue", "ST", (33.0, 1.0)),
]

# Red defends the +x goal in a 4-4-2 mid block. The gap this scenario is built
# around is the ~10m corridor between the midfield line at x=28 and the back
# line at x=38.
RED: list[PlayerSpec] = [
    PlayerSpec(21, "red", "GK", (49.0, 0.0), is_keeper=True),
    PlayerSpec(22, "red", "RB", (38.0, -15.0)),
    PlayerSpec(23, "red", "CB", (39.0, -5.0)),
    PlayerSpec(24, "red", "CB", (39.0, 5.0)),
    PlayerSpec(25, "red", "LB", (38.0, 15.0)),
    PlayerSpec(26, "red", "RM", (28.0, -19.0)),
    PlayerSpec(27, "red", "CM", (28.0, -6.0)),
    PlayerSpec(28, "red", "CM", (28.0, 6.0)),
    PlayerSpec(29, "red", "LM", (28.0, 19.0)),
    PlayerSpec(30, "red", "ST", (18.0, -5.0)),
    PlayerSpec(31, "red", "ST", (18.0, 5.0)),
]

ALL_PLAYERS: list[PlayerSpec] = BLUE + RED
PLAYERS_BY_NUMBER: dict[int, PlayerSpec] = {p.number: p for p in ALL_PLAYERS}

# The possession sequence. It works the ball right, then plays the line-breaking
# pass to the striker dropping into the gap between the red lines.
PASS_SEQUENCE: list[Pass] = [
    Pass(1.6, 4, 10),
    Pass(3.4, 10, 2),
    Pass(5.2, 2, 7),
    Pass(8.2, 7, 9),
]


@dataclass
class Scenario:
    """A complete synthetic passage of play.

    Motion is deterministic given `seed`, so a regression test can assert exact
    positions and the rendered clip is reproducible.
    """

    fps: int = 25
    duration_s: float = 10.0
    seed: int = 11
    players: list[PlayerSpec] = field(default_factory=lambda: list(ALL_PLAYERS))
    passes: list[Pass] = field(default_factory=lambda: list(PASS_SEQUENCE))

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Per-player smooth wander, built from a couple of sine components with
        # random phase. Cheap, C-infinity, and it never drifts away from home
        # the way integrated random noise would.
        self._phase = {
            p.number: self._rng.uniform(0.0, 2.0 * math.pi, size=4) for p in self.players
        }
        self._wander_amp = {
            p.number: (0.0, 0.0) if p.is_keeper else (1.1, 1.5) for p in self.players
        }

    @property
    def num_frames(self) -> int:
        return int(round(self.fps * self.duration_s))

    # -- ball ------------------------------------------------------------

    def _pass_flight(self, idx: int) -> tuple[float, float]:
        """Start and end time of pass `idx`, from its length at pass speed."""
        p = self.passes[idx]
        a = np.asarray(PLAYERS_BY_NUMBER[p.frm].home)
        b = np.asarray(PLAYERS_BY_NUMBER[p.to].home)
        dist = float(np.linalg.norm(b - a))
        return p.start_s, p.start_s + max(0.25, dist / PASS_SPEED_MPS)

    def _possession_state(self, t: float) -> tuple[int | None, int | None, float]:
        """Who has the ball at time `t`.

        Returns `(carrier, receiver, progress)`. While the ball is at a player's
        feet, `carrier` is set and `receiver` is None. Mid-pass, both are set
        and `progress` runs 0 to 1. This shape means the caller does not have to
        re-derive whether a pass is in flight.
        """
        carrier = self.passes[0].frm if self.passes else None
        for i, p in enumerate(self.passes):
            start, end = self._pass_flight(i)
            if t < start:
                break
            if start <= t < end:
                return p.frm, p.to, (t - start) / (end - start)
            carrier = p.to
        return carrier, None, 0.0

    # -- player motion ---------------------------------------------------

    def _ball_anchor(self, t: float) -> tuple[float, float]:
        """Where the teams *think* the ball is, for collective shifting.

        Using the pass origin for the whole flight would make the block jerk on
        arrival; using the live ball position makes defenders react instantly,
        which is superhuman. Interpolating with an ease gives the block the
        slight lag that real defensive shifts have.
        """
        carrier, receiver, prog = self._possession_state(t)
        if carrier is None:
            return (0.0, 0.0)
        a = PLAYERS_BY_NUMBER[carrier].home
        if receiver is None:
            return a
        b = PLAYERS_BY_NUMBER[receiver].home
        ease = prog * prog * (3.0 - 2.0 * prog)
        return (a[0] + (b[0] - a[0]) * ease, a[1] + (b[1] - a[1]) * ease)

    def _striker_run(self, t: float) -> tuple[float, float]:
        """Blue 9 drops off the back line into the gap between the red lines.

        This is the scripted piece of intent in the scenario. Without it the
        striker sits on the last defender and there is no line-breaking pass for
        the tactics engine to find.
        """
        start, peak = 5.6, 8.6
        if t <= start:
            return (0.0, 0.0)
        u = min(1.0, (t - start) / (peak - start))
        ease = u * u * (3.0 - 2.0 * u)
        # Backwards into the pocket at roughly x=33, and slightly to the right
        # to line up with where the ball is coming from.
        return (-1.0 * ease, 4.5 * ease)

    def positions_at(self, t: float) -> dict[int, tuple[float, float]]:
        bx, by = self._ball_anchor(t)
        out: dict[int, tuple[float, float]] = {}

        for spec in self.players:
            hx, hy = spec.home
            if spec.is_keeper:
                # Keepers track the ball laterally but stay near their line.
                out[spec.number] = (hx, float(np.clip(by * 0.25, -6.0, 6.0)))
                continue

            # Collective shift toward the ball. The defending side slides harder
            # and stays compact; the team in possession spreads to offer angles.
            if spec.team == "red":
                shift_y = 0.55 * by
                shift_x = 0.22 * (bx - 20.0)
            else:
                shift_y = 0.18 * by
                shift_x = 0.12 * (bx - 15.0)

            ph = self._phase[spec.number]
            ax, ay = self._wander_amp[spec.number]
            wander_x = ax * math.sin(0.55 * t + ph[0]) + 0.4 * ax * math.sin(1.3 * t + ph[1])
            wander_y = ay * math.sin(0.47 * t + ph[2]) + 0.4 * ay * math.sin(1.1 * t + ph[3])

            x = hx + shift_x + wander_x
            y = hy + shift_y + wander_y

            if spec.number == 9 and spec.team == "blue":
                rx, ry = self._striker_run(t)
                x += rx
                y += ry

            x = float(np.clip(x, -pitch.HALF_LENGTH + 1.0, pitch.HALF_LENGTH - 1.0))
            y = float(np.clip(y, -pitch.HALF_WIDTH + 1.0, pitch.HALF_WIDTH - 1.0))
            out[spec.number] = (x, y)

        return out

    def ball_at(self, t: float, positions: dict[int, tuple[float, float]]) -> tuple[
        tuple[float, float], float, int | None
    ]:
        """Ball position, height, and possessor at time `t`.

        Takes the already-computed player positions so the ball lands on the
        receiver's *actual* position rather than their formation home, which is
        what makes the pass look like it was aimed at a moving target.
        """
        carrier, receiver, prog = self._possession_state(t)
        if carrier is None:
            return (0.0, 0.0), BALL_RADIUS_M, None

        if receiver is None:
            cx, cy = positions[carrier]
            # At the carrier's feet, nudged toward the goal they attack.
            side = 0.6 if PLAYERS_BY_NUMBER[carrier].team == "blue" else -0.6
            return (cx + side, cy), BALL_RADIUS_M, carrier

        ax, ay = positions[carrier]
        bx, by = positions[receiver]
        x = ax + (bx - ax) * prog
        y = ay + (by - ay) * prog
        # A driven pass lifts slightly off the deck; the parabola is what makes
        # the ball sprite rise off the grass in the render.
        height = BALL_RADIUS_M + 0.9 * math.sin(math.pi * prog)
        return (x, y), height, None

    def frames(self) -> list[ScenarioFrame]:
        out: list[ScenarioFrame] = []
        for i in range(self.num_frames):
            t = i / self.fps
            pos = self.positions_at(t)
            ball, height, possessor = self.ball_at(t, pos)
            out.append(
                ScenarioFrame(
                    frame=i,
                    time_s=t,
                    positions=pos,
                    ball=ball,
                    ball_height=height,
                    possessor=possessor,
                )
            )
        return out

    def ground_truth(self) -> dict:
        """Serialisable ground truth, written next to the rendered clip."""
        return {
            "fps": self.fps,
            "durationS": self.duration_s,
            "players": [
                {
                    "number": p.number,
                    "team": p.team,
                    "role": p.role,
                    "isKeeper": p.is_keeper,
                }
                for p in self.players
            ],
            "passes": [
                {"startS": p.start_s, "from": p.frm, "to": p.to} for p in self.passes
            ],
            "frames": [
                {
                    "frame": f.frame,
                    "timeS": round(f.time_s, 4),
                    "ball": [round(f.ball[0], 3), round(f.ball[1], 3)],
                    "possessor": f.possessor,
                    "positions": {
                        str(num): [round(xy[0], 3), round(xy[1], 3)]
                        for num, xy in f.positions.items()
                    },
                }
                for f in self.frames()
            ],
        }
