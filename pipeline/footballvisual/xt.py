"""Expected Threat: what a position on the pitch is worth.

The tactical engine can already say a lane is open. It cannot say whether the
pass is *worth playing*, because it has no notion that a metre gained near the
opposition box matters more than a metre gained in your own half. Expected
Threat supplies exactly that: a scalar per pitch cell, the probability that a
possession starting there ends in a goal within the next few actions.

The formulation is Karun Singh's, and the recursion is the whole idea. From any
cell a team either shoots or moves the ball:

    xT(z) = s(z) * g(z)  +  m(z) * sum over z' of T(z -> z') * xT(z')

where `s` and `m` are the shares of actions from `z` that are shots and moves,
`g` is the conversion rate of shots from `z`, and `T` is where moves from `z`
end up. Value flows backwards from the goal through the passes that get there,
so the model learns that the half spaces are worth more than the touchline
without anyone saying so.

That is a fixed point, and it is reached by iteration. Starting from the pure
shooting value (xT = s*g everywhere) and re-substituting, each pass adds one
more action of look-ahead, so pass k values a possession that scores within k
actions. It converges because `m < 1` strictly wherever there is any shooting
at all, which makes the operator a contraction.

Trained here on StatsBomb open data, which is free and public. The grid it
emits is committed to the repo, because training needs network access and
nothing else in this project does.

Note on coordinates: StatsBomb uses a 120x80 pitch, origin at a corner, with
the attacking direction always +x. This project uses metres, origin at the
centre spot. Conversion happens once, at read time, in `_to_cell`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

OPEN_DATA = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb's nominal pitch. Not 105x68: their coordinates are in yards-ish
# units on a fixed 120x80 grid regardless of the real stadium, which is what
# makes events from different grounds comparable in the first place.
SB_LENGTH = 120.0
SB_WIDTH = 80.0

# Grid resolution. 12x8 rather than Singh's 16x12 because the transition matrix
# has cells^2 entries to estimate, and 96 cells means 9,216 transition
# probabilities against roughly 200k moves. At 16x12 that becomes 36,864 pairs
# and the tail of the matrix is mostly noise. 12x8 also lands close to the
# thirds-and-channels the shape is described in elsewhere in this project.
GRID_X = 12
GRID_Y = 8

# Set-piece restarts are excluded: a corner is not a thing a team can choose to
# do from open play, so letting corners contribute their conversion rate to the
# cell they are taken from would value the corner flag like a chance.
DEAD_BALL = {"Corner", "Free Kick", "Throw-in", "Kick Off", "Goal Kick", "Penalty"}


@dataclass
class Counts:
    """Raw tallies, before any probability is computed."""

    shots: np.ndarray = field(default_factory=lambda: np.zeros((GRID_Y, GRID_X)))
    goals: np.ndarray = field(default_factory=lambda: np.zeros((GRID_Y, GRID_X)))
    moves: np.ndarray = field(default_factory=lambda: np.zeros((GRID_Y, GRID_X)))
    # (from_y, from_x, to_y, to_x); the destinations of successful moves.
    transitions: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID_Y, GRID_X, GRID_Y, GRID_X))
    )

    def add(self, other: "Counts") -> None:
        self.shots += other.shots
        self.goals += other.goals
        self.moves += other.moves
        self.transitions += other.transitions

    @property
    def actions(self) -> int:
        return int(self.shots.sum() + self.moves.sum())


def _to_cell(x: float, y: float) -> tuple[int, int] | None:
    """StatsBomb coordinates to a (row, col) grid cell.

    Clamped rather than dropped at the edges: StatsBomb occasionally records a
    location fractionally off the pitch, and a ball on the byline is a real
    position rather than a corrupt one.
    """
    if x is None or y is None:
        return None
    cx = int(np.clip(x / SB_LENGTH * GRID_X, 0, GRID_X - 1))
    cy = int(np.clip(y / SB_WIDTH * GRID_Y, 0, GRID_Y - 1))
    return cy, cx


def _fetch(url: str) -> object:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def list_matches(competition_id: int, season_id: int) -> list[int]:
    matches = _fetch(f"{OPEN_DATA}/matches/{competition_id}/{season_id}.json")
    return [m["match_id"] for m in matches]  # type: ignore[index]


def count_match(events: Iterable[dict]) -> Counts:
    """Tally shots, goals, moves and transitions for one match.

    A "move" is a pass or a carry that kept possession. Failed moves are counted
    in the denominator (they were still an action taken from that cell) but
    contribute no transition, which is what makes a risky pass into the box
    worth less than its destination alone would suggest.
    """
    counts = Counts()

    for event in events:
        kind = event.get("type", {}).get("name")
        if kind not in ("Pass", "Carry", "Shot"):
            continue

        start = _to_cell(*event.get("location", [None, None])[:2])
        if start is None:
            continue
        sy, sx = start

        if kind == "Shot":
            shot = event.get("shot", {})
            if shot.get("type", {}).get("name") in DEAD_BALL:
                continue
            counts.shots[sy, sx] += 1
            if shot.get("outcome", {}).get("name") == "Goal":
                counts.goals[sy, sx] += 1
            continue

        detail = event.get(kind.lower(), {})
        if kind == "Pass" and detail.get("type", {}).get("name") in DEAD_BALL:
            continue

        counts.moves[sy, sx] += 1

        # An outcome key on a pass means it did not reach a team mate.
        # Carries have no outcome and are complete by construction.
        if kind == "Pass" and detail.get("outcome") is not None:
            continue

        end = _to_cell(*detail.get("end_location", [None, None])[:2])
        if end is None:
            continue
        counts.transitions[sy, sx, end[0], end[1]] += 1

    return counts


def iter_events(match_ids: Iterable[int], progress: bool = True) -> Iterator[list[dict]]:
    """Stream one match of events at a time.

    Deliberately a generator that yields and forgets. A season of events is
    hundreds of megabytes, and none of it is needed once it has been tallied, so
    nothing is written to disk and only one match is ever resident.
    """
    ids = list(match_ids)
    for i, match_id in enumerate(ids, 1):
        try:
            events = _fetch(f"{OPEN_DATA}/events/{match_id}.json")
        except Exception as error:  # network, or a match with no event file
            if progress:
                print(f"  skip {match_id}: {error}")
            continue
        if progress and (i % 10 == 0 or i == len(ids)):
            print(f"  {i}/{len(ids)} matches")
        yield events  # type: ignore[misc]


def symmetrise(counts: Counts) -> Counts:
    """Fold the tallies about the halfway line's long axis.

    The pitch is mirror symmetric across its length, so the left wing and the
    right wing are the same place. Any difference the data shows between them is
    overwhelmingly noise, and adding each cell to its mirror doubles the sample
    behind every estimate. This is the same reflection argument the retrieval
    embedding rests on, applied to the training set instead of a query.

    It is not free of assumptions. Real football is slightly asymmetric, because
    most players are right footed, so this deliberately trades a small real
    effect for a large reduction in variance. With a few hundred matches that is
    clearly the right side of the trade; with tens of thousands it would not be.

    Transitions mirror on both ends at once, since reflecting a pass reflects
    where it started and where it finished.
    """
    folded = Counts()
    for name in ("shots", "goals", "moves"):
        raw = getattr(counts, name)
        setattr(folded, name, raw + raw[::-1, :])
    folded.transitions = counts.transitions + counts.transitions[::-1, :, ::-1, :]
    return folded


def solve(
    counts: Counts,
    max_iterations: int = 400,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, list[float]]:
    """Iterate the recursion to its fixed point.

    Runs to a tolerance rather than a fixed iteration count, and returns the
    per-iteration maximum change so convergence is evidence rather than an
    assumption.

    The iteration count needs to be large. Convergence is geometric at the rate
    of the move share `m`, and away from the box barely one action in a hundred
    is a shot, so `m` sits at about 0.99 and each pass shrinks the error by only
    that much. A dozen iterations looks plausible and leaves the build-up third
    still climbing, which understates exactly the part of the pitch this
    project's clip is played in. Hundreds of iterations of a 96x96 contraction
    costs milliseconds, so there is no reason to stop early.
    """
    total = counts.shots + counts.moves
    with np.errstate(invalid="ignore", divide="ignore"):
        shot_share = np.where(total > 0, counts.shots / total, 0.0)
        move_share = np.where(total > 0, counts.moves / total, 0.0)
        goal_rate = np.where(counts.shots > 0, counts.goals / np.maximum(counts.shots, 1), 0.0)

    # Transitions are divided by moves *attempted*, not moves that succeeded.
    #
    # This is load bearing and it is easy to get wrong. Normalising each row to
    # sum to 1 would say that a move from this cell always arrives somewhere,
    # which deletes the only absorbing state the chain has. With `m` at about
    # 0.99 and passes able to reach anywhere by some chain, value then diffuses
    # until every cell holds the same number: the fixed point becomes "a
    # possession eventually ends in a goal from wherever you are", which is
    # true, useless, and not what xT means.
    #
    # Dividing by attempts leaves each row summing to that cell's completion
    # rate. The missing mass is the turnover, it absorbs at zero value, and the
    # recursion gets the contraction it needs. The symptom of getting this wrong
    # is a grid that is nearly flat and slightly *decreasing* toward the goal,
    # which is what shipped for one run of this file before the tolerance loop
    # made it visible. Twelve iterations hid it, because diffusion had not
    # finished yet.
    attempts = counts.moves[:, :, None, None]
    transition = np.where(attempts > 0, counts.transitions / np.maximum(attempts, 1), 0.0)

    xt = np.zeros((GRID_Y, GRID_X))
    deltas: list[float] = []
    for _ in range(max_iterations):
        # Value carried in from every destination, weighted by how often a move
        # from this cell arrives there.
        carried = np.einsum("ijkl,kl->ij", transition, xt)
        updated = shot_share * goal_rate + move_share * carried
        delta = float(np.abs(updated - xt).max())
        deltas.append(delta)
        xt = updated
        if delta < tolerance:
            break

    return xt, deltas


def save_counts(counts: Counts, path: str, matches: int) -> None:
    np.savez_compressed(
        path,
        shots=counts.shots,
        goals=counts.goals,
        moves=counts.moves,
        transitions=counts.transitions,
        matches=np.array([matches]),
    )


def load_counts(path: str) -> tuple[Counts, int]:
    data = np.load(path)
    counts = Counts()
    counts.shots = data["shots"]
    counts.goals = data["goals"]
    counts.moves = data["moves"]
    counts.transitions = data["transitions"]
    return counts, int(data["matches"][0])


def train(
    seasons: list[tuple[int, int]],
    limit_per_season: int | None = None,
    fold: bool = True,
    cache: str | None = None,
) -> dict:
    """Tally every requested season and solve, returning a JSON-ready model.

    `cache` stores the raw tallies. Downloading 600 matches takes ten minutes and
    solving takes milliseconds, so anything that changes only the solver, the
    grid resolution or the folding should never pay for the download twice.
    """
    counts = Counts()
    used = 0

    if cache and os.path.exists(cache):
        counts, used = load_counts(cache)
        print(f"loaded tallies for {used} matches from {cache}")
    else:
        for competition_id, season_id in seasons:
            match_ids = list_matches(competition_id, season_id)
            if limit_per_season:
                match_ids = match_ids[:limit_per_season]
            print(f"competition {competition_id} season {season_id}: {len(match_ids)} matches")
            for events in iter_events(match_ids):
                counts.add(count_match(events))
                used += 1
        if cache:
            save_counts(counts, cache, used)
            print(f"cached tallies to {cache}")

    raw_actions = counts.actions
    xt, deltas = solve(symmetrise(counts) if fold else counts)
    return {
        "grid": [[round(v, 6) for v in row] for row in xt.tolist()],
        "rows": GRID_Y,
        "cols": GRID_X,
        "matches": used,
        "actions": raw_actions,
        "shots": int(counts.shots.sum()),
        "goals": int(counts.goals.sum()),
        "mirrored": fold,
        "iterations": len(deltas),
        "convergence": [round(d, 10) for d in deltas[-3:]],
        "source": "StatsBomb open data",
        "seasons": [{"competition_id": c, "season_id": s} for c, s in seasons],
    }
