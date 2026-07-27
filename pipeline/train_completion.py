"""Calibrate pass completion against real outcomes, and write the fitted model.

Run explicitly, like `train_xt.py`: it needs network access and several minutes,
and it exists so that nothing at runtime needs either.

    python pipeline/train_completion.py

Only competitions with StatsBomb 360 data can be used, because the fit needs the
freeze frame of defenders at the moment of the pass. The split is by match
rather than by pass: passes within a match are correlated through team, tactics
and opponent, so a random split over passes would leak and report a skill score
better than the model deserves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footballvisual.completion import fit_logistic, load_match, samples_from_match
from footballvisual.xt import list_matches

# 360 seasons, checked against competitions.json.
SEASONS = [
    (55, 282),   # UEFA Euro 2024
    (55, 43),    # UEFA Euro 2020
    (43, 106),   # FIFA World Cup 2022
    (11, 90),    # La Liga 2020/2021
]

MATCHES_PER_SEASON = 26
HOLDOUT_FRACTION = 0.25

OUT = (
    Path(__file__).resolve().parents[1]
    / "web" / "src" / "lib" / "tactics" / "completion-model.json"
)


def main() -> None:
    train_samples = []
    test_samples = []

    for competition_id, season_id in SEASONS:
        ids = list_matches(competition_id, season_id)[:MATCHES_PER_SEASON]
        cut = int(len(ids) * (1 - HOLDOUT_FRACTION))
        print(f"competition {competition_id} season {season_id}: "
              f"{cut} train, {len(ids) - cut} holdout matches")

        for i, match_id in enumerate(ids):
            loaded = load_match(match_id)
            if loaded is None:
                continue
            got = samples_from_match(*loaded)
            (train_samples if i < cut else test_samples).extend(got)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(ids)}  train={len(train_samples)} test={len(test_samples)}")

    print(f"\ntrain passes {len(train_samples)}, holdout passes {len(test_samples)}")

    # Does distance earn its coefficient? Decided on the holdout, not by taste.
    results = {}
    for use_distance in (False, True):
        beta, diagnostics = fit_logistic(train_samples, use_distance=use_distance)
        x = np.clip(np.array([s.margin_s for s in test_samples]), -1.5, 2.0)
        y = np.array([1.0 if s.completed else 0.0 for s in test_samples])
        d = np.clip(np.array([s.distance_m for s in test_samples]), 0, 60) / 30.0
        columns = [np.ones_like(x), x] + ([d] if use_distance else [])
        p = 1.0 / (1.0 + np.exp(-(np.column_stack(columns) @ np.array(beta))))
        skill = float(1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2))
        results[use_distance] = (beta, diagnostics, skill)
        label = "margin + distance" if use_distance else "margin only"
        print(f"{label:20s} train skill {diagnostics['brier_skill']:.4f}  "
              f"holdout skill {skill:.4f}")

    use_distance = results[True][2] > results[False][2] + 0.002
    beta, diagnostics, holdout_skill = results[use_distance]
    print(f"\nchose: {'margin + distance' if use_distance else 'margin only'}")

    model = {
        "coefficients": [round(b, 6) for b in beta],
        "uses_distance": use_distance,
        "margin_clip": [-1.5, 2.0],
        "distance_scale": 30.0,
        "distance_clip": [0.0, 60.0],
        "train": diagnostics,
        "holdout": {
            "passes": len(test_samples),
            "brier_skill": round(holdout_skill, 4),
        },
        "source": "StatsBomb open data, 360 freeze frames",
        "seasons": [{"competition_id": c, "season_id": s} for c, s in SEASONS],
    }
    OUT.write_text(json.dumps(model, indent=1) + "\n")
    print(f"wrote {OUT}")

    print("\nreliability (observed vs predicted by margin band):")
    for b in diagnostics["reliability"]:
        print(f"  {b['margin_from']:+.2f}..{b['margin_to']:+.2f}  "
              f"n={b['passes']:6d}  observed {b['observed']:.3f}  "
              f"predicted {b['predicted']:.3f}")


if __name__ == "__main__":
    main()
