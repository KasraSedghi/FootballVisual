"""Train the Expected Threat grid and write it into the web app.

Run explicitly, never as part of `make demo`: it needs network access, takes
several minutes, and produces a model that is committed to the repo precisely so
that nothing else in the project ever needs either.

    python pipeline/train_xt.py

The competitions below are the open-data seasons with enough matches to be worth
tallying. They mix men's and women's football and several eras, which is a real
limitation and is recorded in the model file: a single league-season would be
cleaner football but far too little of it, and xT is dominated by pitch geometry
rather than by competition.
"""

from __future__ import annotations

import json
from pathlib import Path

from footballvisual.xt import train

SEASONS = [
    (55, 282),   # UEFA Euro 2024
    (55, 43),    # UEFA Euro 2020
    (43, 106),   # FIFA World Cup 2022
    (43, 3),     # FIFA World Cup 2018
    (11, 90),    # La Liga 2020/2021
    (7, 108),    # Ligue 1 2021/2022
    (7, 235),    # Ligue 1 2022/2023
    (9, 281),    # 1. Bundesliga 2023/2024
    (1267, 107), # African Cup of Nations 2023
    (44, 107),   # Major League Soccer 2023
    (53, 106),   # UEFA Women's Euro 2022
    (72, 107),   # Women's World Cup 2023
    (37, 42),    # FA Women's Super League 2019/2020
]

OUT = Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "tactics" / "xt-grid.json"


def main() -> None:
    cache = Path(__file__).resolve().parents[1] / "data" / "xt-counts.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    model = train(SEASONS, cache=str(cache))
    OUT.write_text(json.dumps(model, indent=1) + "\n")

    print(f"\nwrote {OUT}")
    print(f"matches {model['matches']}  actions {model['actions']}  goals {model['goals']}")
    print(f"iterations {model['iterations']}  convergence tail {model['convergence']}")
    row = model["rows"] // 2
    print("centre row, own goal to theirs:")
    print("  " + "  ".join(f"{v:.4f}" for v in model["grid"][row]))


if __name__ == "__main__":
    main()
