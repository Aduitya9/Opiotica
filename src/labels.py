"""The nine label sources that get trained, and the three that are only analysed.

Measured in pilot R5 on all 1,200 images, same experts, same outlines - only the fusion rule
changes:

    unanimous (7/7)            mean vCDR 0.359   refer@0.6  3.9%   sens 0.38  spec 1.00
    >= 5 of 7                             0.434              6.2%        0.54       0.99
    majority (>= 4 of 7)                  0.456              7.8%        0.64       0.98
    >= 3 of 7 (Drishti-like)              0.481              9.9%        0.71       0.97
    union (>= 1 of 7)                     0.583             32.5%        0.91       0.74
    average >= 0.5 (RIM-ONE-like)         0.456              7.8%        0.64       0.98
    official merge (8th expert)           0.475              9.8%        0.70       0.97

`majority` and `average >= 0.5` are IDENTICAL to three decimals, so only one of them is trained.
That one observation saves six training runs with no loss of information.
"""

from __future__ import annotations

import numpy as np

#: Trained. Nine models, three seeds each.
TRAINED = [f"user{k}" for k in range(1, 8)] + ["majority", "official"]

#: Analysed from the annotations only - never trained. See the docstring for why.
ANALYSED_ONLY = ["unanimous", "union", "staple"]

#: The two extremes that decide the day-4 gate.
EXTREMES = ("user1", "user7")

#: The label used for the seed-noise floor (5 seeds, same label).
NOISE_FLOOR_LABEL = "official"


def fuse(masks, rule: str) -> np.ndarray:
    """Fuse a stack of expert masks (n_raters, H, W) by a named rule."""
    stack = np.asarray(masks, dtype=bool)
    k = stack.shape[0]
    votes = stack.sum(axis=0)
    if rule == "unanimous":
        return votes >= k
    if rule == "majority":
        return votes >= int(np.ceil(k / 2)) + (1 if k % 2 == 0 else 0)
    if rule == "union":
        return votes >= 1
    if rule.startswith("atleast"):
        return votes >= int(rule.replace("atleast", ""))
    raise ValueError(f"unknown fusion rule: {rule}")


def label_sources(expert_cups, expert_discs, official_cup, official_disc) -> dict:
    """Every trained label source for one eye, as (cup, disc) mask pairs.

    `expert_cups` / `expert_discs` are stacks of seven boolean masks in rater order.
    """
    out = {f"user{k + 1}": (expert_cups[k], expert_discs[k]) for k in range(len(expert_cups))}
    out["majority"] = (fuse(expert_cups, "majority"), fuse(expert_discs, "majority"))
    out["official"] = (official_cup, official_disc)
    return out
