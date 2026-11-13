"""The frozen crop: one box per eye, shared by every label source.

Fix A1 of the audit. If the crop moves when the convention changes, the experiment is
contaminated at step one and nothing downstream can repair it, because a model trained on a
liberal expert would also be seeing a different field of view. So:

1. ONE disc localiser is trained, on the official merge only, and then frozen.
2. It emits ONE box per eye. That box is written to `crops/boxes.csv` and never recomputed.
3. Every one of the nine label sources is cropped with that same box - byte-identical pixels.
4. `union_containment` then proves the box holds the UNION of all seven experts' discs. Any
   eye where it does not is reported, not silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MARGIN = 2.0  # box side = margin x disc diameter


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box of a boolean mask as (top, left, bottom, right), end-exclusive."""
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if not rows.size or not cols.size:
        raise ValueError("empty mask")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def square_box(disc: np.ndarray, shape: tuple[int, int], margin: float = DEFAULT_MARGIN):
    """A square box centred on the disc, with side = margin x the disc's larger diameter.

    Clamped to the image, and the side is kept square after clamping so that every crop has the
    same aspect ratio before resizing.
    """
    top, left, bottom, right = mask_box(disc)
    cy, cx = (top + bottom) / 2.0, (left + right) / 2.0
    side = margin * max(bottom - top, right - left)
    half = side / 2.0
    height, width = shape
    half = min(half, cy, cx, height - cy, width - cx)
    return (
        int(round(cy - half)),
        int(round(cx - half)),
        int(round(cy + half)),
        int(round(cx + half)),
    )


def crop(image: np.ndarray, box) -> np.ndarray:
    """Apply a box to an image or mask. Identical box -> identical pixels, for every label."""
    top, left, bottom, right = box
    return image[top:bottom, left:right]


def contains(box, inner_box) -> bool:
    """Does `box` fully contain `inner_box`?"""
    return (
        box[0] <= inner_box[0]
        and box[1] <= inner_box[1]
        and box[2] >= inner_box[2]
        and box[3] >= inner_box[3]
    )


def union_containment(box, expert_discs) -> dict:
    """Check the frozen box against every expert's disc. Reports, never silently drops.

    `expert_discs` is an iterable of boolean masks in full-image coordinates.
    """
    inner = [mask_box(d) for d in expert_discs]
    union = (
        min(b[0] for b in inner),
        min(b[1] for b in inner),
        max(b[2] for b in inner),
        max(b[3] for b in inner),
    )
    return {
        "contains_union": contains(box, union),
        "union_box": union,
        "slack_top": union[0] - box[0],
        "slack_left": union[1] - box[1],
        "slack_bottom": box[2] - union[2],
        "slack_right": box[3] - union[3],
    }


def smallest_safe_margin(disc: np.ndarray, expert_discs, shape, grid=np.arange(1.2, 3.01, 0.1)):
    """The smallest margin at which the box still holds every expert's disc, for this eye.

    Running this over the whole release gives the margin the frozen localiser must use. Choosing
    it from the data, once, before training, is the honest way to fix a hyper-parameter that
    would otherwise be tuned against the outcome.
    """
    for margin in grid:
        box = square_box(disc, shape, margin=float(margin))
        if union_containment(box, expert_discs)["contains_union"]:
            return float(margin)
    return float("nan")


def summarise_boxes(rows) -> pd.DataFrame:
    """Tidy the per-eye containment records into the table that goes into the paper."""
    frame = pd.DataFrame(rows)
    if "contains_union" in frame:
        frame["contains_union"] = frame["contains_union"].astype(bool)
    return frame
