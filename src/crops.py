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


def square_box(disc: np.ndarray, shape: tuple[int, int], margin: float = DEFAULT_MARGIN,
               clamp: bool = False):
    """A square box centred on the disc, with side = margin x the disc's larger diameter.

    The box MAY extend past the image border; `crop` pads those parts. That is deliberate.

    An earlier version clamped the box to stay inside the image, which silently shrank it for
    discs near an edge: `half` collapsed to the edge distance, so the box could no longer hold
    the seven experts' disc union however large the margin grew. Measured on all 1,200 REFUGE
    eyes, that capped containment at 1190 and made the pre-registered rule - "the smallest
    margin containing the union on 100% of eyes" - unsatisfiable at any margin. The ten
    failures all sat about one disc radius from a border (median nearest edge 167 px, against
    419 px for the release as a whole). Padding instead reaches 1200/1200 at margin 1.8.

    `clamp=True` reproduces the old behaviour, and exists only so the regression test can show
    the difference.
    """
    top, left, bottom, right = mask_box(disc)
    cy, cx = (top + bottom) / 2.0, (left + right) / 2.0
    half = margin * max(bottom - top, right - left) / 2.0
    if clamp:
        height, width = shape
        half = min(half, cy, cx, height - cy, width - cx)
    return (
        int(round(cy - half)),
        int(round(cx - half)),
        int(round(cy + half)),
        int(round(cx + half)),
    )


def crop(image: np.ndarray, box, fill: int = 0) -> np.ndarray:
    """Apply a box to an image or mask, padding wherever the box falls outside the image.

    Identical box -> identical pixels, for every label source. Plain slicing cannot be used
    here: a negative index would wrap around to the far side of the array and silently paste
    the wrong part of the retina into the crop.

    Fundus photographs already sit on a black surround, so `fill=0` matches the image.
    """
    top, left, bottom, right = (int(v) for v in box)
    height, width = image.shape[:2]
    out = np.full((bottom - top, right - left) + image.shape[2:], fill, dtype=image.dtype)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(bottom, height), min(right, width)
    if y0 < y1 and x0 < x1:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    return out


def padding_fraction(box, shape) -> float:
    """How much of the box falls outside the image. Reported, never silently ignored."""
    top, left, bottom, right = box
    height, width = shape
    inside = (max(0, min(bottom, height) - max(top, 0))
              * max(0, min(right, width) - max(left, 0)))
    total = (bottom - top) * (right - left)
    return float(1.0 - inside / total) if total else 1.0


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
