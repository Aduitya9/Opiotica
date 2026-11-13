"""vCDR, convention offsets, referral rates and the agreement statistics.

The vertical cup-to-disc ratio is the screening measurement; referral cut-offs in the clinical
literature are 0.6 and 0.7. Everything downstream of segmentation reduces to this one number,
which is exactly why whose outlines produced it matters.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

CUTOFFS = (0.6, 0.7)
DELTA_ANN = 0.269  # user1 - user7, measured on all 1,200 REFUGE images (pilot R2/R6)


def vertical_extent(mask: np.ndarray) -> int:
    """Number of rows the mask spans — the 'vertical diameter' used by the vCDR."""
    rows = np.nonzero(mask.any(axis=1))[0]
    return int(rows.max() - rows.min() + 1) if rows.size else 0


def vcdr(cup: np.ndarray, disc: np.ndarray) -> float:
    """Vertical cup-to-disc ratio. NaN rather than inf when the disc mask is empty."""
    d = vertical_extent(disc)
    return vertical_extent(cup) / d if d else float("nan")


def referral_rate(values, cutoff: float = 0.6) -> float:
    """Fraction of eyes referred at a fixed vCDR cut-off."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float((v >= cutoff).mean()) if v.size else float("nan")


def percentile_cutoff(values, top_fraction: float = 0.15) -> float:
    """The label-free cut-off that refers the top `top_fraction` of this population.

    Cancels any monotone distortion of style, so it is the SHIFT fix that needs no labels.
    """
    v = np.asarray(values, dtype=float)
    return float(np.nanquantile(v, 1.0 - top_fraction))


def offsets(wide: pd.DataFrame, columns=None) -> pd.Series:
    """Each column's mean signed difference from the row-wise mean of `columns`."""
    cols = list(columns or wide.columns)
    consensus = wide[cols].mean(axis=1)
    return wide[cols].sub(consensus, axis=0).mean(axis=0)


def pairwise_disagreement(decisions: np.ndarray) -> float:
    """Mean fraction of eyes on which two label sources make a different referral call."""
    n = decisions.shape[1]
    return float(
        np.mean(
            [
                (decisions[:, i] != decisions[:, j]).mean()
                for i, j in itertools.combinations(range(n), 2)
            ]
        )
    )


def icc(wide: pd.DataFrame) -> tuple[float, float]:
    """ICC(2,1) absolute agreement and ICC(3,1) consistency, two-way random/mixed, single rater.

    The gap between them is the part of the disagreement that is systematic — the part a level
    correction can remove without any labels.
    """
    x = wide.dropna().to_numpy(dtype=float)
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (resid**2).sum() / ((n - 1) * (k - 1))
    icc21 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    icc31 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
    return float(icc21), float(icc31)


def cluster_bootstrap(values, groups, statistic=np.mean, n: int = 10_000, seed: int = 0):
    """Bootstrap CI resampling whole clusters (patients), not individual eyes.

    PAPILA has two eyes per patient; treating them as independent narrows every interval
    that is reported about it.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    keys = np.unique(groups)
    index = {k: np.nonzero(groups == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for b in range(n):
        picked = rng.choice(keys, size=keys.size, replace=True)
        draws[b] = statistic(np.concatenate([values[index[k]] for k in picked]))
    return float(statistic(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def convention_spread(frame: pd.DataFrame, columns, cutoff: float = 0.6) -> pd.DataFrame:
    """Per-eye min, max and spread across conventions, and whether the cut-off is straddled.

    This replaces conformal prediction as the primary uncertainty measure: it needs no ground
    truth, because there is no true vCDR to be calibrated against.
    """
    cols = list(columns)
    out = pd.DataFrame(index=frame.index)
    out["min"] = frame[cols].min(axis=1)
    out["max"] = frame[cols].max(axis=1)
    out["spread"] = out["max"] - out["min"]
    out["convention_sensitive"] = (out["min"] < cutoff) & (out["max"] >= cutoff)
    return out


def gate(delta_model: float, sd_seed: float, delta_ann: float = DELTA_ANN) -> dict:
    """The pre-registered day-4 decision. See PREREGISTRATION.md §5. No threshold moves here."""
    retention = delta_model / delta_ann
    passed = bool(delta_model > 0.135 and delta_model > 3 * sd_seed)
    return {
        "delta_model": float(delta_model),
        "delta_ann": float(delta_ann),
        "retention": float(retention),
        "sd_seed": float(sd_seed),
        "noise_floor_3sd": float(3 * sd_seed),
        "passes_half_of_annotation_gap": bool(delta_model > 0.135),
        "clears_seed_noise": bool(delta_model > 3 * sd_seed),
        "PASS": passed,
        "verdict": (
            "PASS - models inherit the convention; continue to the full 9-source grid."
            if passed
            else "FAIL - report the negative result: conventions differ in annotations but are "
            "washed out by training. Do not change the design or the threshold."
        ),
    }
