"""The src modules must reproduce the pilot numbers this study is built on.

If any of these drift, a number in PREREGISTRATION.md has become wrong and the pre-registration
must be amended rather than quietly corrected.

Run:  python -m pytest whose-eyes/tests -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src import crops, labels, vcdr  # noqa: E402

MEASURED = os.path.join(ROOT, "analysis", "r6_official_merge.csv")
RATERS = [f"user{k}" for k in range(1, 8)]


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(MEASURED, dtype={"folder": str})


def test_release_is_complete(table):
    assert len(table) == 1200
    assert table[RATERS + ["official_vcdr"]].notna().all().all()


def test_expert_offsets_match_the_preregistration(table):
    off = vcdr.offsets(table, RATERS)
    assert off["user1"] == pytest.approx(0.148, abs=0.002)
    assert off["user7"] == pytest.approx(-0.121, abs=0.002)
    # the other five sit within +/- 0.022 of the consensus
    middle = off.drop(["user1", "user7"]).abs()
    assert middle.max() < 0.023


def test_the_annotation_gap_is_the_preregistered_delta(table):
    delta = (table["user1"] - table["user7"]).mean()
    assert delta == pytest.approx(vcdr.DELTA_ANN, abs=0.002)


def test_referral_rates_span_seventeen_fold(table):
    rates = {r: vcdr.referral_rate(table[r], 0.6) for r in RATERS}
    assert rates["user1"] == pytest.approx(0.568, abs=0.005)
    assert rates["user7"] == pytest.approx(0.033, abs=0.005)
    assert max(rates.values()) / min(rates.values()) > 15


def test_official_merge_sits_at_consensus(table):
    """The audit's second finding: the released REFUGE label is NOT skewed."""
    consensus = table[RATERS].mean(axis=1)
    offset = (table["official_vcdr"] - consensus).mean()
    assert abs(offset) < 0.02, "if this grows, the 'official label is well centred' claim dies"
    assert vcdr.referral_rate(table["official_vcdr"], 0.6) == pytest.approx(0.098, abs=0.005)


def test_icc_gap_is_the_systematic_part(table):
    icc21, icc31 = vcdr.icc(table[RATERS])
    assert icc21 == pytest.approx(0.436, abs=0.01)
    assert icc31 == pytest.approx(0.689, abs=0.01)
    assert icc31 > icc21, "consistency must exceed absolute agreement, or there is no level shift"


def test_percentile_cutoff_equalises_referral_counts(table):
    decisions = np.column_stack(
        [(table[r] >= vcdr.percentile_cutoff(table[r], 0.15)).to_numpy() for r in RATERS]
    )
    counts = decisions.sum(axis=0)
    # Not exactly equal: `>=` admits every eye tied at the quantile, and vCDR values repeat.
    # Measured spread is 180-184 of 1,200 (2%), against 40-682 for a fixed 0.6 cut-off.
    assert counts.max() - counts.min() <= 0.05 * counts.min(), "referred counts must be ~equal"
    assert (table[RATERS] >= 0.6).sum().max() / (table[RATERS] >= 0.6).sum().min() > 15
    fixed = (table[RATERS] >= 0.6).to_numpy()
    assert vcdr.pairwise_disagreement(decisions) < vcdr.pairwise_disagreement(fixed)


def test_convention_spread_flags_straddling_eyes(table):
    spread = vcdr.convention_spread(table, RATERS, cutoff=0.6)
    assert (spread["spread"] >= 0).all()
    straddle = spread["convention_sensitive"].mean()
    assert 0.2 < straddle < 0.8, f"implausible straddle fraction {straddle:.3f}"


def test_vcdr_is_nan_not_inf_on_an_empty_disc():
    """The r5 bug: an empty disc mask returned inf and silently poisoned every comparison."""
    cup = np.zeros((10, 10), dtype=bool)
    cup[4:6, 4:6] = True
    assert np.isnan(vcdr.vcdr(cup, np.zeros((10, 10), dtype=bool)))


def test_gate_is_the_preregistered_rule():
    clear = vcdr.gate(delta_model=0.20, sd_seed=0.01)
    assert clear["PASS"] and clear["retention"] > 0.5
    too_small = vcdr.gate(delta_model=0.10, sd_seed=0.01)
    assert not too_small["PASS"]
    noisy = vcdr.gate(delta_model=0.20, sd_seed=0.09)  # 3 x 0.09 = 0.27 > 0.20
    assert not noisy["PASS"], "an effect inside the seed-noise floor must never pass"


def test_frozen_box_contains_every_expert_disc():
    """Fix A1 in miniature: one box, several outlines, containment proven not assumed."""
    shape = (200, 200)
    base = np.zeros(shape, dtype=bool)
    base[80:120, 80:120] = True
    liberal = np.zeros(shape, dtype=bool)
    liberal[70:130, 72:128] = True
    box = crops.square_box(base, shape, margin=2.0)
    report = crops.union_containment(box, [base, liberal])
    assert report["contains_union"]
    tight = crops.square_box(base, shape, margin=1.0)
    assert not crops.union_containment(tight, [base, liberal])["contains_union"]
    assert crops.smallest_safe_margin(base, [base, liberal], shape) <= 2.0


PNG_MEASURED = os.path.join(ROOT, "analysis", "r7_png_mask_vcdr.csv")


@pytest.fixture(scope="module")
def png_table() -> pd.DataFrame:
    if not os.path.exists(PNG_MEASURED):
        pytest.skip("r7_png_mask_vcdr.csv not generated yet")
    return pd.read_csv(PNG_MEASURED, dtype={"image_id": str})


def test_png_masks_agree_with_the_ellipse_sheets(png_table):
    """The training targets are the PNG masks; the ellipses are only an analysis convenience.

    If these two ever diverge by more than the tolerance, a number the paper quotes has moved
    and PREREGISTRATION.md needs an amendment - not a quiet edit.
    """
    assert len(png_table) == 1200
    png_off = vcdr.offsets(png_table, RATERS)
    ell_off = vcdr.offsets(pd.read_csv(MEASURED, dtype={"folder": str}), RATERS)
    assert (png_off - ell_off).abs().max() < 0.02, "per-expert offsets drifted apart"


def test_delta_ann_from_png_masks_still_matches_the_gate(png_table):
    """The gate in PREREGISTRATION section 5 is defined against DELTA_ANN. Guard it."""
    delta = float((png_table["user1"] - png_table["user7"]).mean())
    assert delta == pytest.approx(0.2647, abs=0.002)
    assert abs(delta - vcdr.DELTA_ANN) < 0.01, (
        "Delta_ann moved beyond the pre-registered tolerance; amend section 10 before the gate")


def test_user1_referral_rate_is_the_knife_edge_number(png_table):
    """Amendment 3: user1's distribution is centred on the cut-off, so their referral rate
    is the least stable figure in the study. The middle five are not."""
    rates = {r: vcdr.referral_rate(png_table[r], 0.6) for r in RATERS}
    assert rates["user1"] == pytest.approx(0.667, abs=0.01)
    assert rates["user7"] == pytest.approx(0.034, abs=0.01)
    middle = [rates[r] for r in RATERS if r not in ("user1", "user7")]
    assert 0.08 <= min(middle) and max(middle) <= 0.14, "the middle five should be stable"
    assert rates["user1"] / rates["user7"] > 15


def test_edge_adjacent_disc_is_contained_by_padding_not_clipping():
    """Regression: notebook 00's margin table plateaued at 1190/1200 because of this.

    A disc about one radius from the border made the old clamped box collapse to the edge
    distance, so no margin could hold the experts' union. Ten real REFUGE eyes hit it.
    """
    shape = (400, 400)
    disc = np.zeros(shape, dtype=bool)
    disc[150:250, 10:110] = True           # 100 px disc, centre 60 px from the left edge
    liberal = np.zeros(shape, dtype=bool)
    liberal[140:260, 0:130] = True         # a more generous expert, 130 px wide
    # clamped: half collapses 90 -> 60, so the box ends at x=120 and clips the union at x=130

    clamped = crops.square_box(disc, shape, margin=1.8, clamp=True)
    padded = crops.square_box(disc, shape, margin=1.8)
    assert not crops.union_containment(clamped, [disc, liberal])["contains_union"]
    assert crops.union_containment(padded, [disc, liberal])["contains_union"]
    assert padded[1] < 0, "the fixed box must be allowed to extend past the border"


def test_crop_pads_instead_of_wrapping_around():
    """A negative slice index would paste the far side of the retina into the crop."""
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = crops.crop(image, (-3, -3, 5, 5))
    assert out.shape == (8, 8)
    assert (out[:3, :] == 0).all() and (out[:, :3] == 0).all(), "outside must be padding"
    assert (out[3:, 3:] == image[:5, :5]).all(), "inside must be the real pixels"
    assert crops.padding_fraction((-3, -3, 5, 5), (10, 10)) == pytest.approx(1 - 25 / 64)


def test_majority_equals_average_at_half():
    """Why only nine sources are trained: these two rules coincide, as measured in R5."""
    stack = np.random.default_rng(0).random((7, 40, 40)) > 0.5
    assert (labels.fuse(stack, "majority") == (stack.mean(axis=0) >= 0.5)).all()
    assert len(labels.TRAINED) == 9
