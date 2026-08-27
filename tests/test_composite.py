"""Candidate generation: constraints, traceability and the pick list."""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.composite.candidates import (
    CV_PRIMARY,
    EST_MASS_KG,
    INTERVAL_IDS,
    generate_candidates,
    pick_list,
    quality,
)
from geomet_sampler.models import Availability


def build_run(
    n: int = 30, *, grade_pct: float = 6.0, length: float = 1.0, hole: str = "H1"
) -> pd.DataFrame:
    """A single usable run of identical intervals, HQ half core at SG 2.7."""
    rows = []
    for i in range(n):
        rows.append(
            {
                M.HOLE_ID: hole,
                M.INTERVAL_ID: f"{hole}_{i}",
                M.PARENT_SAMPLE_ID: f"{hole}_{i}",
                M.RUN_ID: "R1",
                M.FROM_M: i * length,
                M.TO_M: (i + 1) * length,
                M.LENGTH_M: length,
                M.MASS_KG: 3.848 * length,
                M.elem_col("Zn"): grade_pct * 10_000.0,
                M.GEOMET_DOMAIN: "VOLCANIC",
                M.GRADE_BIN: "HIGH",
                M.ALLOCATION_DOMAIN: "VOLCANIC-OXIDE-HIGH",
                M.bm_attr_col("rock_type"): "VOLCANIC",
                M.bm_attr_col("weathering"): "OXIDE",
                M.attr_col("rock_type"): "VOLCANIC",
                M.attr_col("weathering"): "OXIDE",
                M.PERIOD: "1",
                M.DOMAIN_MATCH: True,
                M.IS_PLANNED: False,
                M.AVAILABILITY: Availability.AVAILABLE.value,
                M.OUTSIDE_MODEL: False,
                "usable": True,
                M.X_MID: 0.0,
                M.Y_MID: 0.0,
                M.Z_MID: -i * length,
            }
        )
    return pd.DataFrame(rows)


def test_every_candidate_satisfies_the_length_and_mass_constraints(cfg):
    out, _ = generate_candidates(build_run(), cfg)
    assert not out.empty
    assert (out[M.LENGTH_M] >= cfg.compositing.min_length_m - 1e-9).all()
    assert (out[M.LENGTH_M] <= cfg.compositing.max_length_m + 1e-9).all()
    assert (out[EST_MASS_KG] >= cfg.compositing.min_mass_kg - 1e-9).all()


def test_a_run_too_light_to_reach_min_mass_yields_no_candidate(cfg):
    """Length alone is not enough: the core has to weigh what the test needs."""
    short = build_run(n=8)  # 8 m of HQ half core is about 31 kg, below min_mass_kg
    out, _ = generate_candidates(short, cfg)
    assert out.empty


def test_candidates_record_their_constituent_intervals(cfg):
    out, _ = generate_candidates(build_run(), cfg)
    for row in out.itertuples(index=False):
        members = list(getattr(row, INTERVAL_IDS))
        assert len(members) == row.n_intervals
        assert len(set(members)) == len(members)


def test_grade_bin_comes_from_the_length_weighted_mean(cfg):
    """A composite is binned on what it actually assays, not on its first interval."""
    run = build_run(n=20, grade_pct=1.0)
    run.loc[:2, M.elem_col("Zn")] = 60_000.0  # a few high intervals at the top
    run[M.GRADE_BIN] = "LOW"
    out, _ = generate_candidates(run, cfg)
    assert set(out[M.GRADE_BIN]) == {"LOW"}
    assert (out[M.ALLOCATION_DOMAIN].str.endswith("LOW")).all()


def test_high_variability_composites_are_rejected(cfg):
    run = build_run(n=30, grade_pct=6.0)
    run[M.elem_col("Zn")] = [10.0 if i % 2 else 200_000.0 for i in range(30)]
    out, _ = generate_candidates(run, cfg)
    assert out.empty or (out[CV_PRIMARY] <= cfg.compositing.max_cv_primary).all()


def test_unavailable_core_supplies_no_candidates_and_is_recorded(cfg):
    run = build_run()
    run[M.AVAILABILITY] = Availability.UNAVAILABLE.value
    out, gaps = generate_candidates(run, cfg)
    assert out.empty
    assert any(gap.key == "unavailable_core" for gap in gaps)


def test_unusable_intervals_never_enter_a_candidate(cfg):
    run = build_run()
    run.loc[10:14, "usable"] = False
    out, _ = generate_candidates(run, cfg)
    excluded = set(run.loc[10:14, M.INTERVAL_ID])
    for row in out.itertuples(index=False):
        assert not excluded & set(getattr(row, INTERVAL_IDS))


def test_only_one_candidate_per_start_interval_survives(cfg):
    """Deduplication keeps the greedy loop tractable on long runs."""
    out, _ = generate_candidates(build_run(n=40), cfg)
    assert out[M.FROM_M].is_unique


def test_quality_prefers_mass_margin_and_penalises_a_domain_mismatch(cfg):
    base = {EST_MASS_KG: 50.0, CV_PRIMARY: 0.2, M.DOMAIN_MATCH: True}
    heavier = {**base, EST_MASS_KG: 60.0}
    noisier = {**base, CV_PRIMARY: 0.8}
    mismatched = {**base, M.DOMAIN_MATCH: False}
    assert quality(heavier, cfg) > quality(base, cfg)
    assert quality(noisier, cfg) < quality(base, cfg)
    assert quality(mismatched, cfg) < quality(base, cfg)


def test_quality_caps_the_credit_for_extra_mass(cfg):
    """A very heavy composite must not outweigh domain fit."""
    huge = {EST_MASS_KG: 5000.0, CV_PRIMARY: 0.2, M.DOMAIN_MATCH: True}
    assert quality(huge, cfg) <= cfg.selection.mass_margin_cap


# ------------------------------------------------------------------ pick list


def test_pick_list_expands_composites_back_to_physical_samples(cfg):
    run = build_run()
    candidates, _ = generate_candidates(run, cfg)
    selected = candidates.head(1).copy()
    selected.insert(0, "composite_id", ["GM-001"])

    picks = pick_list(selected, run)
    assert len(picks) == int(selected["n_intervals"].iloc[0])
    assert set(picks["composite_id"]) == {"GM-001"}
    assert picks[M.SAMPLE_ID].notna().all()
    assert picks[M.LENGTH_M].sum() == pytest.approx(float(selected[M.LENGTH_M].iloc[0]))


def test_pick_list_is_sorted_by_hole_then_depth(cfg):
    run = pd.concat([build_run(hole="H2"), build_run(hole="H1")], ignore_index=True)
    run[M.RUN_ID] = run[M.HOLE_ID]
    candidates, _ = generate_candidates(run, cfg)
    selected = candidates.groupby(M.HOLE_ID, as_index=False).head(1).copy()
    selected.insert(0, "composite_id", [f"GM-{i:03d}" for i in range(1, len(selected) + 1)])

    picks = pick_list(selected, run)
    assert picks[[M.HOLE_ID, M.FROM_M]].apply(tuple, axis=1).is_monotonic_increasing
