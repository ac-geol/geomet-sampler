"""Allocation targets: tonnage splits, rounding, floors and existing coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.allocate.targets import (
    DEFICIT,
    EXISTING,
    TARGET,
    TONNES_PCT,
    compute_targets,
    domain_tonnage,
    existing_coverage,
    largest_remainder,
    unmet_gaps,
)


def blocks(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame[M.BLOCK_TONNES] = frame["volume"] * frame[M.DENSITY]
    return frame


def synthetic_model(split: dict[str, int], *, period: str = "1") -> pd.DataFrame:
    """One block per unit of the requested split, so tonnage shares are exact."""
    rows = []
    for domain, count in split.items():
        for _ in range(count):
            rows.append(
                {
                    M.ALLOCATION_DOMAIN: domain,
                    M.PERIOD: period,
                    "volume": 1000.0,
                    M.DENSITY: 2.7,
                    M.bm_elem_col("Zn"): 10_000.0,
                }
            )
    return blocks(rows)


def test_scheduled_tonnage_reproduces_a_known_split(cfg):
    model = synthetic_model({"A": 60, "B": 30, "C": 10})
    table = domain_tonnage(model, cfg)
    shares = dict(zip(table[M.ALLOCATION_DOMAIN], table[TONNES_PCT], strict=True))
    assert shares["A"] == pytest.approx(60.0)
    assert shares["B"] == pytest.approx(30.0)
    assert shares["C"] == pytest.approx(10.0)


def test_blocks_without_a_period_are_not_scheduled(cfg):
    model = pd.concat(
        [synthetic_model({"A": 50}), synthetic_model({"B": 50}, period="")], ignore_index=True
    )
    table = domain_tonnage(model, cfg)
    assert list(table[M.ALLOCATION_DOMAIN]) == ["A"]


def test_period_weights_shift_the_target_toward_early_production(cfg):
    early = synthetic_model({"A": 50}, period="1")
    late = synthetic_model({"B": 50}, period="9")
    table = domain_tonnage(pd.concat([early, late], ignore_index=True), cfg)
    weighted = dict(zip(table[M.ALLOCATION_DOMAIN], table["weighted_pct"], strict=True))
    assert weighted["A"] > 50.0 > weighted["B"]


def test_proportional_allocation_reproduces_the_tonnage_split(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 100
    cfg.allocation.min_samples_per_domain = 0
    table, _ = compute_targets(synthetic_model({"A": 60, "B": 30, "C": 10}), cfg)
    targets = dict(zip(table[M.ALLOCATION_DOMAIN], table[TARGET], strict=True))
    assert targets["A"] == pytest.approx(60.0)
    assert targets["B"] == pytest.approx(30.0)
    assert targets["C"] == pytest.approx(10.0)


def test_rounded_deficits_sum_exactly_to_total_samples(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.min_samples_per_domain = 0
    for total in (7, 13, 40, 41):
        cfg.allocation.total_samples = total
        table, _ = compute_targets(synthetic_model({"A": 33, "B": 33, "C": 34}), cfg)
        assert int(table[DEFICIT].sum()) == total


def test_largest_remainder_gives_the_extra_sample_to_the_largest_fraction():
    values = pd.Series([1.7, 1.2, 1.1])
    assert list(largest_remainder(values, 5)) == [2, 2, 1]
    assert largest_remainder(values, 5).sum() == 5


def test_a_material_domain_receives_the_minimum_number_of_samples(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 100
    cfg.allocation.min_samples_per_domain = 3
    cfg.allocation.min_domain_tonnage_pct = 1.0
    table, _ = compute_targets(synthetic_model({"BULK": 970, "MINOR": 30}), cfg)
    minor = table.set_index(M.ALLOCATION_DOMAIN).loc["MINOR"]
    assert minor[TARGET] >= 3


def test_a_domain_below_the_tonnage_floor_gets_no_minimum(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 10
    cfg.allocation.min_samples_per_domain = 3
    cfg.allocation.min_domain_tonnage_pct = 5.0
    table, _ = compute_targets(synthetic_model({"BULK": 999, "TRACE": 1}), cfg)
    trace = table.set_index(M.ALLOCATION_DOMAIN).loc["TRACE"]
    assert trace[TARGET] < 3


def test_neyman_gives_more_samples_to_the_more_variable_domain(cfg):
    """A variable domain needs more samples per tonne than a homogeneous one.

    The homogeneous domain is not rescued here by a fabricated variance: coverage of a
    genuinely uniform domain is the job of min_samples_per_domain, which is switched off
    in this test so the Neyman weighting itself is what is being measured.
    """
    steady = synthetic_model({"STEADY": 50})
    variable = synthetic_model({"VARIABLE": 50})
    variable[M.bm_elem_col("Zn")] = [1_000.0 if i % 2 else 90_000.0 for i in range(len(variable))]
    cfg.allocation.method = "neyman"
    cfg.allocation.total_samples = 20
    cfg.allocation.min_samples_per_domain = 0
    table, _ = compute_targets(pd.concat([steady, variable], ignore_index=True), cfg)
    targets = dict(zip(table[M.ALLOCATION_DOMAIN], table[TARGET], strict=True))
    assert targets["VARIABLE"] > targets["STEADY"]


def test_risk_multiplier_raises_a_domains_target(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 20
    cfg.allocation.min_samples_per_domain = 0
    model = synthetic_model({"A": 50, "B": 50})
    plain, _ = compute_targets(model, cfg)
    cfg.allocation.risk_multipliers = {"A": 3.0}
    risky, _ = compute_targets(model, cfg)
    assert (
        risky.set_index(M.ALLOCATION_DOMAIN).loc["A", TARGET]
        > plain.set_index(M.ALLOCATION_DOMAIN).loc["A", TARGET]
    )


def test_existing_testwork_reduces_the_deficit(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 10
    cfg.allocation.min_samples_per_domain = 0
    model = synthetic_model({"A": 50, "B": 50})
    existing = pd.Series({"A": 4.0})
    table, _ = compute_targets(model, cfg, existing)
    by_domain = table.set_index(M.ALLOCATION_DOMAIN)
    assert by_domain.loc["A", EXISTING] == 4.0
    assert by_domain.loc["A", DEFICIT] < by_domain.loc["B", DEFICIT]


def test_existing_testwork_domain_is_derived_from_the_recorded_depths():
    intervals = pd.DataFrame(
        {
            M.HOLE_ID: ["H1", "H1"],
            M.FROM_M: [0.0, 10.0],
            M.TO_M: [10.0, 20.0],
            M.ALLOCATION_DOMAIN: ["A", "B"],
        }
    )
    existing = pd.DataFrame(
        {
            M.HOLE_ID: ["H1"],
            M.FROM_M: [1.0],
            M.TO_M: [9.0],
            M.GEOMET_DOMAIN: [None],
        }
    )
    counts, issues = existing_coverage(existing, intervals)
    assert counts.to_dict() == {"A": 1.0}
    assert issues == []


def test_unmatchable_existing_testwork_is_reported_not_ignored():
    intervals = pd.DataFrame(
        {M.HOLE_ID: ["H1"], M.FROM_M: [0.0], M.TO_M: [10.0], M.ALLOCATION_DOMAIN: ["A"]}
    )
    existing = pd.DataFrame(
        {M.HOLE_ID: ["NOPE"], M.FROM_M: [1.0], M.TO_M: [2.0], M.GEOMET_DOMAIN: [None]}
    )
    counts, issues = existing_coverage(existing, intervals)
    assert counts.empty
    assert [i.check for i in issues] == ["existing_testwork_undomained"]


def test_unfilled_deficits_become_gap_register_entries(cfg):
    cfg.allocation.method = "proportional"
    cfg.allocation.total_samples = 10
    table, _ = compute_targets(synthetic_model({"A": 50, "B": 50}), cfg)
    gaps = unmet_gaps(table, pd.Series({"A": 5.0}))
    assert [g.key for g in gaps] == ["B"]
    assert gaps[0].detail["shortfall"] > 0
