"""Selection: determinism, constraints, and finding the obvious answer."""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.allocate.targets import DEFICIT
from geomet_sampler.composite.candidates import CV_PRIMARY, EST_MASS_KG, INTERVAL_IDS, QUALITY
from geomet_sampler.select.greedy import COMPOSITE_ID, REASON, check_no_interval_reused, select
from geomet_sampler.select.objective import (
    allocation_fit,
    domain_match_rate,
    mean_min_separation,
    objective,
)


def candidate(
    cid: str,
    domain: str,
    *,
    hole: str = "H1",
    from_m: float = 0.0,
    length: float = 10.0,
    mass: float = 55.0,
    cv: float = 0.2,
    period: str = "1",
    quality: float = 1.0,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    planned: bool = False,
    match: object = True,
) -> dict:
    return {
        "candidate_id": cid,
        M.HOLE_ID: hole,
        M.RUN_ID: f"R_{hole}",
        M.FROM_M: from_m,
        M.TO_M: from_m + length,
        M.LENGTH_M: length,
        "n_intervals": 5,
        EST_MASS_KG: mass,
        M.GEOMET_DOMAIN: domain,
        M.PERIOD: period,
        M.GRADE_BIN: "HIGH",
        M.ALLOCATION_DOMAIN: domain,
        M.DOMAIN_MATCH: match,
        M.IS_PLANNED: planned,
        M.AVAILABILITY: "AVAILABLE",
        CV_PRIMARY: cv,
        INTERVAL_IDS: tuple(f"{cid}_i{k}" for k in range(5)),
        M.X_MID: xyz[0],
        M.Y_MID: xyz[1],
        M.Z_MID: xyz[2],
        QUALITY: quality,
    }


def allocation(targets: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {M.ALLOCATION_DOMAIN: list(targets), DEFICIT: [float(v) for v in targets.values()]}
    )


def test_greedy_finds_the_obvious_answer(cfg):
    """One deficit per domain, one clearly better candidate in each."""
    cfg.selection.min_separation_m = 0.0
    cfg.selection.swap_iterations = 0
    frame = pd.DataFrame(
        [
            candidate("best_A", "A", hole="H1", quality=1.0),
            candidate("poor_A", "A", hole="H2", quality=0.2),
            candidate("best_B", "B", hole="H3", quality=1.0),
            candidate("poor_B", "B", hole="H4", quality=0.2),
        ]
    )
    selected, _, issues = select(frame, allocation({"A": 1, "B": 1}), cfg)
    assert set(selected["candidate_id"]) == {"best_A", "best_B"}
    assert not [i for i in issues if i.severity.value == "ERROR"]


def test_selection_is_deterministic_under_a_fixed_seed(cfg):
    frame = pd.DataFrame(
        [
            candidate(f"c{i}", "A", hole=f"H{i}", quality=1.0, xyz=(i * 100.0, 0.0, 0.0))
            for i in range(10)
        ]
    )
    runs = [select(frame.copy(), allocation({"A": 4}), cfg)[0] for _ in range(3)]
    first = list(runs[0]["candidate_id"])
    assert all(list(r["candidate_id"]) == first for r in runs)


def test_ties_break_on_the_lower_cv(cfg):
    cfg.selection.swap_iterations = 0
    frame = pd.DataFrame(
        [
            candidate("noisy", "A", hole="H1", cv=0.9),
            candidate("clean", "A", hole="H2", cv=0.1),
        ]
    )
    selected, _, _ = select(frame, allocation({"A": 1}), cfg)
    assert list(selected["candidate_id"]) == ["clean"]


def test_period_weighting_prefers_earlier_production(cfg):
    cfg.selection.swap_iterations = 0
    frame = pd.DataFrame(
        [
            candidate("late", "A", hole="H1", period="9"),
            candidate("early", "A", hole="H2", period="1"),
        ]
    )
    selected, _, _ = select(frame, allocation({"A": 1}), cfg)
    assert list(selected["candidate_id"]) == ["early"]


def test_a_hole_never_exceeds_max_samples_per_hole(cfg):
    cfg.selection.max_samples_per_hole = 2
    cfg.selection.min_separation_m = 0.0
    frame = pd.DataFrame([candidate(f"c{i}", "A", hole="H1", from_m=i * 20.0) for i in range(6)])
    selected, _, issues = select(frame, allocation({"A": 5}), cfg)
    assert len(selected) == 2
    assert "selection_exhausted" in [i.check for i in issues]


def test_minimum_separation_is_respected(cfg):
    cfg.selection.min_separation_m = 50.0
    cfg.selection.max_samples_per_hole = 5
    frame = pd.DataFrame(
        [
            candidate("near1", "A", hole="H1", from_m=0.0, xyz=(0.0, 0.0, 0.0)),
            candidate("near2", "A", hole="H2", from_m=0.0, xyz=(10.0, 0.0, 0.0)),
            candidate("far", "A", hole="H3", from_m=0.0, xyz=(500.0, 0.0, 0.0)),
        ]
    )
    selected, _, _ = select(frame, allocation({"A": 3}), cfg)
    points = selected[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            assert ((points[i] - points[j]) ** 2).sum() ** 0.5 >= 50.0


def test_overlapping_composites_in_one_hole_are_never_both_selected(cfg):
    cfg.selection.max_samples_per_hole = 5
    cfg.selection.min_separation_m = 0.0
    frame = pd.DataFrame(
        [
            candidate("a", "A", hole="H1", from_m=0.0, length=20.0),
            candidate("b", "A", hole="H1", from_m=10.0, length=20.0),
            candidate("c", "A", hole="H1", from_m=40.0, length=20.0),
        ]
    )
    selected, _, _ = select(frame, allocation({"A": 3}), cfg)
    spans = sorted(zip(selected[M.FROM_M], selected[M.TO_M], strict=True))
    assert len(spans) == 2  # the middle candidate overlaps whichever is taken first
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert start >= end


def test_planned_holes_are_down_weighted_against_real_core(cfg):
    cfg.selection.swap_iterations = 0
    frame = pd.DataFrame(
        [
            candidate("planned", "A", hole="P1", planned=True),
            candidate("real", "A", hole="H1"),
        ]
    )
    selected, _, _ = select(frame, allocation({"A": 1}), cfg)
    assert list(selected["candidate_id"]) == ["real"]


def test_a_domain_with_no_deficit_is_never_sampled(cfg):
    frame = pd.DataFrame([candidate("a", "A", hole="H1"), candidate("b", "B", hole="H2")])
    selected, _, _ = select(frame, allocation({"A": 1, "B": 0}), cfg)
    assert set(selected[M.ALLOCATION_DOMAIN]) == {"A"}


def test_every_selection_records_a_readable_reason(cfg):
    frame = pd.DataFrame([candidate("a", "A", hole="H1", mass=52.1, cv=0.41, period="2")])
    selected, _, _ = select(frame, allocation({"A": 1}), cfg)
    reason = selected[REASON].iloc[0]
    assert "fills A deficit" in reason
    assert "52.1 kg" in reason
    assert "CV 0.41" in reason


def test_a_domain_mismatch_is_called_out_in_the_reason(cfg):
    frame = pd.DataFrame([candidate("a", "A", hole="H1", match=False)])
    selected, _, _ = select(frame, allocation({"A": 1}), cfg)
    assert "disagrees" in selected[REASON].iloc[0]


def test_no_interval_appears_in_two_composites(cfg):
    cfg.selection.max_samples_per_hole = 5
    cfg.selection.min_separation_m = 0.0
    frame = pd.DataFrame([candidate(f"c{i}", "A", hole="H1", from_m=i * 30.0) for i in range(4)])
    selected, _, _ = select(frame, allocation({"A": 4}), cfg)
    assert check_no_interval_reused(selected) == []


def test_an_empty_candidate_table_is_an_error_not_a_crash(cfg):
    empty = pd.DataFrame(
        columns=[
            "candidate_id",
            M.HOLE_ID,
            M.FROM_M,
            M.TO_M,
            M.ALLOCATION_DOMAIN,
            EST_MASS_KG,
            CV_PRIMARY,
            QUALITY,
            INTERVAL_IDS,
        ]
    )
    selected, _, issues = select(empty, allocation({"A": 2}), cfg)
    assert selected.empty
    assert "no_candidates" in [i.check for i in issues]


# ------------------------------------------------------------------ objective


def test_allocation_fit_counts_shortfall_and_overshoot():
    selected = pd.DataFrame({M.ALLOCATION_DOMAIN: ["A", "A", "A"]})
    assert allocation_fit(selected, allocation({"A": 1})) == pytest.approx(2.0)
    assert allocation_fit(selected, allocation({"A": 3})) == pytest.approx(0.0)


def test_mean_min_separation_is_the_mean_nearest_neighbour_distance():
    selected = pd.DataFrame({M.X_MID: [0.0, 30.0], M.Y_MID: [0.0, 40.0], M.Z_MID: [0.0, 0.0]})
    assert mean_min_separation(selected) == pytest.approx(50.0)
    assert mean_min_separation(selected.head(1)) == 0.0


def test_domain_match_rate_ignores_composites_that_cannot_be_compared():
    selected = pd.DataFrame({M.DOMAIN_MATCH: [True, False, pd.NA]})
    assert domain_match_rate(selected) == pytest.approx(0.5)


def test_objective_prefers_the_programme_that_fits_the_allocation(cfg):
    good = pd.DataFrame(
        [candidate("a", "A", xyz=(0.0, 0.0, 0.0)), candidate("b", "B", xyz=(100.0, 0.0, 0.0))]
    )
    bad = pd.DataFrame(
        [candidate("a", "A", xyz=(0.0, 0.0, 0.0)), candidate("c", "A", xyz=(100.0, 0.0, 0.0))]
    )
    targets = allocation({"A": 1, "B": 1})
    assert objective(good, targets, cfg) > objective(bad, targets, cfg)


def test_composite_ids_are_assigned_in_a_stable_order(cfg):
    frame = pd.DataFrame(
        [candidate(f"c{i}", "A", hole=f"H{i}", xyz=(i * 100.0, 0, 0)) for i in range(4)]
    )
    selected, _, _ = select(frame, allocation({"A": 3}), cfg)
    assert list(selected[COMPOSITE_ID]) == ["GM-001", "GM-002", "GM-003"]
