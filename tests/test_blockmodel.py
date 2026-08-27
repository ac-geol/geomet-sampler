"""Block model assignment, domain matching and the shared domain vocabulary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.blockmodel.assign import (
    assign_blocks,
    off_model_fraction,
    record_domain_match,
    resolve_attributes,
)
from geomet_sampler.domains import (
    UNKNOWN,
    allocation_domain,
    assign_grade_bin,
    compose_domain,
    is_scheduled,
    label_blocks,
    label_intervals,
    period_sort_key,
)

ROLES = ["rock_type", "weathering"]


def one_block(**overrides) -> pd.DataFrame:
    row = {
        M.BLOCK_X: 100.0,
        M.BLOCK_Y: 200.0,
        M.BLOCK_Z: 300.0,
        M.BLOCK_DX: 20.0,
        M.BLOCK_DY: 20.0,
        M.BLOCK_DZ: 10.0,
        M.PERIOD: "2",
        M.DENSITY: 2.7,
        M.bm_attr_col("rock_type"): "VOLCANIC",
        M.bm_attr_col("weathering"): "OXIDE",
        M.bm_elem_col("Zn"): 30_000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def interval_at(x: float, y: float, z: float, **overrides) -> pd.DataFrame:
    row = {M.HOLE_ID: "H1", M.X_MID: x, M.Y_MID: y, M.Z_MID: z}
    row.update(overrides)
    return pd.DataFrame([row])


def test_an_interval_inside_a_block_takes_its_period_and_attributes():
    out, issues = assign_blocks(
        interval_at(105.0, 205.0, 302.0), one_block(), roles=ROLES, elements=["Zn"]
    )
    assert not out[M.OUTSIDE_MODEL].iloc[0]
    assert out[M.PERIOD].iloc[0] == "2"
    assert out[M.bm_attr_col("rock_type")].iloc[0] == "VOLCANIC"
    assert out[M.bm_elem_col("Zn")].iloc[0] == 30_000.0
    assert not [i for i in issues if i.check == "outside_model"]


def test_the_nearest_block_is_not_enough_without_containment():
    """Nearest-centroid alone would attach a block hundreds of metres away."""
    out, issues = assign_blocks(
        interval_at(1000.0, 200.0, 300.0), one_block(), roles=ROLES, elements=["Zn"]
    )
    assert out[M.OUTSIDE_MODEL].iloc[0]
    assert pd.isna(out[M.PERIOD].iloc[0])
    assert [i.check for i in issues] == ["outside_model"]


def test_containment_uses_each_blocks_own_dimensions():
    block = one_block()
    inside_dz = interval_at(100.0, 200.0, 304.9)  # within dz/2 = 5
    outside_dz = interval_at(100.0, 200.0, 306.0)
    assert not assign_blocks(inside_dz, block, roles=ROLES, elements=[])[0][M.OUTSIDE_MODEL].iloc[0]
    assert assign_blocks(outside_dz, block, roles=ROLES, elements=[])[0][M.OUTSIDE_MODEL].iloc[0]


def test_an_interval_with_no_desurveyed_midpoint_is_left_outside_the_model():
    out, issues = assign_blocks(
        interval_at(np.nan, np.nan, np.nan), one_block(), roles=ROLES, elements=[]
    )
    assert out[M.OUTSIDE_MODEL].all()
    assert "no_desurveyed_midpoints" in [i.check for i in issues]


def test_an_empty_block_model_leaves_everything_outside_and_says_so():
    out, issues = assign_blocks(
        interval_at(1.0, 1.0, 1.0), one_block().iloc[0:0], roles=ROLES, elements=[]
    )
    assert out[M.OUTSIDE_MODEL].all()
    assert "block_model_empty" in [i.check for i in issues]


def test_off_model_fraction_reports_the_share_outside():
    frame = pd.DataFrame({M.OUTSIDE_MODEL: [True, False, False, False]})
    assert off_model_fraction(frame) == pytest.approx(0.25)
    assert off_model_fraction(pd.DataFrame()) == 0.0


# ----------------------------------------------------------- domain matching


def test_a_role_logged_on_both_sides_is_compared():
    frame = pd.DataFrame(
        {
            M.dh_attr_col("rock_type"): ["VOLCANIC", "SEDIMENT"],
            M.bm_attr_col("rock_type"): ["volcanic", "VOLCANIC"],
        }
    )
    out = record_domain_match(frame, ["rock_type"])
    assert list(out[M.DOMAIN_MATCH]) == [True, False]


def test_nothing_comparable_leaves_domain_match_null_rather_than_true():
    """An unlogged dataset must not score as a perfect match."""
    frame = pd.DataFrame({M.bm_attr_col("rock_type"): ["VOLCANIC"]})
    out = record_domain_match(frame, ["rock_type"])
    assert out[M.DOMAIN_MATCH].isna().all()


def test_the_logged_domain_is_compared_only_against_a_role_the_user_names():
    frame = pd.DataFrame({M.GEOMET_DOMAIN: ["VOLCANIC"], M.bm_attr_col("rock_type"): ["SEDIMENT"]})
    assert record_domain_match(frame, ["rock_type"])[M.DOMAIN_MATCH].isna().all()
    named = record_domain_match(frame, ["rock_type"], domain_match_attribute="rock_type")
    assert not bool(named[M.DOMAIN_MATCH].iloc[0])


def test_a_missing_value_on_either_side_is_not_counted_as_disagreement():
    frame = pd.DataFrame(
        {M.dh_attr_col("rock_type"): [None], M.bm_attr_col("rock_type"): ["VOLCANIC"]}
    )
    assert record_domain_match(frame, ["rock_type"])[M.DOMAIN_MATCH].isna().all()


# ------------------------------------------------------- attribute resolution


def test_logged_geology_wins_over_the_model_where_it_exists():
    """Compositing boundaries must follow the material actually in the tray."""
    frame = pd.DataFrame(
        {
            M.dh_attr_col("rock_type"): ["LOGGED", None],
            M.bm_attr_col("rock_type"): ["MODELLED", "MODELLED"],
        }
    )
    out = resolve_attributes(frame, ["rock_type"])
    assert list(out[M.attr_col("rock_type")]) == ["LOGGED", "MODELLED"]


def test_a_role_on_only_one_side_resolves_from_that_side():
    modelled = resolve_attributes(pd.DataFrame({M.bm_attr_col("w"): ["OXIDE"]}), ["w"])
    logged = resolve_attributes(pd.DataFrame({M.dh_attr_col("w"): ["FRESH"]}), ["w"])
    assert modelled[M.attr_col("w")].iloc[0] == "OXIDE"
    assert logged[M.attr_col("w")].iloc[0] == "FRESH"


def test_a_role_on_neither_side_resolves_to_null():
    out = resolve_attributes(pd.DataFrame({"other": [1]}), ["w"])
    assert out[M.attr_col("w")].isna().all()


# ------------------------------------------------------------- domain labels


def test_grade_bins_are_applied_in_the_elements_declared_units(cfg):
    """Edges are written in percent; the data is held in ppm."""
    frame = pd.DataFrame({M.elem_col("Zn"): [10_000.0, 30_000.0]})
    assert list(assign_grade_bin(frame, cfg, on="drillhole")) == ["LOW", "HIGH"]


def test_a_grade_column_that_is_absent_bins_to_null(cfg):
    out = assign_grade_bin(pd.DataFrame({"other": [1.0]}), cfg, on="drillhole")
    assert out.isna().all()


def test_blocks_and_intervals_are_labelled_by_the_same_rule(cfg):
    blocks = label_blocks(one_block().assign(**{M.GRADE_BIN: None}), cfg)
    intervals = label_intervals(
        pd.DataFrame(
            [
                {
                    M.elem_col("Zn"): 30_000.0,
                    M.bm_attr_col("rock_type"): "VOLCANIC",
                    M.bm_attr_col("weathering"): "OXIDE",
                }
            ]
        ),
        cfg,
    )
    assert blocks[M.ALLOCATION_DOMAIN].iloc[0] == intervals[M.ALLOCATION_DOMAIN].iloc[0]


def test_a_missing_allocation_component_becomes_unknown_not_a_crash(cfg):
    out = allocation_domain(pd.DataFrame({"nothing": [1]}), cfg, on="block_model")
    assert out.iloc[0] == "UNKNOWN-UNKNOWN-UNKNOWN"


def test_compose_domain_uppercases_and_marks_missing_parts():
    assert compose_domain(["volcanic", "oxide", "high"]) == "VOLCANIC-OXIDE-HIGH"
    assert compose_domain(["volcanic", None, "high"]) == f"VOLCANIC-{UNKNOWN}-HIGH"
    assert compose_domain([np.nan]) == UNKNOWN


def test_scheduling_excludes_blanks_and_configured_periods():
    period = pd.Series(["1", "", None, "nan", "WASTE"])
    assert list(is_scheduled(period, ["WASTE"])) == [True, False, False, False, False]


def test_periods_sort_numerically_where_they_can():
    assert sorted(["10", "2", "1"], key=period_sort_key) == ["1", "2", "10"]
    assert sorted(["Q1", "2"], key=period_sort_key) == ["2", "Q1"]
    assert period_sort_key(None)[0] == 2
