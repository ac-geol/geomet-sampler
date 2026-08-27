"""Interval framework: splitting at logged contacts, and traceability back to samples."""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.intervals.merge import build_framework, discretise_planned
from geomet_sampler.models import Severity


def assay(rows):
    return pd.DataFrame(rows)


def litho(rows):
    return pd.DataFrame(rows)


LOOKUP = pd.DataFrame(
    {
        M.LOGGED_CODE: ["VOLC", "SED", "INTR"],
        M.GEOMET_DOMAIN: ["VOLCANIC", "SEDIMENT", "INTRUSIVE"],
    }
)


def test_assay_interval_is_split_at_a_logged_contact():
    a = assay(
        [
            {
                M.HOLE_ID: "H1",
                M.SAMPLE_ID: "S1",
                M.FROM_M: 10.0,
                M.TO_M: 12.0,
                M.DENSITY: 2.7,
                M.elem_col("Zn"): 30000.0,
                M.LOW_RECOVERY: False,
            }
        ]
    )
    lith = litho(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 11.0, M.LOGGED_CODE: "VOLC"},
            {M.HOLE_ID: "H1", M.FROM_M: 11.0, M.TO_M: 20.0, M.LOGGED_CODE: "SED"},
        ]
    )
    out, issues = build_framework(a, lith, LOOKUP)

    assert len(out) == 2
    assert list(out[M.FROM_M]) == [10.0, 11.0]
    assert list(out[M.TO_M]) == [11.0, 12.0]
    assert list(out[M.GEOMET_DOMAIN]) == ["VOLCANIC", "SEDIMENT"]
    # grade is constant within the parent sample, so both halves inherit it unchanged
    assert list(out[M.elem_col("Zn")]) == [30000.0, 30000.0]
    assert list(out[M.PARENT_SAMPLE_ID]) == ["S1", "S1"]
    assert out[M.INTERVAL_ID].is_unique
    assert not [i for i in issues if i.severity is Severity.ERROR]


def test_unsplit_interval_keeps_the_physical_sample_id_unsuffixed():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 1.0, M.DENSITY: 2.7}])
    lith = litho([{M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 5.0, M.LOGGED_CODE: "VOLC"}])
    out, _ = build_framework(a, lith, LOOKUP)
    assert list(out[M.INTERVAL_ID]) == ["S1"]
    assert list(out[M.PARENT_SAMPLE_ID]) == ["S1"]


def test_lengths_are_preserved_by_splitting():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 10.0, M.DENSITY: 2.7}])
    lith = litho(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 3.0, M.LOGGED_CODE: "VOLC"},
            {M.HOLE_ID: "H1", M.FROM_M: 3.0, M.TO_M: 7.0, M.LOGGED_CODE: "SED"},
            {M.HOLE_ID: "H1", M.FROM_M: 7.0, M.TO_M: 10.0, M.LOGGED_CODE: "INTR"},
        ]
    )
    out, _ = build_framework(a, lith, LOOKUP)
    assert len(out) == 3
    assert out[M.LENGTH_M].sum() == pytest.approx(10.0)


def test_coincident_boundary_does_not_create_a_zero_length_sliver():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 5.0, M.DENSITY: 2.7}])
    lith = litho(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 5.0, M.LOGGED_CODE: "VOLC"},
            {M.HOLE_ID: "H1", M.FROM_M: 5.0, M.TO_M: 9.0, M.LOGGED_CODE: "SED"},
        ]
    )
    out, _ = build_framework(a, lith, LOOKUP)
    assert len(out) == 1
    assert out[M.LENGTH_M].iloc[0] == pytest.approx(5.0)


def test_unmapped_logged_code_is_an_error_not_a_silent_drop():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 2.0, M.DENSITY: 2.7}])
    lith = litho([{M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 5.0, M.LOGGED_CODE: "MYSTERY"}])
    out, issues = build_framework(a, lith, LOOKUP)
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert [i.check for i in errors] == ["logged_code_unmapped"]
    assert "MYSTERY" in errors[0].message
    assert len(out) == 1  # kept, so the user can see it in the report


def test_assay_beyond_logged_extent_is_reported_and_carries_no_domain():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 50.0, M.TO_M: 51.0, M.DENSITY: 2.7}])
    lith = litho([{M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 10.0, M.LOGGED_CODE: "VOLC"}])
    out, issues = build_framework(a, lith, LOOKUP)
    assert out[M.GEOMET_DOMAIN].isna().all()
    assert "interval_not_logged" in [i.check for i in issues]


def test_hole_without_lithology_is_reported():
    a = assay([{M.HOLE_ID: "H9", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 2.0, M.DENSITY: 2.7}])
    lith = litho([{M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 10.0, M.LOGGED_CODE: "VOLC"}])
    out, issues = build_framework(a, lith, LOOKUP)
    assert "no_lithology" in [i.check for i in issues]
    assert len(out) == 1


def test_drillhole_attribute_logged_against_lithology_is_carried_through():
    a = assay([{M.HOLE_ID: "H1", M.SAMPLE_ID: "S1", M.FROM_M: 0.0, M.TO_M: 4.0, M.DENSITY: 2.7}])
    lith = litho(
        [
            {
                M.HOLE_ID: "H1",
                M.FROM_M: 0.0,
                M.TO_M: 2.0,
                M.LOGGED_CODE: "VOLC",
                M.dh_attr_col("weathering"): "OXIDE",
            },
            {
                M.HOLE_ID: "H1",
                M.FROM_M: 2.0,
                M.TO_M: 8.0,
                M.LOGGED_CODE: "VOLC",
                M.dh_attr_col("weathering"): "FRESH",
            },
        ]
    )
    out, _ = build_framework(a, lith, LOOKUP)
    assert list(out[M.dh_attr_col("weathering")]) == ["OXIDE", "FRESH"]


def test_planned_holes_are_discretised_at_the_configured_step():
    class FakeTrace:
        total_depth = 10.0

    out = discretise_planned({"P1": FakeTrace()}, ["P1"], 2.5)
    assert len(out) == 4
    assert out[M.IS_PLANNED].all()
    assert out[M.LENGTH_M].sum() == pytest.approx(10.0)
