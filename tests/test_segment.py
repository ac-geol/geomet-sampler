"""Segmentation, and the correctness guarantee that matters most.

A composite that spans a lithology contact, a weathering front, a core gap or a hole
boundary is wrong in a way nobody notices until the testwork comes back. The guarantee
is structural: composites are generated only within runs. These tests assert the
property directly against generated candidates, not just against the run table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.composite.candidates import INTERVAL_IDS, generate_candidates
from geomet_sampler.segment.runs import segment_runs, short_run_gaps, summarise_runs


def intervals(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame[M.LENGTH_M] = frame[M.TO_M] - frame[M.FROM_M]
    for column, default in (
        (M.INTERVAL_ID, None),
        (M.LOW_RECOVERY, False),
        (M.OUTSIDE_MODEL, False),
        (M.elem_col("Zn"), 1.0),
    ):
        if column not in frame.columns:
            frame[column] = default
    if frame[M.INTERVAL_ID].isna().all():
        frame[M.INTERVAL_ID] = [f"I{i}" for i in range(len(frame))]
    return frame


def run_segment(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    options = {
        "hard_break_on": [M.GEOMET_DOMAIN],
        "max_interval_gap_m": 0.1,
        "break_on_low_recovery": True,
        "break_on_missing_primary_grade": True,
        "primary_element": "Zn",
    }
    options.update(kwargs)
    return segment_runs(frame, **options)


# ------------------------------------------------------------- break reasons


def test_domain_change_starts_a_new_run():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 1, M.TO_M: 2, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 2, M.TO_M: 3, M.GEOMET_DOMAIN: "B"},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 2
    assert out["break_reason"].iloc[2] == "attribute:geomet_domain"


def test_hole_change_starts_a_new_run():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H2", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 2
    assert out["break_reason"].iloc[1] == "hole"


def test_gap_beyond_tolerance_starts_a_new_run_but_a_small_gap_does_not():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0.0, M.TO_M: 1.0, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 1.05, M.TO_M: 2.0, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 5.0, M.TO_M: 6.0, M.GEOMET_DOMAIN: "A"},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 2
    assert out["break_reason"].iloc[2] == "gap"


def test_low_recovery_interval_is_isolated():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 1, M.TO_M: 2, M.GEOMET_DOMAIN: "A", M.LOW_RECOVERY: True},
            {M.HOLE_ID: "H1", M.FROM_M: 2, M.TO_M: 3, M.GEOMET_DOMAIN: "A"},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 3
    assert list(out["usable"]) == [True, False, True]


def test_missing_primary_grade_breaks_the_run():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A", M.elem_col("Zn"): 1.0},
            {
                M.HOLE_ID: "H1",
                M.FROM_M: 1,
                M.TO_M: 2,
                M.GEOMET_DOMAIN: "A",
                M.elem_col("Zn"): np.nan,
            },
            {M.HOLE_ID: "H1", M.FROM_M: 2, M.TO_M: 3, M.GEOMET_DOMAIN: "A", M.elem_col("Zn"): 1.0},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 3
    assert not out["usable"].iloc[1]


def test_interval_outside_the_block_model_breaks_the_run():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 1, M.TO_M: 2, M.GEOMET_DOMAIN: "A", M.OUTSIDE_MODEL: True},
            {M.HOLE_ID: "H1", M.FROM_M: 2, M.TO_M: 3, M.GEOMET_DOMAIN: "A"},
        ]
    )
    out = run_segment(frame)
    assert out[M.RUN_ID].nunique() == 3


def test_a_user_invented_attribute_role_breaks_runs_without_code_changes():
    """Nothing enumerates roles, so an arbitrary one must work unchanged."""
    frame = intervals(
        [
            {
                M.HOLE_ID: "H1",
                M.FROM_M: 0,
                M.TO_M: 1,
                M.GEOMET_DOMAIN: "A",
                M.attr_col("alteration"): "SERICITE",
            },
            {
                M.HOLE_ID: "H1",
                M.FROM_M: 1,
                M.TO_M: 2,
                M.GEOMET_DOMAIN: "A",
                M.attr_col("alteration"): "CHLORITE",
            },
        ]
    )
    out = run_segment(frame, hard_break_on=[M.GEOMET_DOMAIN, "alteration"])
    assert out[M.RUN_ID].nunique() == 2
    assert out["break_reason"].iloc[1] == "attribute:alteration"


def test_run_summary_reports_extent_and_length():
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 2, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 2, M.TO_M: 5, M.GEOMET_DOMAIN: "A"},
        ]
    )
    summary = summarise_runs(run_segment(frame), [M.GEOMET_DOMAIN])
    assert len(summary) == 1
    assert summary[M.LENGTH_M].iloc[0] == pytest.approx(5.0)
    assert summary["n_intervals"].iloc[0] == 2


def test_short_runs_go_to_the_gap_register_rather_than_vanishing():
    frame = intervals([{M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 2, M.GEOMET_DOMAIN: "A"}])
    summary = summarise_runs(run_segment(frame), [M.GEOMET_DOMAIN])
    gaps = short_run_gaps(summary, min_length_m=6.0, min_mass_kg=40.0)
    assert len(gaps) == 1
    assert "shorter than min_length_m" in gaps[0].reason


# ----------------------------------------------- the guarantee, end to end


HARD_BREAK_COLUMNS = [M.GEOMET_DOMAIN, M.attr_col("weathering")]


def test_no_candidate_composite_spans_a_hard_break(pipeline):
    """The correctness rule: check every generated candidate against its intervals.

    This asserts the property on the real candidate table rather than trusting that
    runs were built correctly, so weakening the run logic would fail here too.
    """
    intervals_by_id = pipeline.intervals.set_index(M.INTERVAL_ID)
    assert len(pipeline.candidates) > 0, "fixture produced no candidates to check"

    for row in pipeline.candidates.itertuples(index=False):
        members = intervals_by_id.loc[list(getattr(row, INTERVAL_IDS))]
        assert members[M.HOLE_ID].nunique() == 1, f"{row.candidate_id} spans two holes"
        assert members[M.RUN_ID].nunique() == 1, f"{row.candidate_id} spans two runs"
        for column in HARD_BREAK_COLUMNS:
            if column in members.columns:
                assert members[column].nunique(dropna=False) == 1, (
                    f"{row.candidate_id} spans more than one {column}"
                )


def test_no_candidate_spans_a_core_gap(pipeline):
    intervals_by_id = pipeline.intervals.set_index(M.INTERVAL_ID)
    tolerance = pipeline.cfg.domaining.max_interval_gap_m
    for row in pipeline.candidates.itertuples(index=False):
        members = intervals_by_id.loc[list(getattr(row, INTERVAL_IDS))].sort_values(M.FROM_M)
        gaps = members[M.FROM_M].to_numpy()[1:] - members[M.TO_M].to_numpy()[:-1]
        assert (gaps <= tolerance + 1e-9).all(), f"{row.candidate_id} spans a core gap"


def test_no_candidate_contains_an_unusable_interval(pipeline):
    intervals_by_id = pipeline.intervals.set_index(M.INTERVAL_ID)
    for row in pipeline.candidates.itertuples(index=False):
        members = intervals_by_id.loc[list(getattr(row, INTERVAL_IDS))]
        assert members["usable"].all(), f"{row.candidate_id} contains an unusable interval"
        assert not members[M.OUTSIDE_MODEL].any()


def test_candidates_are_contiguous_over_their_reported_extent(pipeline):
    intervals_by_id = pipeline.intervals.set_index(M.INTERVAL_ID)
    for row in pipeline.candidates.itertuples(index=False):
        members = intervals_by_id.loc[list(getattr(row, INTERVAL_IDS))]
        assert members[M.FROM_M].min() == pytest.approx(row.from_m)
        assert members[M.TO_M].max() == pytest.approx(row.to_m)


def test_generate_candidates_on_an_empty_table_returns_an_empty_frame(cfg):
    empty = intervals([]).reindex(
        columns=[M.HOLE_ID, M.FROM_M, M.TO_M, M.LENGTH_M, M.RUN_ID, M.MASS_KG]
    )
    out, gaps = generate_candidates(empty, cfg)
    assert out.empty


# --------------------------------------------------------- degenerate inputs


def test_segmenting_an_empty_table_returns_an_empty_table():
    empty = pd.DataFrame(columns=[M.HOLE_ID, M.FROM_M, M.TO_M, M.LENGTH_M, M.GEOMET_DOMAIN])
    out = run_segment(empty)
    assert out.empty
    assert M.RUN_ID in out.columns


def test_a_hard_break_key_absent_from_the_data_is_skipped_not_fatal():
    """A role declared in config but missing from a partial table must not crash."""
    frame = intervals(
        [
            {M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A"},
            {M.HOLE_ID: "H1", M.FROM_M: 1, M.TO_M: 2, M.GEOMET_DOMAIN: "A"},
        ]
    )
    out = run_segment(frame, hard_break_on=[M.GEOMET_DOMAIN, "never_logged"])
    assert out[M.RUN_ID].nunique() == 1


def test_summarising_no_runs_gives_an_empty_summary():
    empty = pd.DataFrame(columns=[M.HOLE_ID, M.FROM_M, M.TO_M, M.LENGTH_M, M.RUN_ID])
    assert summarise_runs(empty, [M.GEOMET_DOMAIN]).empty


def test_no_runs_means_no_gap_entries():
    assert short_run_gaps(pd.DataFrame(), min_length_m=6.0, min_mass_kg=40.0) == []


def test_an_unusable_run_is_not_reported_as_a_short_run():
    """It is already accounted for by its break reason; reporting it twice is noise."""
    frame = intervals(
        [{M.HOLE_ID: "H1", M.FROM_M: 0, M.TO_M: 1, M.GEOMET_DOMAIN: "A", M.LOW_RECOVERY: True}]
    )
    summary = summarise_runs(run_segment(frame), [M.GEOMET_DOMAIN])
    assert short_run_gaps(summary, min_length_m=6.0, min_mass_kg=40.0) == []
