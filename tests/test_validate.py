"""Validation checks.

Nothing here modifies data. The point of each check is that a problem gets named and
counted rather than quietly changing the answer, so the tests assert on the issues
produced, not on any repair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.desurvey import build_traces
from geomet_sampler.io.readers import Dataset, load_all
from geomet_sampler.models import Severity
from geomet_sampler.validate.checks import (
    check_model_extent,
    check_toe_above_collar,
    report_frame,
    summarise,
    validate_inputs,
)


@pytest.fixture
def dataset(cfg) -> Dataset:
    data, _ = load_all(cfg)
    return data


def checks(issues, name):
    return [i for i in issues if i.check == name]


def test_the_fixture_data_raises_no_errors(dataset, cfg):
    assert [i.check for i in validate_inputs(dataset, cfg) if i.severity is Severity.ERROR] == []


def test_duplicate_collars_are_an_error(dataset, cfg):
    dataset.collar = pd.concat([dataset.collar, dataset.collar.head(1)], ignore_index=True)
    found = checks(validate_inputs(dataset, cfg), "collar_duplicate")
    assert found and found[0].severity is Severity.ERROR


def test_a_hole_with_no_collar_record_is_an_error(dataset, cfg):
    dataset.assay.loc[0, M.HOLE_ID] = "NOT_A_HOLE"
    found = checks(validate_inputs(dataset, cfg), "orphan_hole")
    assert found and "NOT_A_HOLE" in found[0].message


def test_an_interval_that_ends_before_it_starts_is_an_error(dataset, cfg):
    dataset.assay.loc[0, M.TO_M] = dataset.assay.loc[0, M.FROM_M] - 1.0
    found = checks(validate_inputs(dataset, cfg), "interval_order")
    assert found and found[0].severity is Severity.ERROR


def test_overlapping_intervals_are_an_error(dataset, cfg):
    dataset.assay.loc[1, M.FROM_M] = dataset.assay.loc[0, M.FROM_M]
    found = checks(validate_inputs(dataset, cfg), "interval_overlap")
    assert found and found[0].severity is Severity.ERROR


def test_a_blank_depth_is_an_error(dataset, cfg):
    dataset.assay.loc[0, M.FROM_M] = np.nan
    found = checks(validate_inputs(dataset, cfg), "interval_depth_missing")
    assert found and found[0].severity is Severity.ERROR


def test_gaps_are_graded_against_the_configured_tolerance(dataset, cfg):
    dataset.assay.loc[1, M.FROM_M] += 0.05  # within tolerance
    dataset.assay.loc[5, M.FROM_M] += 5.0  # beyond it
    issues = validate_inputs(dataset, cfg)
    severities = {i.severity for i in checks(issues, "interval_gap")}
    assert Severity.INFO in severities
    assert Severity.WARN in severities


def test_an_interval_below_the_collar_total_depth_is_a_warning(dataset, cfg):
    dataset.assay.loc[0, M.TO_M] = 9_999.0
    found = checks(validate_inputs(dataset, cfg), "beyond_total_depth")
    assert found and found[0].severity is Severity.WARN


def test_an_out_of_range_azimuth_is_an_error(dataset, cfg):
    dataset.survey.loc[0, M.AZIMUTH] = 400.0
    found = checks(validate_inputs(dataset, cfg), "azimuth_range")
    assert found and "Azimuth" in found[0].message  # named in the user's own column


def test_an_out_of_range_dip_is_an_error(dataset, cfg):
    dataset.survey.loc[0, M.DIP] = -120.0
    found = checks(validate_inputs(dataset, cfg), "dip_range")
    assert found and found[0].severity is Severity.ERROR


def test_upward_pointing_dips_after_normalisation_are_flagged(dataset, cfg):
    """The classic convention error, visible before anything is desurveyed."""
    dataset.survey[M.DIP] = dataset.survey[M.DIP].abs()
    found = checks(validate_inputs(dataset, cfg), "dip_sign")
    assert found and "dip_convention" in found[0].message


def test_incomplete_survey_stations_are_reported(dataset, cfg):
    dataset.survey.loc[0, M.AZIMUTH] = np.nan
    assert checks(validate_inputs(dataset, cfg), "survey_incomplete")


def test_a_repeated_survey_depth_is_reported(dataset, cfg):
    dataset.survey = pd.concat(
        [dataset.survey, dataset.survey.head(1)], ignore_index=True
    ).sort_values([M.HOLE_ID, M.DEPTH])
    assert checks(validate_inputs(dataset, cfg), "survey_duplicate_depth")


def test_an_empty_survey_file_is_an_error(dataset, cfg):
    dataset.survey = dataset.survey.iloc[0:0]
    found = checks(validate_inputs(dataset, cfg), "survey_empty")
    assert found and found[0].severity is Severity.ERROR


def test_an_implausible_density_is_an_error(dataset, cfg):
    """After normalisation, 2700 means the units were declared wrong."""
    dataset.assay.loc[0, M.DENSITY] = 2700.0
    found = checks(validate_inputs(dataset, cfg), "density_range")
    assert found and "density_units" in found[0].message


def test_a_missing_density_is_a_warning(dataset, cfg):
    dataset.assay.loc[0, M.DENSITY] = np.nan
    found = checks(validate_inputs(dataset, cfg), "density_missing")
    assert found and found[0].severity is Severity.WARN


def test_duplicate_block_centroids_are_an_error(dataset, cfg):
    dataset.block_model = pd.concat(
        [dataset.block_model, dataset.block_model.head(1)], ignore_index=True
    )
    found = checks(validate_inputs(dataset, cfg), "block_duplicate_centroid")
    assert found and found[0].severity is Severity.ERROR


def test_a_block_model_with_no_scheduled_period_is_an_error(dataset, cfg):
    dataset.block_model[M.PERIOD] = None
    found = checks(validate_inputs(dataset, cfg), "period_empty")
    assert found and "Allocation targets come from scheduled tonnage" in found[0].message


def test_unscheduled_blocks_are_reported_as_information(dataset, cfg):
    dataset.block_model.loc[0, M.PERIOD] = ""
    found = checks(validate_inputs(dataset, cfg), "period_unscheduled")
    assert found and found[0].severity is Severity.INFO


def test_a_zero_size_block_is_an_error(dataset, cfg):
    dataset.block_model.loc[0, M.BLOCK_DZ] = 0.0
    found = checks(validate_inputs(dataset, cfg), "block_dimension")
    assert found and found[0].severity is Severity.ERROR


def test_an_empty_block_model_is_an_error(dataset, cfg):
    dataset.block_model = dataset.block_model.iloc[0:0]
    found = checks(validate_inputs(dataset, cfg), "block_model_empty")
    assert found and found[0].severity is Severity.ERROR


def test_a_logged_code_outside_the_lookup_is_an_error(dataset, cfg):
    dataset.litho.loc[0, M.LOGGED_CODE] = "UNKNOWN_UNIT"
    found = checks(validate_inputs(dataset, cfg), "logged_code_unmapped")
    assert found and "UNKNOWN_UNIT" in found[0].message


# ---------------------------------------------------- convention sanity checks


def test_an_inverted_dip_convention_is_caught_by_the_toe_check(dataset, cfg):
    """Toe above collar is the signature of a positive-down file read as negative-down."""
    dataset.survey[M.DIP] = dataset.survey[M.DIP].abs()
    traces, _ = build_traces(dataset.collar, dataset.survey)
    found = check_toe_above_collar(traces, dataset.collar)
    assert found and found[0].severity is Severity.ERROR
    assert "dip_convention" in found[0].message


def test_a_correctly_orientated_hole_passes_the_toe_check(dataset, cfg):
    traces, _ = build_traces(dataset.collar, dataset.survey)
    assert check_toe_above_collar(traces, dataset.collar) == []


def test_mostly_off_model_drilling_is_an_error_about_coordinates():
    intervals = pd.DataFrame({M.OUTSIDE_MODEL: [True] * 9 + [False]})
    found = check_model_extent(intervals, error_fraction=0.5)
    assert found and "different coordinate systems or different length units" in found[0].message


def test_a_little_off_model_drilling_is_accepted():
    intervals = pd.DataFrame({M.OUTSIDE_MODEL: [True] + [False] * 9})
    assert check_model_extent(intervals, error_fraction=0.5) == []


def test_extent_check_on_an_empty_table_says_nothing():
    assert check_model_extent(pd.DataFrame(), error_fraction=0.5) == []


# ------------------------------------------------------------------ reporting


def test_the_report_lists_errors_before_warnings(dataset, cfg):
    dataset.collar = pd.concat([dataset.collar, dataset.collar.head(1)], ignore_index=True)
    dataset.assay.loc[0, M.DENSITY] = np.nan
    frame = report_frame(validate_inputs(dataset, cfg))
    assert frame["severity"].iloc[0] == "ERROR"
    assert set(frame.columns) == {"severity", "check", "source", "hole_id", "count", "message"}


def test_an_empty_report_still_has_the_expected_columns():
    frame = report_frame([])
    assert frame.empty
    assert "severity" in frame.columns


def test_summarise_counts_every_severity(dataset, cfg):
    counts = summarise(validate_inputs(dataset, cfg))
    assert set(counts) == {"ERROR", "WARN", "INFO"}
