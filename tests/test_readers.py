"""Readers: file quirks, unit normalisation and mapping failures.

Real exports are messy. The readers absorb that, and the messiness is reproduced in the
fixtures deliberately, because hand-cleaning the files would only move the problem to
the next export.
"""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.config import load_config
from geomet_sampler.errors import MappingError
from geomet_sampler.io.readers import load_all, read_assay, read_availability, read_survey
from geomet_sampler.models import Availability, Severity
from geomet_sampler.units import FEET_TO_METRES


def test_a_utf8_bom_on_the_first_header_does_not_break_the_mapping(cfg):
    """The assay fixture carries a BOM, exactly as the reference export does."""
    assay, _ = read_assay(cfg)
    assert M.HOLE_ID in assay.columns
    assert assay[M.HOLE_ID].str.startswith("DDH").all()


def test_a_fully_quoted_file_still_yields_numbers(cfg):
    """The survey fixture is quoted throughout, so every field arrives as a string."""
    survey, _ = read_survey(cfg)
    assert pd.api.types.is_float_dtype(survey[M.DEPTH])
    assert pd.api.types.is_float_dtype(survey[M.DIP])


def test_inconsistent_header_casing_across_files_is_handled(cfg):
    """The lithology fixture uses lowercase headers while the others do not."""
    dataset, _ = load_all(cfg)
    assert set(dataset.litho.columns) >= {M.HOLE_ID, M.FROM_M, M.TO_M, M.LOGGED_CODE}


def test_a_missing_sample_id_column_generates_ids_and_warns(cfg):
    assay, issues = read_assay(cfg)
    assert assay[M.SAMPLE_ID].notna().all()
    warning = next(i for i in issues if i.check == "tier2_fallback")
    assert warning.severity is Severity.WARN
    assert "pick list is degraded" in warning.message


def test_grades_are_normalised_to_ppm_at_load(cfg):
    """Zn is declared in percent; internally everything is ppm."""
    assay, _ = read_assay(cfg)
    upper = assay[assay[M.FROM_M] < 30.0][M.elem_col("Zn")]
    assert upper.mean() == pytest.approx(60_000.0, rel=0.02)


def test_dips_are_normalised_to_negative_down(cfg):
    survey, _ = read_survey(cfg)
    assert (survey[M.DIP] <= 0).all()


def test_lengths_are_converted_when_the_file_is_in_feet(cfg, tmp_path):
    metres, _ = read_survey(cfg)
    cfg.conventions.length_units = cfg.conventions.length_units.__class__.FT
    feet, _ = read_survey(cfg)
    assert (
        (feet[M.DEPTH] / metres[M.DEPTH].replace(0.0, pd.NA))
        .dropna()
        .round(6)
        .eq(round(FEET_TO_METRES, 6))
        .all()
    )


def test_site_availability_vocabulary_maps_onto_the_three_states(cfg):
    availability, issues = read_availability(cfg)
    assert set(availability[M.AVAILABILITY]) <= {s.value for s in Availability}
    assert Availability.UNAVAILABLE.value in set(availability[M.AVAILABILITY])
    assert not [i for i in issues if i.severity is Severity.ERROR]


def test_an_unmapped_availability_value_is_an_error(cfg, project_root):
    """An unrecognised state would fail closed and silently shrink the candidate pool."""
    path = project_root / "data" / "availability.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original.replace(b"IN STORAGE", b"WHO KNOWS"))
        _, issues = read_availability(cfg)
        assert [i.check for i in issues] == ["availability_unmapped"]
        assert issues[0].severity is Severity.ERROR
    finally:
        path.write_bytes(original)


def test_a_wrong_column_name_names_the_columns_that_are_present(cfg):
    cfg.sources.assay.columns[M.FROM_M] = "NotAColumn"
    with pytest.raises(MappingError) as excinfo:
        read_assay(cfg)
    message = str(excinfo.value)
    assert "NotAColumn" in message
    assert "Columns present" in message
    assert "From_m" in message


def test_every_bad_mapping_is_reported_in_one_pass(cfg):
    """Fix the whole config in one go, not one column per run."""
    cfg.sources.assay.columns[M.FROM_M] = "Nope1"
    cfg.sources.litho.columns[M.LOGGED_CODE] = "Nope2"
    with pytest.raises(MappingError) as excinfo:
        load_all(cfg)
    message = str(excinfo.value)
    assert "Nope1" in message
    assert "Nope2" in message


def test_block_tonnage_is_computed_at_load(cfg):
    dataset, _ = load_all(cfg)
    blocks = dataset.block_model
    expected = blocks[M.BLOCK_DX] * blocks[M.BLOCK_DY] * blocks[M.BLOCK_DZ] * blocks[M.DENSITY]
    assert blocks[M.BLOCK_TONNES].equals(expected)


def test_the_mapping_table_survives_for_translating_messages_back(cfg):
    dataset, _ = load_all(cfg)
    assert dataset.mappings["assay"][M.FROM_M] == "From_m"


def test_config_relative_paths_resolve_against_the_config_file(project_root):
    cfg = load_config(project_root / "project.yaml")
    assert cfg.sources.collar.path.is_absolute()
    assert cfg.sources.collar.path.exists()
