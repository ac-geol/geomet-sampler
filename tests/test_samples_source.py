"""A single desurveyed samples table in place of collar, survey, assay and litho."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from geomet_sampler import models as M
from geomet_sampler.composite.candidates import EST_MASS_KG
from geomet_sampler.config import Config, load_config
from geomet_sampler.errors import MappingError
from geomet_sampler.intervals.merge import framework_from_samples
from geomet_sampler.io.readers import load_all, read_samples
from geomet_sampler.pipeline import run_pipeline
from geomet_sampler.select.greedy import COMPOSITE_ID
from geomet_sampler.validate.checks import check_hole_direction, check_supplied_geometry, hole_ends

from .fixtures import SAMPLES_COLUMNS, write_samples_project


@pytest.fixture(scope="session")
def samples_root(tmp_path_factory, project_root):
    root = tmp_path_factory.mktemp("samples_project")
    write_samples_project(root, project_root)
    return root


@pytest.fixture
def samples_cfg(samples_root) -> Config:
    return load_config(samples_root / "project.yaml")


def _raw_config(root) -> dict:
    return yaml.safe_load((root / "project.yaml").read_text())


def _write_config(root, raw: dict, name: str = "project.yaml"):
    path = root / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# ----------------------------------------------------------------- equivalence


@pytest.mark.parametrize("feet", [False, True], ids=["metres", "feet"])
def test_a_samples_table_selects_exactly_what_the_separate_files_select(
    tmp_path, project_root, feet
):
    """The two layouts describe the same intervals, so the answer must not change.

    The feet variant also renames every column, so a leaked name or a convention
    applied twice would show up here as a different selection.
    """
    config = write_samples_project(tmp_path, project_root, feet=feet)
    separate = run_pipeline(load_config(project_root / "project.yaml"))
    single = run_pipeline(load_config(config))

    assert single.report.errors == []
    columns = [COMPOSITE_ID, M.HOLE_ID, M.FROM_M, M.TO_M, M.LENGTH_M, M.ALLOCATION_DOMAIN]
    pd.testing.assert_frame_equal(
        separate.selected[columns].reset_index(drop=True),
        single.selected[columns].reset_index(drop=True),
        atol=1e-6,
    )
    pd.testing.assert_frame_equal(
        separate.picks[[COMPOSITE_ID, M.INTERVAL_ID, M.FROM_M, M.TO_M]].reset_index(drop=True),
        single.picks[[COMPOSITE_ID, M.INTERVAL_ID, M.FROM_M, M.TO_M]].reset_index(drop=True),
        atol=1e-6,
    )
    assert separate.selected[EST_MASS_KG].sum() == pytest.approx(
        single.selected[EST_MASS_KG].sum(), rel=1e-6
    )


def test_the_summary_says_where_the_geometry_came_from(samples_cfg):
    state = run_pipeline(samples_cfg)
    assert state.summary["geometry_source"] == "supplied by desurveyed samples table"
    assert "holes_in_collar" not in state.summary


# ---------------------------------------------------------------------- config


def test_samples_and_separate_files_cannot_be_mixed(samples_root, project_root):
    raw = _raw_config(samples_root)
    raw["sources"]["collar"] = _raw_config(project_root)["sources"]["collar"]
    with pytest.raises(ValidationError, match="replaces collar, survey, assay, litho"):
        Config.model_validate(raw)


def test_leaving_out_one_separate_file_names_both_ways_to_fix_it(project_root):
    raw = _raw_config(project_root)
    del raw["sources"]["survey"]
    with pytest.raises(ValidationError, match="missing survey.*single desurveyed table"):
        Config.model_validate(raw)


def test_dip_convention_is_not_needed_without_a_survey(samples_cfg):
    assert samples_cfg.conventions.dip_convention is None


def test_dip_convention_is_still_required_when_a_survey_is_read(project_root):
    raw = _raw_config(project_root)
    del raw["conventions"]["dip_convention"]
    with pytest.raises(ValidationError, match="dip_convention is required when survey"):
        Config.model_validate(raw)


# ---------------------------------------------------------------------- reader


def test_the_reader_maps_coordinates_and_codes_to_canonical_names(samples_cfg):
    samples, _ = read_samples(samples_cfg)
    for col in M.SAMPLES_REQUIRED:
        assert col in samples.columns
    assert not set(SAMPLES_COLUMNS.values()) & set(samples.columns)
    assert samples[M.Z_FROM].gt(samples[M.Z_TO]).all()  # the fixture holes go down


def test_coordinates_are_converted_with_the_declared_length_units(tmp_path, project_root):
    metres, _ = read_samples(load_config(write_samples_project(tmp_path / "m", project_root)))
    feet, _ = read_samples(
        load_config(write_samples_project(tmp_path / "ft", project_root, feet=True))
    )
    for col in (M.FROM_M, M.TO_M, *M.FROM_XYZ, *M.TO_XYZ, *M.MID_XYZ):
        np.testing.assert_allclose(metres[col], feet[col], atol=1e-6)


def test_every_coordinate_must_be_mapped(samples_root):
    raw = _raw_config(samples_root)
    del raw["sources"]["samples"]["columns"]["z_mid"]
    cfg = load_config(_write_config(samples_root, raw, "no_zmid.yaml"))
    with pytest.raises(MappingError, match="z_mid must be mapped"):
        load_all(cfg)


def test_unsampled_rows_are_kept_with_blank_grades_and_a_generated_id(tmp_path, project_root):
    config = write_samples_project(tmp_path, project_root)
    path = tmp_path / "data" / "samples.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.loc[[3, 4], "S_LABNO"] = ""
    table.loc[[3, 4], "S_ZN"] = ""
    table.to_csv(path, index=False)

    cfg = load_config(config)
    samples, issues = read_samples(cfg)
    assert len(samples) == len(table)
    assert samples[M.SAMPLE_ID].str.len().gt(0).all()
    assert samples[M.elem_col("Zn")].isna().sum() == 2
    blank = [i for i in issues if i.check == "sample_id_blank"]
    assert blank and blank[0].count == 2


def test_unsampled_rows_break_runs_rather_than_joining_composites(tmp_path, project_root):
    config = write_samples_project(tmp_path, project_root)
    path = tmp_path / "data" / "samples.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    unsampled = table.index[(table["S_HOLE"] == "DDH001") & (table["S_FROM"].astype(float) < 30)]
    middle = unsampled[len(unsampled) // 2]
    table.loc[middle, ["S_LABNO", "S_ZN"]] = ""
    table.to_csv(path, index=False)

    state = run_pipeline(load_config(config), through="candidates")
    gap_id = state.intervals.loc[
        (state.intervals[M.HOLE_ID] == "DDH001")
        & np.isclose(state.intervals[M.FROM_M], float(table.loc[middle, "S_FROM"])),
        M.INTERVAL_ID,
    ].item()
    for ids in state.candidates["interval_ids"]:
        assert gap_id not in ids


# ------------------------------------------------------------------- framework


def _two_rows(**overrides) -> pd.DataFrame:
    rows = {
        M.HOLE_ID: ["H1", "H1"],
        M.SAMPLE_ID: ["S1", "S1"],
        M.FROM_M: [0.0, 1.0],
        M.TO_M: [1.0, 2.0],
        M.LOGGED_CODE: ["VOLC", None],
        M.LOW_RECOVERY: [False, False],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_the_framework_keeps_every_row_and_splits_nothing():
    lookup = pd.DataFrame({M.LOGGED_CODE: ["VOLC"], M.GEOMET_DOMAIN: ["VOLCANIC"]})
    framework, issues = framework_from_samples(_two_rows(), lookup)
    assert len(framework) == 2
    assert framework[M.GEOMET_DOMAIN].tolist()[0] == "VOLCANIC"
    assert pd.isna(framework[M.GEOMET_DOMAIN].tolist()[1])
    # a sample ID on two rows keeps the ID and gets distinct interval IDs
    assert framework[M.PARENT_SAMPLE_ID].tolist() == ["S1", "S1"]
    assert framework[M.INTERVAL_ID].is_unique
    checks = {i.check for i in issues}
    assert {"boundaries_inherited", "interval_not_logged", "sample_split_upstream"} <= checks


def test_an_unmapped_logged_code_is_an_error_in_this_layout_too():
    lookup = pd.DataFrame({M.LOGGED_CODE: ["TUFF"], M.GEOMET_DOMAIN: ["VOLCANIC"]})
    _, issues = framework_from_samples(_two_rows(), lookup)
    assert any(i.check == "logged_code_unmapped" and i.severity == "ERROR" for i in issues)


# ------------------------------------------------------------ geometry checks


def _hole(length_scale: float = 1.0, *, upward: bool = False, n: int = 4) -> pd.DataFrame:
    """A vertical hole of 1 m rows; coordinates scaled against the depths."""
    sign = 1.0 if upward else -1.0
    from_m = np.arange(n, dtype=float)
    to_m = from_m + 1.0
    z_from = 100.0 + sign * from_m * length_scale
    z_to = 100.0 + sign * to_m * length_scale
    return pd.DataFrame(
        {
            M.HOLE_ID: "H1",
            M.FROM_M: from_m,
            M.TO_M: to_m,
            M.X_FROM: 0.0,
            M.Y_FROM: 0.0,
            M.Z_FROM: z_from,
            M.X_TO: 0.0,
            M.Y_TO: 0.0,
            M.Z_TO: z_to,
            M.X_MID: 0.0,
            M.Y_MID: 0.0,
            M.Z_MID: (z_from + z_to) / 2.0,
        }
    )


def _checks(frame: pd.DataFrame) -> dict[str, str]:
    return {i.check: i.severity.value for i in check_supplied_geometry(frame)}


def test_consistent_geometry_raises_nothing():
    assert _checks(_hole()) == {}


def test_coordinates_in_feet_against_depths_in_metres_is_an_error():
    found = _checks(_hole(length_scale=1 / 0.3048))
    assert found == {"geometry_unit_mismatch": "ERROR"}
    [issue] = check_supplied_geometry(_hole(length_scale=1 / 0.3048))
    assert "coordinates look like feet" in issue.message


def test_depths_in_feet_against_coordinates_in_metres_is_an_error():
    [issue] = check_supplied_geometry(_hole(length_scale=0.3048))
    assert issue.check == "geometry_unit_mismatch"
    assert "depths look like feet" in issue.message


def test_any_other_length_disagreement_is_a_warning():
    assert _checks(_hole(length_scale=1.5)) == {"geometry_length_mismatch": "WARN"}


def test_a_midpoint_off_the_interval_is_a_warning():
    frame = _hole()
    frame.loc[1, M.X_MID] = 5.0
    assert _checks(frame) == {"geometry_midpoint_offset": "WARN"}


def test_rows_that_do_not_join_up_are_a_warning():
    frame = _hole()
    for col in (M.X_FROM, M.X_TO, M.X_MID):
        frame.loc[2:, col] = 50.0
    assert _checks(frame) == {"geometry_discontinuous": "WARN"}


def test_an_upward_hole_is_listed_but_not_rejected():
    """Underground holes drilled upward are real; only the user knows which ones."""
    frame = pd.concat(
        [
            _hole(upward=True),
            _hole().assign(**{M.HOLE_ID: "H2"}),
            _hole().assign(**{M.HOLE_ID: "H3"}),
        ],
        ignore_index=True,
    )
    assert check_supplied_geometry(frame) == []
    ends = hole_ends(frame, traces={})
    assert ends[M.RISES].tolist() == [True, False, False]
    assert ends.loc[0, list(M.COLLAR_XYZ)].tolist() == [0.0, 0.0, 100.0]
    assert ends.loc[0, list(M.TOE_XYZ)].tolist() == [0.0, 0.0, 104.0]
    found = check_hole_direction(ends)
    assert [(i.check, i.severity.value) for i in found] == [("hole_rises_with_depth", "INFO")]
    assert found[0].detail["holes"] == ["H1"]


def test_a_hole_that_rises_and_falls_is_a_warning():
    down = _hole(n=2)
    up = _hole(n=2, upward=True)
    up[[M.FROM_M, M.TO_M]] += 2.0
    up[[M.Z_FROM, M.Z_TO, M.Z_MID]] -= 4.0  # continues from where the downward part ended
    frame = pd.concat([down, up], ignore_index=True)
    assert _checks(frame)["hole_changes_vertical_direction"] == "WARN"


def test_missing_coordinates_are_reported_not_dropped():
    frame = _hole()
    frame.loc[0, M.Z_MID] = np.nan
    found = check_supplied_geometry(frame)
    assert found[0].check == "coordinates_missing"
    assert found[0].count == 1


def test_the_validation_report_says_which_checks_did_not_run(samples_cfg):
    state = run_pipeline(samples_cfg, through="load")
    checks = {i.check for i in state.report.issues}
    assert "geometry_supplied" in checks
    assert not checks & {"collar_duplicate", "orphan_hole", "survey_empty"}


# ------------------------------------------------------------------------ init


def test_init_drafts_a_samples_source_from_one_table(tmp_path, project_root):
    from typer.testing import CliRunner

    from geomet_sampler.cli import app

    header = (
        "HoleID,SampleID,From_m,To_m,Lith,X_From,Y_From,Z_From,X_To,Y_To,Z_To,"
        "X_Mid,Y_Mid,Z_Mid,Zn_pct,SG"
    )
    table = tmp_path / "desurveyed.csv"
    table.write_text(
        header + "\nH1,S1,0,1,VOLC,0,0,100,0,0,99,0,0,99.5,1.2,2.7\n", encoding="utf-8"
    )
    out = tmp_path / "draft.yaml"
    result = CliRunner().invoke(
        app,
        [
            "init",
            "--out",
            str(out),
            "--samples",
            str(table),
            "--block-model",
            str(project_root / "data" / "block_model.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(out.read_text())
    assert "collar" not in raw["sources"]
    assert "dip_convention" not in raw["conventions"]
    columns = raw["sources"]["samples"]["columns"]
    for canonical, column in {
        "hole_id": "HoleID",
        "sample_id": "SampleID",
        "from_m": "From_m",
        "to_m": "To_m",
        "logged_code": "Lith",
        "x_from": "X_From",
        "z_to": "Z_To",
        "y_mid": "Y_Mid",
        "z_mid": "Z_Mid",
    }.items():
        assert columns[canonical] == column
    assert raw["elements"]["Zn"]["drillhole"]["field"] == "Zn_pct"
    assert "# CONFIRM" in out.read_text()
