"""Whole-pipeline behaviour, including the disguise smoke test."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from geomet_sampler import models as M
from geomet_sampler.cli import app
from geomet_sampler.composite.candidates import EST_MASS_KG, INTERVAL_IDS
from geomet_sampler.config import load_config
from geomet_sampler.errors import ValidationFailedError
from geomet_sampler.pipeline import run_pipeline
from geomet_sampler.report.excel import build_sheets, write_outputs
from geomet_sampler.select.greedy import COMPOSITE_ID, REASON

from .fixtures import write_disguised_project

runner = CliRunner()


# ------------------------------------------------------------------- schema


def test_the_pipeline_runs_clean_on_the_fixture(pipeline):
    assert [i.check for i in pipeline.report.errors] == []
    assert len(pipeline.selected) > 0


def test_selected_composites_carry_the_full_schema(pipeline):
    expected = {
        COMPOSITE_ID,
        M.HOLE_ID,
        M.RUN_ID,
        M.FROM_M,
        M.TO_M,
        M.LENGTH_M,
        EST_MASS_KG,
        M.GEOMET_DOMAIN,
        M.PERIOD,
        M.GRADE_BIN,
        M.ALLOCATION_DOMAIN,
        M.DOMAIN_MATCH,
        M.IS_PLANNED,
        INTERVAL_IDS,
        REASON,
    }
    assert expected <= set(pipeline.selected.columns)


def test_declared_attributes_and_elements_appear_without_being_enumerated(pipeline):
    """The columns come from config at runtime, so a new role needs no code change."""
    for role in pipeline.cfg.attributes:
        assert M.attr_col(role) in pipeline.selected.columns
    for element in pipeline.cfg.elements_with_drillhole():
        assert M.wtd_elem_col(element) in pipeline.selected.columns


def test_every_composite_traces_back_to_its_intervals(pipeline):
    """A composite without its interval list is useless to the core shed."""
    assert not pipeline.picks.empty
    for row in pipeline.selected.itertuples(index=False):
        picks = pipeline.picks[pipeline.picks[COMPOSITE_ID] == getattr(row, COMPOSITE_ID)]
        assert len(picks) == row.n_intervals
        assert picks[M.LENGTH_M].sum() == pytest.approx(row.length_m, abs=1e-6)


def test_no_pick_list_interval_appears_in_two_composites(pipeline):
    assert pipeline.picks[M.INTERVAL_ID].is_unique


def test_every_selected_composite_meets_the_mass_requirement(pipeline):
    """A recommendation the user cannot physically collect is worse than none."""
    assert (pipeline.selected[EST_MASS_KG] >= pipeline.cfg.compositing.min_mass_kg).all()


def test_every_selection_carries_a_reason(pipeline):
    assert pipeline.selected[REASON].str.len().gt(0).all()


def test_unfilled_deficits_are_recorded_rather_than_dropped(pipeline):
    achieved = pipeline.allocation["achieved"].sum()
    shortfall = (pipeline.allocation["deficit"] - pipeline.allocation["achieved"]).clip(lower=0)
    domain_gaps = [g for g in pipeline.gaps if g.scope == "domain"]
    assert len(domain_gaps) == int((shortfall > 0).sum())
    assert achieved + shortfall.sum() >= pipeline.cfg.allocation.total_samples


def test_the_run_is_reproducible(project_root):
    first = run_pipeline(load_config(project_root / "project.yaml"))
    second = run_pipeline(load_config(project_root / "project.yaml"))
    pd.testing.assert_frame_equal(
        first.selected.reset_index(drop=True), second.selected.reset_index(drop=True)
    )


# -------------------------------------------------------- the disguise test


def test_renaming_every_column_and_changing_conventions_changes_nothing(tmp_path, project_root):
    """The smoke test from CLAUDE.md.

    Rename every column, quote lengths in feet and dips positive-down, and write a
    config for it. The tool must produce identical output. If it does not, a convention
    or a column name has leaked out of the readers and into the library.
    """
    disguised_root = tmp_path / "disguised"
    disguised_root.mkdir()
    disguised_config = write_disguised_project(disguised_root, project_root)

    original = run_pipeline(load_config(project_root / "project.yaml"))
    disguised = run_pipeline(load_config(disguised_config))

    columns = [COMPOSITE_ID, M.HOLE_ID, M.FROM_M, M.TO_M, M.LENGTH_M, M.ALLOCATION_DOMAIN]
    pd.testing.assert_frame_equal(
        original.selected[columns].reset_index(drop=True),
        disguised.selected[columns].reset_index(drop=True),
        atol=1e-6,
    )
    pd.testing.assert_frame_equal(
        original.allocation[[M.ALLOCATION_DOMAIN, "deficit", "achieved"]].reset_index(drop=True),
        disguised.allocation[[M.ALLOCATION_DOMAIN, "deficit", "achieved"]].reset_index(drop=True),
    )
    assert original.selected[EST_MASS_KG].sum() == pytest.approx(
        disguised.selected[EST_MASS_KG].sum(), rel=1e-6
    )


# ------------------------------------------------------------------ outputs


def test_the_workbook_has_every_expected_sheet(pipeline):
    sheets = build_sheets(pipeline)
    assert list(sheets) == [
        "Summary",
        "Allocation",
        "Composites",
        "Pick_List",
        "Gap_Register",
        "Validation",
        "Candidates",
    ]


def test_grades_are_reported_back_in_the_declared_units(pipeline):
    """Internally ppm; a geologist reading the workbook wants percent."""
    composites = build_sheets(pipeline)["Composites"]
    assert "Zn_pct" in composites.columns
    assert composites["Zn_pct"].max() < 100.0


def test_attribute_roles_are_reported_under_their_own_names(pipeline):
    composites = build_sheets(pipeline)["Composites"]
    for role in pipeline.cfg.attributes:
        assert role in composites.columns


def test_outputs_are_written_where_the_config_says(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path / "out"
    written = write_outputs(pipeline)
    for path in written.values():
        assert path.exists()
    assert (tmp_path / "out" / "pick_list.csv").exists()


# ---------------------------------------------------------------------- CLI


def test_validate_command_reports_and_exits_zero_on_clean_data(project_root):
    result = runner.invoke(app, ["validate", "--config", str(project_root / "project.yaml")])
    assert result.exit_code == 0, result.output
    assert "0 errors" in result.output


def test_run_command_produces_the_workbook(project_root, tmp_path):
    result = runner.invoke(app, ["run", "--config", str(project_root / "project.yaml")])
    assert result.exit_code == 0, result.output
    assert "samples_selected" in result.output


def test_stage_commands_write_their_own_tables(project_root, tmp_path):
    config = str(project_root / "project.yaml")
    for command, filename in (
        ("desurvey", "desurveyed.csv"),
        ("candidates", "candidates.csv"),
        ("allocate", "allocation.csv"),
    ):
        out = tmp_path / filename
        result = runner.invoke(app, [command, "--config", config, "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()


def test_validation_errors_stop_the_run_unless_forced(project_root, tmp_path, monkeypatch):
    """A bad dip convention must not quietly produce a plausible-looking answer."""
    import yaml

    raw = yaml.safe_load((project_root / "project.yaml").read_text())
    raw["conventions"]["dip_convention"] = "positive_down"
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # paths in the fixture config are relative to its own directory
    raw["sources"] = {
        k: ({**v, "path": str(project_root / v["path"])} if isinstance(v, dict) else v)
        for k, v in raw["sources"].items()
    }
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationFailedError, match="toe_above_collar"):
        run_pipeline(load_config(broken))

    forced = run_pipeline(load_config(broken), force=True)
    assert forced.report.has_errors()


def test_init_writes_a_draft_marked_for_confirmation(project_root, tmp_path):
    out = tmp_path / "draft.yaml"
    result = runner.invoke(
        app,
        [
            "init",
            "--out",
            str(out),
            "--collar",
            str(project_root / "data" / "collar.csv"),
            "--survey",
            str(project_root / "data" / "survey.csv"),
            "--assay",
            str(project_root / "data" / "assay.csv"),
            "--litho",
            str(project_root / "data" / "litho.csv"),
            "--block-model",
            str(project_root / "data" / "block_model.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "# CONFIRM" in text
    assert "hole_id: HoleID" in text
    assert "dip_convention: negative_down" in text
    assert "Zn" in text
