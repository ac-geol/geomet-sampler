"""Reporting: unit translation, workbook structure and plots."""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.io.writers import present, write_csv, write_workbook
from geomet_sampler.report.excel import build_sheets, write_outputs, write_validation_report
from geomet_sampler.report.plots import write_plots


def test_grades_are_converted_back_to_their_declared_units(cfg):
    out = present(pd.DataFrame({M.elem_col("Zn"): [60_000.0]}), cfg)
    assert out["Zn_pct"].iloc[0] == pytest.approx(6.0)


def test_a_weighted_grade_alongside_an_interval_grade_keeps_both_headers(cfg):
    """Two columns with one name would corrupt the sheet without saying so."""
    frame = pd.DataFrame({M.elem_col("Zn"): [60_000.0], M.wtd_elem_col("Zn"): [15_000.0]})
    out = present(frame, cfg)
    assert list(out.columns) == ["Zn_pct", "Zn_pct_wtd"]
    assert out["Zn_pct"].iloc[0] == pytest.approx(6.0)
    assert out["Zn_pct_wtd"].iloc[0] == pytest.approx(1.5)


def test_block_model_grades_keep_their_own_declared_units(cfg):
    out = present(pd.DataFrame({M.bm_elem_col("Zn"): [20_000.0]}), cfg)
    assert out["bm_Zn_pct"].iloc[0] == pytest.approx(2.0)


def test_attribute_roles_are_presented_under_their_own_names(cfg):
    frame = pd.DataFrame(
        {M.attr_col("rock_type"): ["VOLCANIC"], M.bm_attr_col("weathering"): ["OXIDE"]}
    )
    out = present(frame, cfg)
    assert list(out.columns) == ["rock_type", "bm_weathering"]


def test_interval_lists_are_flattened_for_reading(cfg):
    out = present(pd.DataFrame({M.INTERVAL_IDS_COL: [("A", "B", "C")]}), cfg)
    assert out[M.INTERVAL_IDS_COL].iloc[0] == "A, B, C"


def test_presenting_an_empty_frame_is_harmless(cfg):
    assert present(pd.DataFrame(), cfg).empty
    assert present(None, cfg).empty


def test_the_workbook_is_written_with_every_sheet(pipeline, tmp_path):
    path = write_workbook(build_sheets(pipeline), tmp_path / "picks.xlsx")
    assert path.exists()
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {
        "Summary",
        "Allocation",
        "Composites",
        "Pick_List",
        "Gap_Register",
        "Validation",
        "Candidates",
    }
    assert not sheets["Pick_List"].empty


def test_an_empty_sheet_still_gets_written_with_a_note(tmp_path):
    path = write_workbook({"Empty": pd.DataFrame()}, tmp_path / "book.xlsx")
    assert path.exists()


def test_csv_output_goes_where_it_is_asked(tmp_path):
    path = write_csv(pd.DataFrame({"a": [1]}), tmp_path / "nested" / "out.csv")
    assert path.exists()


def test_the_validation_report_is_written_even_for_a_clean_run(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path
    path = write_validation_report(pipeline)
    assert path.exists()
    assert "severity" in pd.read_csv(path).columns or path.read_text().startswith("severity")


def test_only_the_configured_formats_are_written(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path
    pipeline.cfg.reporting.formats = ["csv"]
    written = write_outputs(pipeline)
    assert "workbook" not in written
    assert (tmp_path / "composites.csv").exists()


def test_the_gap_register_can_be_suppressed(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path
    pipeline.cfg.reporting.formats = ["csv"]
    pipeline.cfg.reporting.include_gap_register = False
    write_outputs(pipeline)
    assert not (tmp_path / "gap_register.csv").exists()
    pipeline.cfg.reporting.include_gap_register = True


def test_every_chart_with_data_behind_it_is_drawn(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path
    written = write_plots(pipeline)
    assert {p.stem for p in written} == {
        "allocation_target_vs_achieved",
        "plan_view_selected",
        "long_section_period",
        "grade_distribution",
    }
    assert all(p.stat().st_size > 0 for p in written)


def test_plots_are_skipped_rather_than_failing_when_nothing_was_selected(pipeline, tmp_path):
    pipeline.cfg.project.output_dir = tmp_path / "empty"
    original = pipeline.selected
    try:
        pipeline.selected = original.iloc[0:0]
        written = write_plots(pipeline)
        assert [p.stem for p in written] == ["allocation_target_vs_achieved"]
    finally:
        pipeline.selected = original
