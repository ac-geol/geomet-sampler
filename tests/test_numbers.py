"""Text to numbers: blanks, declared null tokens, detection limits and everything else.

A value that is not a plain number must never become blank, or a grade, without the
report saying so. Below-detection text and sentinels such as -99 are the usual cases.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.config import AssaySource, load_config
from geomet_sampler.errors import ValidationFailedError
from geomet_sampler.io.numbers import NumberRules, parse_grades, parse_numbers
from geomet_sampler.io.readers import load_all, read_assay, read_block_model
from geomet_sampler.models import Severity
from geomet_sampler.pipeline import run_pipeline

from .fixtures import ORIGINAL, write_fixture_project

RULES = NumberRules(source="assay", file="assay.csv")


def text(*values: str) -> pd.Series:
    return pd.Series(list(values), dtype=object)


def only(issues, check):
    found = [i for i in issues if i.check == check]
    assert len(found) == 1, [i.check for i in issues]
    return found[0]


# ------------------------------------------------------------------ plain numbers


def test_plain_numbers_parse_and_an_empty_cell_is_blank_without_comment():
    values, issues = parse_numbers(text("1.5", " 2 ", "", "-3e2"), "From_m", RULES)
    assert values.iloc[[0, 1, 3]].tolist() == [1.5, 2.0, -300.0]
    assert math.isnan(values.iloc[2])
    assert issues == []


def test_text_that_will_not_parse_is_an_error_naming_column_values_and_lines():
    values, issues = parse_numbers(text("1.0", "NS", "2.0", "NS", "n/a"), "From_m", RULES)
    issue = only(issues, "value_unparseable")
    assert issue.severity is Severity.ERROR
    assert issue.count == 3
    assert "From_m" in issue.message and "assay.csv" in issue.message
    assert "'NS' x2" in issue.message and "'n/a' x1" in issue.message
    # data rows 1, 3 and 4 are file lines 3, 5 and 6 under a header line
    assert "file lines 3, 5, 6" in issue.message
    assert "sources.assay.null_values" in issue.message
    assert values.isna().tolist() == [False, True, False, True, True]


@pytest.mark.parametrize("token", ["-", "NA", "na", "nan", "inf"])
def test_no_token_other_than_an_empty_cell_is_blank_unless_declared(token):
    """These used to be read as blank silently. Now the user says what they mean."""
    _, issues = parse_numbers(text("1.0", token), "From_m", RULES)
    assert only(issues, "value_unparseable").severity is Severity.ERROR


def test_declared_null_text_reads_as_blank_and_is_counted():
    rules = NumberRules("assay", "assay.csv", null_values=("NS", "IS"))
    values, issues = parse_numbers(text("1.0", "ns", "IS", " NS "), "From_m", rules)
    assert values.isna().tolist() == [False, True, True, True]
    issue = only(issues, "null_value")
    assert issue.severity is Severity.INFO
    assert issue.count == 3


def test_a_numeric_null_token_matches_by_value():
    rules = NumberRules("assay", "assay.csv", null_values=("-99",))
    values, issues = parse_numbers(text("-99", "-99.0", "-99.00", "-9.9"), "From_m", rules)
    assert values.isna().tolist() == [True, True, True, False]
    assert only(issues, "null_value").count == 3


def test_below_detection_text_outside_a_grade_column_is_an_error():
    _, issues = parse_numbers(text("<0.5"), "BD_tonnes_m3", RULES)
    assert only(issues, "value_unparseable").severity is Severity.ERROR


# ------------------------------------------------------------------------ grades


def test_below_detection_reads_as_half_the_limit():
    values, issues = parse_grades(text("<0.01", "< 0.02", "<1e-3", "0.5"), "Zn_pct", RULES)
    assert values.tolist() == pytest.approx([0.005, 0.01, 0.0005, 0.5])
    issue = only(issues, "below_detection")
    assert issue.severity is Severity.INFO
    assert issue.count == 3


def test_a_negative_grade_is_an_error_and_loads_as_blank():
    """-99 is a sentinel in one database and -0.01 a detection limit in another."""
    values, issues = parse_grades(text("1.0", "-99", "-0.01"), "Zn_pct", RULES)
    issue = only(issues, "grade_negative")
    assert issue.severity is Severity.ERROR
    assert issue.count == 2
    assert "'-99' x1" in issue.message
    assert "negative_is_below_detection" in issue.message
    assert values.isna().tolist() == [False, True, True]


def test_negative_as_detection_limit_reads_as_half_the_limit_when_declared():
    rules = NumberRules("assay", "assay.csv", negative_is_below_detection=True)
    values, issues = parse_grades(text("1.0", "-0.01"), "Zn_pct", rules)
    assert values.tolist() == pytest.approx([1.0, 0.005])
    assert only(issues, "below_detection").severity is Severity.INFO


def test_a_declared_sentinel_is_blank_even_when_negatives_are_detection_limits():
    rules = NumberRules(
        "assay", "assay.csv", null_values=("-99",), negative_is_below_detection=True
    )
    values, issues = parse_grades(text("-99", "-0.01"), "Zn_pct", rules)
    assert math.isnan(values.iloc[0])
    assert values.iloc[1] == pytest.approx(0.005)
    assert {i.check for i in issues} == {"null_value", "below_detection"}


# ------------------------------------------------------------------------ config


def test_null_values_written_as_yaml_numbers_are_kept_as_text():
    src = AssaySource(path="a.csv", density={"fallback_constant": 2.7}, null_values=[-99, "NS"])
    assert src.null_values == ["-99", "NS"]


# ----------------------------------------------------------------------- readers


@pytest.fixture
def project(tmp_path):
    """A private copy of the fixture project, so its files can be edited."""
    return write_fixture_project(tmp_path)


def edit(path, column, rows, value):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.loc[rows, column] = value
    frame.to_csv(path, index=False)


def test_the_assay_reader_reports_sentinels_and_detection_limits_in_user_terms(project):
    data = project.parent / "data"
    edit(data / "assay.csv", "Zn_pct", [0, 1], "<0.01")
    edit(data / "assay.csv", "Zn_pct", [2], "-99")
    cfg = load_config(project)

    assay, issues = read_assay(cfg)
    zn = assay[M.elem_col("Zn")]
    assert zn.iloc[0] == pytest.approx(0.005 * 10_000)  # half the limit, then pct -> ppm
    assert "Zn_pct" in only(issues, "grade_negative").message

    cfg.sources.assay.null_values = ["-99"]
    _, issues = read_assay(cfg)
    assert not [i for i in issues if i.severity is Severity.ERROR]
    assert only(issues, "null_value").count == 1


def test_an_unparseable_block_dimension_names_the_users_column(project):
    dx = ORIGINAL["b_dx"]
    edit(project.parent / "data" / "block_model.csv", dx, [0], "?")
    _, issues = read_block_model(load_config(project))
    issue = only(issues, "value_unparseable")
    assert f"values in {dx} " in issue.message


def test_an_unparseable_value_stops_the_run(project):
    edit(project.parent / "data" / "assay.csv", "From_m", [3], "NS")
    with pytest.raises(ValidationFailedError, match="value_unparseable"):
        run_pipeline(load_config(project))


def test_the_fixture_project_parses_without_a_numeric_issue(cfg):
    _, issues = load_all(cfg)
    numeric = {"value_unparseable", "grade_negative", "null_value", "below_detection"}
    assert not [i for i in issues if i.check in numeric]
