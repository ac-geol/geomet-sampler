"""Core used by earlier testwork must not be recommended again."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from geomet_sampler import models as M
from geomet_sampler.allocate.targets import mark_prior_testwork
from geomet_sampler.composite.candidates import INTERVAL_IDS
from geomet_sampler.config import load_config
from geomet_sampler.models import Severity
from geomet_sampler.pipeline import run_pipeline
from geomet_sampler.segment.runs import BREAK_AFTER_UNUSABLE, BREAK_PRIOR_TESTWORK, segment_runs

from .fixtures import write_fixture_project


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            M.HOLE_ID: ["H1"] * 5,
            M.FROM_M: [0.0, 1.0, 2.0, 3.0, 4.0],
            M.TO_M: [1.0, 2.0, 3.0, 4.0, 5.0],
            M.elem_col("Zn"): [1.0] * 5,
            M.OUTSIDE_MODEL: [False] * 5,
        }
    )


def _testwork(**overrides) -> pd.DataFrame:
    rows = {
        M.SAMPLE_ID: ["TW1"],
        M.HOLE_ID: ["H1"],
        M.FROM_M: [1.5],
        M.TO_M: [3.0],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- marking


def test_every_interval_the_record_overlaps_is_flagged():
    out, issues = mark_prior_testwork(_intervals(), _testwork(), policy="exclude")
    assert out[M.PRIOR_TESTWORK].tolist() == [False, True, True, False, False]
    assert out[M.PRIOR_TESTWORK_IDS].tolist()[1] == ("TW1",)
    assert [i.check for i in issues] == ["prior_testwork_excluded"]


def test_a_record_that_only_touches_an_interval_does_not_flag_it():
    out, _ = mark_prior_testwork(_intervals(), _testwork(**{M.FROM_M: [2.0]}), policy="exclude")
    assert out[M.PRIOR_TESTWORK].tolist() == [False, False, True, False, False]


def test_available_policy_leaves_the_core_in_play():
    out, issues = mark_prior_testwork(_intervals(), _testwork(), policy="available")
    assert not out[M.PRIOR_TESTWORK].any()
    assert issues == []


def test_a_record_without_depths_is_reported_not_guessed():
    record = _testwork(**{M.FROM_M: [float("nan")], M.TO_M: [float("nan")]})
    out, issues = mark_prior_testwork(_intervals(), record, policy="exclude")
    assert not out[M.PRIOR_TESTWORK].any()
    [issue] = issues
    assert issue.check == "prior_testwork_unlocated"
    assert issue.severity is Severity.WARN


def test_no_testwork_file_flags_nothing():
    out, issues = mark_prior_testwork(_intervals(), None, policy="exclude")
    assert not out[M.PRIOR_TESTWORK].any() and issues == []


# ----------------------------------------------------------------- segmentation


def test_consumed_core_breaks_the_run_on_both_sides():
    """A composite must not bridge the gap earlier testwork left in the tray."""
    flagged, _ = mark_prior_testwork(_intervals(), _testwork(), policy="exclude")
    runs = segment_runs(
        flagged,
        hard_break_on=[],
        max_interval_gap_m=0.1,
        break_on_low_recovery=True,
        break_on_missing_primary_grade=True,
        primary_element="Zn",
    )
    assert runs["break_reason"].tolist()[1] == BREAK_PRIOR_TESTWORK
    assert runs["break_reason"].tolist()[3] == BREAK_AFTER_UNUSABLE
    assert runs["usable"].tolist() == [True, False, False, True, True]
    assert runs[M.RUN_ID].iloc[0] != runs[M.RUN_ID].iloc[3]


# ------------------------------------------------------------------ end to end


def _project_with_testwork(root, policy: str):
    config = write_fixture_project(root)
    # the whole upper, high-grade unit of two holes was used for earlier testwork
    pd.DataFrame(
        {
            "sample_id": ["TW-A", "TW-B"],
            "hole_id": ["DDH001", "DDH002"],
            "from_m": [0.0, 0.0],
            "to_m": [30.0, 30.0],
        }
    ).to_csv(root / "data" / "testwork.csv", index=False)
    raw = yaml.safe_load(config.read_text())
    raw["sources"]["existing_testwork"] = {
        "path": "data/testwork.csv",
        "columns": {
            "sample_id": "sample_id",
            "hole_id": "hole_id",
            "from_m": "from_m",
            "to_m": "to_m",
        },
    }
    raw["mass"]["prior_testwork_core"] = policy
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(config)


def test_no_candidate_or_pick_uses_consumed_core(tmp_path):
    state = run_pipeline(_project_with_testwork(tmp_path, "exclude"))
    consumed = set(state.intervals.loc[state.intervals[M.PRIOR_TESTWORK], M.INTERVAL_ID])
    assert len(consumed) == 60  # 30 x 1 m intervals in each of the two holes
    for ids in state.candidates[INTERVAL_IDS]:
        assert not consumed & set(ids)
    assert not consumed & set(state.picks[M.INTERVAL_ID])

    [gap] = [g for g in state.gaps if g.key == "prior_testwork"]
    assert gap.detail["n_intervals"] == 60
    assert gap.detail["testwork_samples"] == "TW-A, TW-B"


def test_the_available_policy_keeps_consumed_core_as_candidates(tmp_path):
    state = run_pipeline(_project_with_testwork(tmp_path, "available"))
    upper = state.intervals[
        state.intervals[M.HOLE_ID].isin(["DDH001", "DDH002"]) & (state.intervals[M.TO_M] <= 30)
    ]
    used = set(upper[M.INTERVAL_ID])
    assert any(used & set(ids) for ids in state.candidates[INTERVAL_IDS])


@pytest.mark.parametrize("policy", ["exclude", "available"])
def test_earlier_testwork_still_reduces_targets_under_either_policy(tmp_path, policy):
    state = run_pipeline(_project_with_testwork(tmp_path, policy))
    assert state.allocation["existing"].sum() == 2
