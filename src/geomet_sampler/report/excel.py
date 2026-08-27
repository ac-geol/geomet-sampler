"""Compose the deliverable workbook and the CSV set.

Sheet order is the order a reviewer reads them: what was run, what was asked for, what
was chosen, what to physically pull, what could not be covered, and what the data looked
like on the way in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import models as M
from ..allocate.targets import DEFICIT, EXISTING, TARGET, TONNES, TONNES_PCT, WEIGHTED_PCT
from ..composite.candidates import CV_PRIMARY, EST_MASS_KG, N_INTERVALS
from ..config import Config
from ..io.writers import present, write_csv, write_workbook
from ..pipeline import PipelineState, gap_frame, summary_frame
from ..select.greedy import COMPOSITE_ID, REASON, SCORE, SELECTION_ORDER
from ..validate.checks import report_frame

WORKBOOK_NAME = "picks.xlsx"


def build_sheets(state: PipelineState) -> dict[str, pd.DataFrame]:
    """The workbook contents, already in reporting units and readable headers."""
    cfg = state.cfg
    return {
        "Summary": summary_frame(state.summary),
        "Allocation": _allocation_sheet(state, cfg),
        "Composites": _composites_sheet(state, cfg),
        "Pick_List": _pick_list_sheet(state, cfg),
        "Gap_Register": gap_frame(state.gaps),
        "Validation": report_frame(state.report.issues),
        "Candidates": present(state.scored, cfg) if state.scored is not None else pd.DataFrame(),
    }


def _allocation_sheet(state: PipelineState, cfg: Config) -> pd.DataFrame:
    if state.allocation is None or state.allocation.empty:
        return pd.DataFrame()
    columns = [
        M.ALLOCATION_DOMAIN,
        TONNES,
        TONNES_PCT,
        WEIGHTED_PCT,
        TARGET,
        EXISTING,
        DEFICIT,
        "achieved",
    ]
    frame = state.allocation[[c for c in columns if c in state.allocation.columns]].copy()
    if "achieved" in frame.columns:
        frame["shortfall"] = frame[DEFICIT] - frame["achieved"]
    return present(frame, cfg, round_to=2)


def _composites_sheet(state: PipelineState, cfg: Config) -> pd.DataFrame:
    if state.selected is None or state.selected.empty:
        return pd.DataFrame()
    drop = [SCORE, SELECTION_ORDER, "quality"]
    frame = state.selected.drop(columns=[c for c in drop if c in state.selected.columns])
    lead = [
        COMPOSITE_ID,
        M.HOLE_ID,
        M.FROM_M,
        M.TO_M,
        M.LENGTH_M,
        N_INTERVALS,
        EST_MASS_KG,
        M.ALLOCATION_DOMAIN,
        M.GEOMET_DOMAIN,
        M.PERIOD,
        M.GRADE_BIN,
        CV_PRIMARY,
        M.DOMAIN_MATCH,
        M.IS_PLANNED,
    ]
    ordered = [c for c in lead if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered and c != REASON]
    ordered += [REASON] if REASON in frame.columns else []
    return present(frame[ordered], cfg, round_to=2)


def _pick_list_sheet(state: PipelineState, cfg: Config) -> pd.DataFrame:
    """The core shed worksheet: one row per interval to physically cut."""
    if state.picks is None or state.picks.empty:
        return pd.DataFrame()
    return present(state.picks, cfg, round_to=2)


def write_outputs(state: PipelineState) -> dict[str, Path]:
    """Write every configured output format. Returns what was written, by name."""
    cfg = state.cfg
    out_dir = cfg.project.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = build_sheets(state)
    written: dict[str, Path] = {}

    if "xlsx" in cfg.reporting.formats:
        written["workbook"] = write_workbook(sheets, out_dir / WORKBOOK_NAME)
    if "csv" in cfg.reporting.formats:
        for name, frame in sheets.items():
            if name == "Gap_Register" and not cfg.reporting.include_gap_register:
                continue
            written[name.lower()] = write_csv(frame, out_dir / f"{name.lower()}.csv")
    return written


def write_validation_report(state: PipelineState) -> Path:
    """The validation report is written even when the run stops on errors."""
    path = state.cfg.project.output_dir / "validation_report.csv"
    return write_csv(report_frame(state.report.issues), path)
