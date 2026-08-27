"""Stage orchestration.

All logic lives in the stage functions below and the modules they call. The CLI reads a
config, calls one of these, and writes the result. A UI would be a third caller of the
same functions.

Each stage caches its result under the output directory, keyed by the config hash, so a
later stage can be rerun without repeating the expensive geometry and candidate work. A
config change changes the hash and invalidates the cache, which is the point: a cached
result must never outlive the config that produced it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import models as M
from .allocate.targets import compute_targets, existing_coverage, tonnage_by_period, unmet_gaps
from .blockmodel.assign import assign_blocks, record_domain_match, resolve_attributes
from .composite.candidates import generate_candidates, pick_list
from .config import Config
from .desurvey import build_traces, desurvey_intervals
from .domains import label_blocks, label_intervals
from .errors import ValidationFailedError
from .intervals.merge import build_framework, discretise_planned
from .io.readers import Dataset, load_all
from .mass.estimate import estimate_interval_mass
from .models import GapEntry, Issue, Severity, ValidationReport
from .segment.runs import segment_runs, short_run_gaps, summarise_runs
from .select.greedy import COMPOSITE_ID, achieved_counts, check_no_interval_reused, select
from .select.objective import components
from .validate.checks import check_model_extent, check_toe_above_collar, validate_inputs

CACHE_DIRNAME = ".cache"


@dataclass
class PipelineState:
    """Everything produced so far. Stages add to it; nothing is removed."""

    cfg: Config
    dataset: Dataset | None = None
    traces: dict = field(default_factory=dict)
    intervals: pd.DataFrame | None = None
    blocks: pd.DataFrame | None = None
    runs: pd.DataFrame | None = None
    candidates: pd.DataFrame | None = None
    allocation: pd.DataFrame | None = None
    selected: pd.DataFrame | None = None
    scored: pd.DataFrame | None = None
    picks: pd.DataFrame | None = None
    report: ValidationReport = field(default_factory=ValidationReport)
    gaps: list[GapEntry] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------- caching


def cache_dir(cfg: Config) -> Path:
    path = cfg.project.output_dir / CACHE_DIRNAME / cfg.config_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_write(cfg: Config, name: str, payload: Any) -> None:
    (cache_dir(cfg) / f"{name}.pkl").write_bytes(pickle.dumps(payload))


def cache_read(cfg: Config, name: str) -> Any | None:
    path = cache_dir(cfg) / f"{name}.pkl"
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


# -------------------------------------------------------------------- stages


def stage_load(cfg: Config) -> PipelineState:
    """Read every source and run the checks that do not need geometry."""
    state = PipelineState(cfg=cfg)
    dataset, issues = load_all(cfg)
    state.dataset = dataset
    state.report.extend(issues)
    state.report.extend(validate_inputs(dataset, cfg))
    return state


def stage_geometry(state: PipelineState) -> PipelineState:
    """Desurvey, build the interval framework, and attach block model attributes."""
    cfg, dataset = state.cfg, state.dataset

    traces, issues = build_traces(
        dataset.collar, dataset.survey, fill_collar_survey=cfg.desurvey.fill_collar_survey
    )
    state.report.extend(issues)
    state.report.extend(check_toe_above_collar(traces, dataset.collar))

    framework, issues = build_framework(dataset.assay, dataset.litho, dataset.domain_lookup)
    state.report.extend(issues)
    framework[M.IS_PLANNED] = False

    planned = _planned_intervals(state, traces)
    if planned is not None and not planned.empty:
        framework = pd.concat([framework, planned], ignore_index=True)

    framework, issues = desurvey_intervals(framework, traces)
    state.report.extend(issues)

    blocks = label_blocks(dataset.block_model, cfg)
    framework, issues = assign_blocks(
        framework,
        blocks,
        roles=list(cfg.attributes),
        elements=cfg.elements_with_block_model(),
    )
    state.report.extend(issues)
    state.report.extend(
        check_model_extent(
            framework, error_fraction=cfg.block_model_options.off_model_error_fraction
        )
    )

    framework = record_domain_match(
        framework,
        list(cfg.attributes),
        domain_match_attribute=cfg.domaining.domain_match_attribute,
    )
    framework = resolve_attributes(framework, list(cfg.attributes))
    framework = label_intervals(framework, cfg)

    framework = estimate_interval_mass(
        framework,
        dataset.availability,
        default_core_diameter_mm=cfg.mass.default_core_diameter_mm,
        default_remaining_fraction=cfg.mass.default_remaining_fraction,
        loss_factor=cfg.mass.loss_factor,
        default_density_t_m3=cfg.sources.assay.density.fallback_constant,
    )

    state.traces = traces
    state.intervals = framework
    state.blocks = blocks
    return state


def _planned_intervals(state: PipelineState, traces: dict) -> pd.DataFrame | None:
    """Planned holes have no assays, so their traces are discretised instead."""
    cfg, dataset = state.cfg, state.dataset
    if dataset.planned_collar is None or dataset.planned_survey is None:
        return None
    planned_traces, issues = build_traces(
        dataset.planned_collar,
        dataset.planned_survey,
        fill_collar_survey=cfg.desurvey.fill_collar_survey,
    )
    state.report.extend(issues)
    traces.update(planned_traces)
    return discretise_planned(
        planned_traces,
        list(planned_traces),
        cfg.block_model_options.planned_discretisation_m,
    )


def stage_candidates(state: PipelineState) -> PipelineState:
    """Segment into runs, then generate candidate composites within them."""
    cfg = state.cfg
    intervals = segment_runs(
        state.intervals,
        hard_break_on=cfg.domaining.hard_break_on,
        max_interval_gap_m=cfg.domaining.max_interval_gap_m,
        break_on_low_recovery=cfg.domaining.break_on_low_recovery,
        break_on_missing_primary_grade=cfg.domaining.break_on_missing_primary_grade,
        primary_element=cfg.primary_element,
    )
    state.intervals = intervals
    state.runs = summarise_runs(intervals, cfg.domaining.hard_break_on)
    state.gaps += short_run_gaps(
        state.runs,
        min_length_m=cfg.compositing.min_length_m,
        min_mass_kg=cfg.compositing.min_mass_kg,
    )

    candidates, gaps = generate_candidates(intervals, cfg)
    state.candidates = candidates
    state.gaps += gaps
    return state


def stage_allocate(state: PipelineState) -> PipelineState:
    """Turn scheduled tonnage into per-domain sample deficits."""
    cfg = state.cfg
    existing, issues = existing_coverage(state.dataset.existing_testwork, state.intervals)
    state.report.extend(issues)
    allocation, issues = compute_targets(state.blocks, cfg, existing)
    state.report.extend(issues)
    state.allocation = allocation
    return state


def stage_select(state: PipelineState) -> PipelineState:
    """Choose the composites and build the pick list."""
    cfg = state.cfg
    selected, scored, issues = select(state.candidates, state.allocation, cfg)
    state.report.extend(issues)
    state.report.extend(check_no_interval_reused(selected))

    achieved = achieved_counts(selected)
    state.allocation["achieved"] = (
        state.allocation[M.ALLOCATION_DOMAIN].map(achieved).fillna(0).astype(int)
    )
    state.gaps += unmet_gaps(state.allocation, achieved)

    state.selected = selected
    state.scored = scored
    state.picks = pick_list(selected, state.intervals, COMPOSITE_ID)
    state.summary = _summarise(state)
    return state


def _summarise(state: PipelineState) -> dict[str, Any]:
    cfg = state.cfg
    selected = state.selected
    summary: dict[str, Any] = {
        "project": cfg.project.name,
        "config": str(cfg.config_path),
        "config_hash": cfg.config_hash,
        "holes_in_collar": len(state.dataset.collar),
        "holes_desurveyed": len(state.traces),
        "framework_intervals": len(state.intervals),
        "intervals_outside_model": int(state.intervals[M.OUTSIDE_MODEL].sum()),
        "runs": 0 if state.runs is None else len(state.runs),
        "candidates": len(state.candidates),
        "allocation_method": cfg.allocation.method,
        "samples_requested": cfg.allocation.total_samples,
        "samples_selected": len(selected),
        "pick_list_intervals": 0 if state.picks is None else len(state.picks),
        "errors": len(state.report.errors),
        "warnings": len(state.report.warnings),
    }
    if selected is not None and not selected.empty:
        from .composite.candidates import EST_MASS_KG

        summary["total_mass_kg"] = round(float(selected[EST_MASS_KG].sum()), 1)
        summary["mean_mass_kg"] = round(float(selected[EST_MASS_KG].mean()), 1)
        summary["holes_used"] = int(selected[M.HOLE_ID].nunique())
        summary.update(
            {k: round(v, 3) for k, v in components(selected, state.allocation, cfg).items()}
        )
    for fallback in state.dataset.fallbacks:
        summary.setdefault("fallbacks", []).append(fallback)
    return summary


# ------------------------------------------------------------------ run order

STAGES = ("load", "geometry", "candidates", "allocate", "select")


def run_pipeline(
    cfg: Config,
    *,
    through: str = "select",
    force: bool = False,
    use_cache: bool = False,
) -> PipelineState:
    """Run stages in order up to and including ``through``.

    Validation ERRORs stop the run unless ``force`` is set, because every later stage
    assumes the geometry and the domains are trustworthy.
    """
    if through not in STAGES:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown stage {through!r}, expected one of {STAGES}")

    if use_cache:
        cached = cache_read(cfg, through)
        if cached is not None:
            return cached

    state = stage_load(cfg)
    _halt_on_errors(state, force)
    if through == "load":
        cache_write(cfg, "load", state)
        return state

    state = stage_geometry(state)
    _halt_on_errors(state, force)
    cache_write(cfg, "geometry", state)
    if through == "geometry":
        return state

    state = stage_candidates(state)
    cache_write(cfg, "candidates", state)
    if through == "candidates":
        return state

    state = stage_allocate(state)
    cache_write(cfg, "allocate", state)
    if through == "allocate":
        return state

    state = stage_select(state)
    cache_write(cfg, "select", state)
    return state


def _halt_on_errors(state: PipelineState, force: bool) -> None:
    errors = state.report.errors
    if not errors or force:
        return
    lines = [f"  {i.severity.value} [{i.check}] {i.message}" for i in errors[:20]]
    raise ValidationFailedError(
        f"{len(errors)} validation errors:\n"
        + "\n".join(lines)
        + "\n\nFix the data or the config, or rerun with --force to continue anyway."
    )


def gap_frame(gaps: list[GapEntry]) -> pd.DataFrame:
    if not gaps:
        return pd.DataFrame(columns=["scope", "key", "reason"])
    return pd.DataFrame([g.as_row() for g in gaps])


def summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in summary.items():
        rows.append({"item": key, "value": "; ".join(value) if isinstance(value, list) else value})
    return pd.DataFrame(rows)


__all__ = [
    "PipelineState",
    "Severity",
    "gap_frame",
    "run_pipeline",
    "stage_allocate",
    "stage_candidates",
    "stage_geometry",
    "stage_load",
    "stage_select",
    "summary_frame",
    "tonnage_by_period",
    "Issue",
]
