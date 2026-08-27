"""Charts for the run.

Matplotlib only, written straight to PNG. No interactive backend, so this is safe to
call from a CLI or a scheduled job.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import models as M  # noqa: E402
from ..allocate.targets import DEFICIT  # noqa: E402
from ..config import Config  # noqa: E402
from ..domains import is_scheduled, period_sort_key  # noqa: E402
from ..pipeline import PipelineState  # noqa: E402
from ..units import ppm_to_units  # noqa: E402

DPI = 140


def write_plots(state: PipelineState) -> list[Path]:
    """Write every chart that has data behind it."""
    out_dir = state.cfg.project.output_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, draw in (
        ("allocation_target_vs_achieved", _allocation_chart),
        ("plan_view_selected", _plan_view),
        ("long_section_period", _long_section),
        ("grade_distribution", _grade_distribution),
    ):
        figure = draw(state)
        if figure is None:
            continue
        path = out_dir / f"{name}.png"
        figure.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def _allocation_chart(state: PipelineState):
    allocation = state.allocation
    if allocation is None or allocation.empty:
        return None
    frame = allocation[allocation[DEFICIT] > 0].copy()
    if frame.empty:
        return None
    achieved = frame.get("achieved", pd.Series(0, index=frame.index))

    y = np.arange(len(frame))
    figure, axes = plt.subplots(figsize=(9, 0.42 * len(frame) + 2))
    axes.barh(y + 0.2, frame[DEFICIT], height=0.38, label="deficit (target)", color="#9DB4D6")
    axes.barh(y - 0.2, achieved, height=0.38, label="achieved", color="#1F3864")
    axes.set_yticks(y, frame[M.ALLOCATION_DOMAIN])
    axes.invert_yaxis()
    axes.set_xlabel("samples")
    axes.set_title("Allocation: deficit vs achieved by domain")
    axes.legend(loc="lower right")
    axes.grid(axis="x", alpha=0.3)
    return figure


def _plan_view(state: PipelineState):
    selected, intervals = state.selected, state.intervals
    if selected is None or selected.empty:
        return None
    figure, axes = plt.subplots(figsize=(8, 7))
    if intervals is not None and not intervals.empty:
        axes.scatter(
            intervals[M.X_MID], intervals[M.Y_MID], s=1, c="#D8D8D8", label="all intervals"
        )
    for domain, group in selected.groupby(M.ALLOCATION_DOMAIN):
        axes.scatter(group[M.X_MID], group[M.Y_MID], s=48, label=str(domain), edgecolor="k")
    axes.set_xlabel("east")
    axes.set_ylabel("north")
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_title("Plan view of selected composites")
    axes.legend(fontsize=7, loc="best", markerscale=0.8)
    return figure


def _long_section(state: PipelineState):
    selected, intervals = state.selected, state.intervals
    if selected is None or selected.empty:
        return None
    figure, axes = plt.subplots(figsize=(9, 6))
    if intervals is not None and not intervals.empty:
        axes.scatter(intervals[M.X_MID], intervals[M.Z_MID], s=1, c="#E0E0E0")
    periods = sorted(selected[M.PERIOD].dropna().unique(), key=period_sort_key)
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, max(len(periods), 1)))
    for colour, period in zip(colours, periods, strict=False):
        group = selected[selected[M.PERIOD] == period]
        axes.scatter(
            group[M.X_MID],
            group[M.Z_MID],
            s=48,
            color=colour,
            edgecolor="k",
            label=f"period {period}",
        )
    axes.set_xlabel("east")
    axes.set_ylabel("RL")
    axes.set_title("Long section of selected composites, coloured by period")
    axes.legend(fontsize=8)
    return figure


def _grade_distribution(state: PipelineState):
    """Mine plan grade against selected sample grade, per domain."""
    cfg: Config = state.cfg
    selected, blocks = state.selected, state.blocks
    element = cfg.primary_element
    block_col, sample_col = M.bm_elem_col(element), M.wtd_elem_col(element)
    if selected is None or selected.empty or sample_col not in selected.columns:
        return None
    if blocks is None or block_col not in blocks.columns:
        return None

    units = cfg.elements[element].block_model.units or cfg.elements[element].drillhole.units
    scheduled = blocks[is_scheduled(blocks[M.PERIOD], cfg.block_model_options.excluded_periods)]
    domains = list(selected[M.ALLOCATION_DOMAIN].dropna().unique())[:8]
    if not domains:
        return None

    figure, axes = plt.subplots(figsize=(9, 0.5 * len(domains) + 2.5))
    for i, domain in enumerate(domains):
        plan = ppm_to_units(
            pd.to_numeric(
                scheduled.loc[scheduled[M.ALLOCATION_DOMAIN] == domain, block_col], errors="coerce"
            ).dropna(),
            units,
        )
        picked = ppm_to_units(
            pd.to_numeric(
                selected.loc[selected[M.ALLOCATION_DOMAIN] == domain, sample_col], errors="coerce"
            ).dropna(),
            units,
        )
        if len(plan):
            axes.scatter(plan, np.full(len(plan), i + 0.16), s=4, c="#9DB4D6", alpha=0.35)
        if len(picked):
            axes.scatter(
                picked, np.full(len(picked), i - 0.16), s=40, c="#1F3864", edgecolor="k", zorder=3
            )
    axes.set_yticks(range(len(domains)), domains, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel(f"{element} ({units.value})")
    axes.set_title("Grade distribution: mine plan (light) vs selected samples (dark)")
    axes.grid(axis="x", alpha=0.3)
    return figure
