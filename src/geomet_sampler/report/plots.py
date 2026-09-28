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
from ..validate.checks import HOLE_ORIENTATION_PLOT  # noqa: E402

DPI = 140
DOWN_COLOUR = "#3B6EA8"
UP_COLOUR = "#D1495B"
#: Above this many upward holes, labelling each one would make the view unreadable.
MAX_HOLE_LABELS = 40


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


def write_hole_orientation(state: PipelineState) -> Path | None:
    """Plan and two sections of every hole's collar and toe, upward holes highlighted.

    Written on every run that builds geometry, whatever ``include_plots`` says: a hole
    loaded upside down or in the wrong place corrupts everything downstream, and looking
    at it is the quickest way to catch that.
    """
    ends = state.hole_ends
    if ends is None or ends.empty:
        return None
    path = state.cfg.project.output_dir / HOLE_ORIENTATION_PLOT
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _hole_orientation(ends)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def orientation_notice(state: PipelineState, path: Path | None) -> str:
    """The reminder printed after a run, so the section view is not left unopened."""
    ends = state.hole_ends
    if ends is None or ends.empty or path is None:
        return ""
    rising = int(ends[M.RISES].sum())
    return (
        f"IMPORTANT: check hole orientation before using these results -> {path}\n"
        f"  {len(ends)} holes, {rising} end above their collar (red). Confirm collars, toes "
        "and upward holes are where you expect.\n"
        "  A wrong dip convention, inverted Z or unit mismatch shows up here first, and "
        "silently corrupts every result after it."
    )


def _hole_orientation(ends: pd.DataFrame):
    cx, cy, cz = (ends[c].to_numpy(float) for c in M.COLLAR_XYZ)
    tx, ty, tz = (ends[c].to_numpy(float) for c in M.TOE_XYZ)
    rises = ends[M.RISES].to_numpy(bool)
    views = (
        ("Plan", (cx, cy), (tx, ty), ("east", "north")),
        ("Section looking north", (cx, cz), (tx, tz), ("east", "RL")),
        ("Section looking east", (cy, cz), (ty, tz), ("north", "RL")),
    )
    figure, axes_row = plt.subplots(1, 3, figsize=(16, 6))
    for axes, (title, (ca, cb), (ta, tb), (xlabel, ylabel)) in zip(axes_row, views, strict=True):
        for up in (False, True):
            mask = rises == up
            if not mask.any():
                continue
            colour = UP_COLOUR if up else DOWN_COLOUR
            axes.plot(
                np.vstack([ca[mask], ta[mask]]),
                np.vstack([cb[mask], tb[mask]]),
                color=colour,
                linewidth=0.9,
                alpha=0.8,
            )
            axes.scatter(ta[mask], tb[mask], s=18, color=colour, edgecolor="k", linewidth=0.4)
        axes.scatter(ca, cb, s=26, marker="s", color="white", edgecolor="k", linewidth=0.8)
        if 0 < rises.sum() <= MAX_HOLE_LABELS:
            for a, b, label in zip(ta[rises], tb[rises], ends.loc[rises, M.HOLE_ID], strict=True):
                axes.annotate(
                    str(label),
                    (a, b),
                    fontsize=6,
                    color=UP_COLOUR,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.25)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="k",
            label="collar",
        ),
        plt.Line2D(
            [], [], marker="o", color=DOWN_COLOUR, markeredgecolor="k", label="toe, hole goes down"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=UP_COLOUR,
            markeredgecolor="k",
            label="toe, hole ends above collar",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "HOLE ORIENTATION: confirm before using any result\n"
        f"{len(ends)} holes, {int(rises.sum())} end above their collar (red). Straight lines "
        "join collar to toe; sections are drawn without vertical exaggeration.",
        fontsize=11,
    )
    figure.subplots_adjust(bottom=0.14)
    return figure


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
