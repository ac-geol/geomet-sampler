"""Grade binning and allocation domain labels.

Both the block model and the drillhole intervals are labelled through these functions,
so a target computed from tonnage and an achieved count from selected composites are
always expressed in the same vocabulary. If they were built separately the two could
drift apart without any error being raised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M
from .config import Config, GradeBins
from .units import grade_to_ppm

#: Label used where an allocation key component has no value.
UNKNOWN = "UNKNOWN"


def bin_edges_ppm(bins: GradeBins, cfg: Config) -> list[float]:
    """Grade bin edges are declared in the element's own units; convert once to ppm."""
    spec = cfg.elements[bins.element]
    side = spec.block_model if bins.source == "block_model" else spec.drillhole
    return list(grade_to_ppm(pd.Series(bins.edges, dtype=float), side.units))


def assign_grade_bin(frame: pd.DataFrame, cfg: Config, *, on: str) -> pd.Series:
    """Label each row with its grade bin.

    ``on`` is ``"block_model"`` or ``"drillhole"`` and names which set of grade columns
    this frame carries, which is not always the same as ``grade_bins.source``: block
    rows can only ever be binned on block grades.
    """
    bins = cfg.domaining.grade_bins
    column = M.bm_elem_col(bins.element) if on == "block_model" else M.elem_col(bins.element)
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    values = pd.to_numeric(frame[column], errors="coerce")
    labelled = pd.cut(
        values,
        bins=bin_edges_ppm(bins, cfg),
        labels=bins.labels,
        include_lowest=True,
        right=False,
    )
    return pd.Series(labelled, index=frame.index).astype("object")


def compose_domain(parts: list[object]) -> str:
    """Join allocation key components into a label. One rule, used everywhere.

    Blocks, intervals and composites all pass through here, so a composite can never end
    up labelled by a rule the targets were not computed with.
    """
    return (
        "-".join(
            UNKNOWN
            if p is None or (isinstance(p, float) and np.isnan(p)) or str(p) == "<NA>"
            else str(p).upper()
            for p in parts
        )
        or UNKNOWN
    )


def allocation_domain(frame: pd.DataFrame, cfg: Config, *, on: str) -> pd.Series:
    """Join the allocation key components into one label, e.g. ``VOLCANIC-OXIDE-HIGH``.

    Attribute roles resolve to block model values on both sides, because allocation
    targets are computed from block model tonnage.
    """
    parts: list[pd.Series] = []
    for key in cfg.domaining.allocation_key:
        if key == M.GRADE_BIN:
            values = frame[M.GRADE_BIN] if M.GRADE_BIN in frame.columns else None
        elif key == M.PERIOD:
            values = frame[M.PERIOD] if M.PERIOD in frame.columns else None
        else:
            column = M.bm_attr_col(key)
            values = frame[column] if column in frame.columns else None
        if values is None:
            values = pd.Series(pd.NA, index=frame.index, dtype="object")
        parts.append(values.astype("string").fillna(UNKNOWN).str.upper())

    if not parts:
        return pd.Series(UNKNOWN, index=frame.index, dtype="object")
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "-" + part
    return joined.astype("object")


def label_blocks(block_model: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add grade bin and allocation domain to the block model."""
    out = block_model.copy()
    out[M.GRADE_BIN] = assign_grade_bin(out, cfg, on="block_model")
    out[M.ALLOCATION_DOMAIN] = allocation_domain(out, cfg, on="block_model")
    return out


def label_intervals(intervals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add grade bin and allocation domain to drillhole intervals.

    The bin is taken from whichever source the user declared. Binning drillhole samples
    on block grades keeps candidates in the same bin vocabulary as the targets, at the
    cost of ignoring the sample's own assay; binning on drillhole grades does the
    reverse. The choice is the user's and is recorded in the config.
    """
    out = intervals.copy()
    source = cfg.domaining.grade_bins.source
    out[M.GRADE_BIN] = assign_grade_bin(out, cfg, on=source)
    out[M.ALLOCATION_DOMAIN] = allocation_domain(out, cfg, on="block_model")
    return out


def is_scheduled(period: pd.Series, excluded: list[str]) -> pd.Series:
    """A block counts toward allocation only if its period is populated and not excluded."""
    text = period.astype("string").str.strip()
    blank = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "null"])
    excluded_set = {str(p).strip().upper() for p in excluded}
    return ~blank & ~text.str.upper().isin(excluded_set)


def period_sort_key(period: object) -> tuple[int, float, str]:
    """Order periods numerically where possible, alphabetically otherwise."""
    if period is None or (isinstance(period, float) and np.isnan(period)):
        return (2, 0.0, "")
    try:
        return (0, float(period), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(period))
