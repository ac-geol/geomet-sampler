"""Segment holes into domain-coherent runs.

A run is a stretch of contiguous intervals within which no hard break occurs. Composites
are generated only inside a run, never across one. That is the structural guarantee that
a composite cannot span a lithology contact, a weathering front, a core gap or a hole
boundary: the boundary is removed from the generator's reach rather than filtered out of
its results afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..models import GapEntry

#: Reasons a new run starts. Written to the run table so any break is explainable.
BREAK_HOLE = "hole"
BREAK_ATTRIBUTE = "attribute"
BREAK_GAP = "gap"
BREAK_LOW_RECOVERY = "low_recovery"
BREAK_MISSING_GRADE = "missing_primary_grade"
BREAK_OUTSIDE_MODEL = "outside_model"
BREAK_UNAVAILABLE = "unavailable"


def segment_runs(
    intervals: pd.DataFrame,
    *,
    hard_break_on: list[str],
    max_interval_gap_m: float,
    break_on_low_recovery: bool,
    break_on_missing_primary_grade: bool,
    primary_element: str,
) -> pd.DataFrame:
    """Assign a run id to every interval, plus the reason its run started.

    ``hard_break_on`` holds domaining key names; they are resolved to columns through
    :func:`geomet_sampler.models.domain_key_col`, so user-invented roles work unchanged.
    """
    out = intervals.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True)
    if out.empty:
        out[M.RUN_ID] = pd.Series(dtype=object)
        out["break_reason"] = pd.Series(dtype=object)
        return out

    n = len(out)
    reason = np.full(n, "", dtype=object)
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    reason[0] = BREAK_HOLE

    hole = out[M.HOLE_ID].to_numpy(object)
    new_hole = np.zeros(n, dtype=bool)
    new_hole[1:] = hole[1:] != hole[:-1]
    _mark(starts, reason, new_hole, BREAK_HOLE)

    from_m = out[M.FROM_M].to_numpy(float)
    to_m = out[M.TO_M].to_numpy(float)
    gap = np.zeros(n, dtype=bool)
    gap[1:] = (from_m[1:] - to_m[:-1]) > max_interval_gap_m
    _mark(starts, reason, gap & ~new_hole, BREAK_GAP)

    for key in hard_break_on:
        column = M.domain_key_col(key)
        if column not in out.columns:
            continue
        values = out[column].astype("string").fillna("\x00__missing__").to_numpy(object)
        changed = np.zeros(n, dtype=bool)
        changed[1:] = values[1:] != values[:-1]
        _mark(starts, reason, changed & ~new_hole, f"{BREAK_ATTRIBUTE}:{key}")

    # An interval that is itself unusable both ends the previous run and starts its own,
    # so no composite can be built across it.
    unusable = pd.Series(False, index=out.index)
    if break_on_low_recovery and M.LOW_RECOVERY in out.columns:
        flagged = out[M.LOW_RECOVERY].fillna(False).astype(bool)
        _mark_series(starts, reason, flagged, BREAK_LOW_RECOVERY)
        unusable |= flagged
    if break_on_missing_primary_grade:
        column = M.elem_col(primary_element)
        missing = out[column].isna() if column in out.columns else pd.Series(True, index=out.index)
        _mark_series(starts, reason, missing, BREAK_MISSING_GRADE)
        unusable |= missing
    if M.OUTSIDE_MODEL in out.columns:
        off = out[M.OUTSIDE_MODEL].fillna(True).astype(bool)
        _mark_series(starts, reason, off, BREAK_OUTSIDE_MODEL)
        unusable |= off

    # the interval after an unusable one also starts fresh
    follows_unusable = np.zeros(n, dtype=bool)
    follows_unusable[1:] = unusable.to_numpy()[:-1]
    _mark(starts, reason, follows_unusable & ~new_hole & ~starts, "after_" + BREAK_LOW_RECOVERY)

    run_index = np.cumsum(starts) - 1
    out[M.RUN_ID] = [f"R{i:06d}" for i in run_index]
    out["break_reason"] = reason
    out["usable"] = ~unusable.to_numpy()
    return out


def _mark(starts: np.ndarray, reason: np.ndarray, mask: np.ndarray, label: str) -> None:
    fresh = mask & ~starts
    starts |= mask
    reason[fresh] = label


def _mark_series(starts: np.ndarray, reason: np.ndarray, mask: pd.Series, label: str) -> None:
    _mark(starts, reason, mask.to_numpy(dtype=bool), label)


def summarise_runs(intervals: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per run: extent, length, mass and the domaining values it carries."""
    if intervals.empty:
        return pd.DataFrame()

    columns = [M.domain_key_col(k) for k in keys]
    columns = [c for c in columns if c in intervals.columns]
    agg: dict[str, tuple[str, str]] = {
        M.HOLE_ID: (M.HOLE_ID, "first"),
        M.FROM_M: (M.FROM_M, "min"),
        M.TO_M: (M.TO_M, "max"),
        "n_intervals": (M.INTERVAL_ID, "size"),
        M.LENGTH_M: (M.LENGTH_M, "sum"),
        "break_reason": ("break_reason", "first"),
        "usable": ("usable", "all"),
    }
    if M.MASS_KG in intervals.columns:
        agg[M.MASS_KG] = (M.MASS_KG, "sum")
    for col in columns:
        agg[col] = (col, "first")
    if M.ALLOCATION_DOMAIN in intervals.columns:
        agg[M.ALLOCATION_DOMAIN] = (M.ALLOCATION_DOMAIN, "first")
    if M.GEOMET_DOMAIN in intervals.columns:
        agg[M.GEOMET_DOMAIN] = (M.GEOMET_DOMAIN, "first")
    if M.PERIOD in intervals.columns:
        agg[M.PERIOD] = (M.PERIOD, "first")

    return intervals.groupby(M.RUN_ID, sort=True).agg(**agg).reset_index()


def short_run_gaps(
    runs: pd.DataFrame, *, min_length_m: float, min_mass_kg: float
) -> list[GapEntry]:
    """Runs that can never yield a composite. Recorded rather than dropped in silence."""
    if runs.empty:
        return []
    entries: list[GapEntry] = []
    for row in runs.itertuples(index=False):
        if not getattr(row, "usable", True):
            continue
        too_short = getattr(row, M.LENGTH_M) < min_length_m
        mass = getattr(row, M.MASS_KG, None)
        too_light = mass is not None and pd.notna(mass) and mass < min_mass_kg
        if not (too_short or too_light):
            continue
        reason = "run shorter than min_length_m" if too_short else "run cannot reach min_mass_kg"
        entries.append(
            GapEntry(
                scope="run",
                key=getattr(row, M.RUN_ID),
                reason=reason,
                detail={
                    M.HOLE_ID: getattr(row, M.HOLE_ID),
                    M.FROM_M: getattr(row, M.FROM_M),
                    M.TO_M: getattr(row, M.TO_M),
                    M.LENGTH_M: round(float(getattr(row, M.LENGTH_M)), 2),
                    M.MASS_KG: None if mass is None else round(float(mass), 1),
                    M.ALLOCATION_DOMAIN: getattr(row, M.ALLOCATION_DOMAIN, None),
                },
            )
        )
    return entries
