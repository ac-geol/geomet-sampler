"""Generate candidate composites inside runs.

Windows are only ever built from intervals belonging to the same run, so a candidate
structurally cannot span a hard break. Nothing here filters out bad candidates after the
fact: the generator has no reach across a boundary in the first place.

Expansion stops as soon as a window exceeds the maximum length or comfortably clears the
target mass, and only the best window per start interval survives, so long runs do not
produce a quadratic candidate table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..config import Config
from ..models import Availability, GapEntry

CANDIDATE_ID = "candidate_id"
EST_MASS_KG = "est_mass_kg"
CV_PRIMARY = "cv_primary"
N_INTERVALS = "n_intervals"
INTERVAL_IDS = "interval_ids"
QUALITY = "quality"

#: Stop expanding a window once it exceeds the target mass by this factor.
MASS_STOP_FACTOR = 1.5


def generate_candidates(
    intervals: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, list[GapEntry]]:
    """Build the candidate composite table from segmented, mass-estimated intervals."""
    primary = cfg.primary_element
    elements = cfg.elements_with_drillhole()
    roles = list(cfg.attributes)

    gaps: list[GapEntry] = []
    usable, supply_gaps = _usable_intervals(intervals)
    gaps += supply_gaps
    if usable.empty:
        return _empty_candidates(roles, elements), gaps

    rows: list[dict] = []
    for run_id, run in usable.groupby(M.RUN_ID, sort=True):
        run = run.sort_values(M.FROM_M)
        rows.extend(
            _windows_for_run(run, run_id, cfg, primary=primary, elements=elements, roles=roles)
        )

    if not rows:
        return _empty_candidates(roles, elements), gaps

    out = pd.DataFrame(rows)
    out = out.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True)
    out[CANDIDATE_ID] = [f"C{i + 1:06d}" for i in range(len(out))]
    ordered = [CANDIDATE_ID] + [c for c in out.columns if c != CANDIDATE_ID]
    return out[ordered], gaps


# --------------------------------------------------------------------- supply


def _usable_intervals(intervals: pd.DataFrame) -> tuple[pd.DataFrame, list[GapEntry]]:
    """Drop intervals that cannot supply core, recording why."""
    gaps: list[GapEntry] = []
    df = intervals
    if M.AVAILABILITY in df.columns:
        unavailable = df[M.AVAILABILITY] == Availability.UNAVAILABLE.value
        if unavailable.any():
            holes = sorted(set(df.loc[unavailable, M.HOLE_ID]))
            gaps.append(
                GapEntry(
                    scope="supply",
                    key="unavailable_core",
                    reason=f"{len(holes)} holes are marked UNAVAILABLE or are absent from the "
                    "availability file, and supply no candidates",
                    detail={"n_intervals": int(unavailable.sum()), "holes": ", ".join(holes[:20])},
                )
            )
        df = df[~unavailable]
    if "usable" in df.columns:
        df = df[df["usable"].fillna(False).astype(bool)]
    return df, gaps


# ---------------------------------------------------------------------- windows


def _windows_for_run(
    run: pd.DataFrame,
    run_id: str,
    cfg: Config,
    *,
    primary: str,
    elements: list[str],
    roles: list[str],
) -> list[dict]:
    comp = cfg.compositing
    n = len(run)
    length = run[M.LENGTH_M].to_numpy(float)
    mass = run[M.MASS_KG].to_numpy(float)
    primary_col = M.elem_col(primary)
    grade = run[primary_col].to_numpy(float) if primary_col in run.columns else np.full(n, np.nan)

    cum_len = np.concatenate([[0.0], np.cumsum(length)])
    cum_mass = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mass))])
    cum_lg = np.concatenate([[0.0], np.cumsum(length * grade)])
    cum_lg2 = np.concatenate([[0.0], np.cumsum(length * grade * grade)])
    cum_elem = {
        name: np.concatenate([[0.0], np.cumsum(length * run[M.elem_col(name)].to_numpy(float))])
        for name in elements
        if M.elem_col(name) in run.columns
    }

    run_bin = _modal(run[M.GRADE_BIN]) if M.GRADE_BIN in run.columns else None
    from_m = run[M.FROM_M].to_numpy(float)
    to_m = run[M.TO_M].to_numpy(float)
    stop_mass = comp.target_mass_kg * MASS_STOP_FACTOR

    out: list[dict] = []
    for i in range(n):
        best: dict | None = None
        for j in range(i + 1, n + 1):
            total_length = cum_len[j] - cum_len[i]
            if total_length > comp.max_length_m:
                break
            total_mass = cum_mass[j] - cum_mass[i]
            if total_length >= comp.min_length_m and total_mass >= comp.min_mass_kg:
                window = _describe(
                    run,
                    i,
                    j,
                    run_id=run_id,
                    cfg=cfg,
                    primary=primary,
                    roles=roles,
                    total_length=total_length,
                    total_mass=total_mass,
                    cum_lg=cum_lg,
                    cum_lg2=cum_lg2,
                    cum_elem=cum_elem,
                    from_m=from_m,
                    to_m=to_m,
                    run_bin=run_bin,
                )
                if window is not None and (best is None or window[QUALITY] > best[QUALITY]):
                    best = window
            if total_mass > stop_mass:
                break
        if best is not None:
            out.append(best)
    return out


def _describe(
    run: pd.DataFrame,
    i: int,
    j: int,
    *,
    run_id: str,
    cfg: Config,
    primary: str,
    roles: list[str],
    total_length: float,
    total_mass: float,
    cum_lg: np.ndarray,
    cum_lg2: np.ndarray,
    cum_elem: dict[str, np.ndarray],
    from_m: np.ndarray,
    to_m: np.ndarray,
    run_bin: object,
) -> dict | None:
    """Validate one window and turn it into a candidate row, or reject it."""
    comp = cfg.compositing
    mean = (cum_lg[j] - cum_lg[i]) / total_length
    if not np.isfinite(mean):
        return None
    variance = max((cum_lg2[j] - cum_lg2[i]) / total_length - mean * mean, 0.0)
    cv = float(np.sqrt(variance) / mean) if mean > 0 else np.inf
    if cv > comp.max_cv_primary:
        return None

    grade_bin = _bin_for(mean, cfg)
    if grade_bin is None:
        return None
    if comp.require_grade_bin_match and run_bin is not None and grade_bin != run_bin:
        return None

    slice_ = run.iloc[i:j]
    row = {
        M.HOLE_ID: slice_[M.HOLE_ID].iloc[0],
        M.RUN_ID: run_id,
        M.FROM_M: float(from_m[i]),
        M.TO_M: float(to_m[j - 1]),
        M.LENGTH_M: float(total_length),
        N_INTERVALS: int(j - i),
        EST_MASS_KG: float(total_mass),
        M.GEOMET_DOMAIN: _first(slice_, M.GEOMET_DOMAIN),
        M.PERIOD: _first(slice_, M.PERIOD),
        M.GRADE_BIN: grade_bin,
        M.ALLOCATION_DOMAIN: _composite_domain(slice_, cfg, grade_bin),
        M.DOMAIN_MATCH: _match(slice_),
        M.IS_PLANNED: bool(slice_[M.IS_PLANNED].any()) if M.IS_PLANNED in slice_ else False,
        M.AVAILABILITY: _first(slice_, M.AVAILABILITY),
        CV_PRIMARY: cv,
        INTERVAL_IDS: tuple(slice_[M.INTERVAL_ID]),
    }
    for role in roles:
        row[M.attr_col(role)] = _first(slice_, M.attr_col(role))
    for name, cum in cum_elem.items():
        row[M.wtd_elem_col(name)] = float((cum[j] - cum[i]) / total_length)
    for axis, col in ((M.X_MID, M.X_MID), (M.Y_MID, M.Y_MID), (M.Z_MID, M.Z_MID)):
        row[axis] = (
            float(np.average(slice_[col], weights=slice_[M.LENGTH_M]))
            if (col in slice_ and slice_[col].notna().all())
            else np.nan
        )
    row[QUALITY] = quality(row, cfg)
    return row


def quality(candidate: dict | pd.Series, cfg: Config) -> float:
    """Intrinsic desirability of a composite, independent of what else is selected.

    Rewards mass margin over the target, penalises grade heterogeneity, and penalises a
    composite whose logged domain disagrees with the modelled one.
    """
    target = cfg.compositing.target_mass_kg
    mass = float(candidate[EST_MASS_KG])
    mass_factor = min(mass / target, cfg.selection.mass_margin_cap) if target > 0 else 1.0
    cv = float(candidate[CV_PRIMARY])
    cv_factor = 1.0 / (1.0 + cv)
    match = (
        candidate.get(M.DOMAIN_MATCH) if isinstance(candidate, dict) else candidate[M.DOMAIN_MATCH]
    )
    match_factor = cfg.selection.domain_mismatch_factor if match is False else 1.0
    return float(mass_factor * cv_factor * match_factor)


# ---------------------------------------------------------------------- helpers


def _bin_for(value_ppm: float, cfg: Config) -> str | None:
    from ..domains import bin_edges_ppm

    edges = bin_edges_ppm(cfg.domaining.grade_bins, cfg)
    labels = cfg.domaining.grade_bins.labels
    for k in range(len(labels)):
        upper_inclusive = k == len(labels) - 1
        if edges[k] <= value_ppm < edges[k + 1] or (upper_inclusive and value_ppm == edges[k + 1]):
            return labels[k]
    return None


def _composite_domain(slice_: pd.DataFrame, cfg: Config, grade_bin: str) -> str:
    """Label the composite from its own values, not from its first interval.

    The composite's length-weighted grade can fall in a different bin from the interval
    it starts with, and allocating it to that interval's domain would quietly count it
    against a target it does not represent.
    """
    from ..domains import compose_domain

    parts: list[object] = []
    for key in cfg.domaining.allocation_key:
        if key == M.GRADE_BIN:
            parts.append(grade_bin)
        elif key == M.PERIOD:
            parts.append(_first(slice_, M.PERIOD))
        else:
            parts.append(_first(slice_, M.bm_attr_col(key)))
    return compose_domain(parts)


def _modal(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return None
    return values.mode().iloc[0]


def _first(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if len(values) else None


def _match(frame: pd.DataFrame):
    if M.DOMAIN_MATCH not in frame.columns:
        return pd.NA
    values = frame[M.DOMAIN_MATCH].dropna()
    if values.empty:
        return pd.NA
    return bool(values.all())


def _empty_candidates(roles: list[str], elements: list[str]) -> pd.DataFrame:
    columns = [
        CANDIDATE_ID,
        M.HOLE_ID,
        M.RUN_ID,
        M.FROM_M,
        M.TO_M,
        M.LENGTH_M,
        N_INTERVALS,
        EST_MASS_KG,
        M.GEOMET_DOMAIN,
        M.PERIOD,
        M.GRADE_BIN,
        M.ALLOCATION_DOMAIN,
        M.DOMAIN_MATCH,
        M.IS_PLANNED,
        M.AVAILABILITY,
        CV_PRIMARY,
        INTERVAL_IDS,
        M.X_MID,
        M.Y_MID,
        M.Z_MID,
        QUALITY,
    ]
    columns += [M.attr_col(r) for r in roles]
    columns += [M.wtd_elem_col(e) for e in elements]
    return pd.DataFrame(columns=columns)


def pick_list(
    selected: pd.DataFrame, intervals: pd.DataFrame, composite_id_col: str = "composite_id"
) -> pd.DataFrame:
    """Explode selected composites back to their constituent sample intervals.

    A composite without its interval list is useless to the core shed, so this is the
    actual deliverable and is built directly from the recorded interval ids.
    """
    if selected.empty:
        return pd.DataFrame()
    lookup = intervals.set_index(M.INTERVAL_ID)
    rows = []
    for row in selected.itertuples(index=False):
        composite_id = getattr(row, composite_id_col)
        for interval_id in getattr(row, INTERVAL_IDS):
            if interval_id not in lookup.index:  # pragma: no cover - defensive
                continue
            source = lookup.loc[interval_id]
            entry = {
                composite_id_col: composite_id,
                M.HOLE_ID: source[M.HOLE_ID],
                M.INTERVAL_ID: interval_id,
                M.SAMPLE_ID: source.get(M.PARENT_SAMPLE_ID),
                M.FROM_M: source[M.FROM_M],
                M.TO_M: source[M.TO_M],
                M.LENGTH_M: source[M.LENGTH_M],
                M.LOGGED_CODE: source.get(M.LOGGED_CODE),
                M.GEOMET_DOMAIN: source.get(M.GEOMET_DOMAIN),
                M.MASS_KG: source.get(M.MASS_KG),
                M.CORE_DIAMETER_MM: source.get(M.CORE_DIAMETER_MM),
            }
            for column in source.index:
                if column.startswith("elem_"):
                    entry[column] = source[column]
            rows.append(entry)
    return pd.DataFrame(rows).sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True)
