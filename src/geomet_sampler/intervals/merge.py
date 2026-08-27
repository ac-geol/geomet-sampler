"""Build the single interval framework every later stage works on.

Assay intervals are the base framework because they carry the grades and the physical
sample identity. Lithology is layered on by splitting assay intervals wherever a logged
boundary falls inside one, so a framework interval never straddles a logged contact.
Grade is constant within an assay sample, so the split halves inherit it unchanged and
no re-averaging is involved.

Every framework row keeps ``parent_sample_id``: without it a composite cannot be turned
into a core shed instruction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..models import Issue, Severity

#: Depths closer than this are the same boundary. Guards against float noise creating
#: zero-length slivers when an assay boundary coincides with a logged contact.
DEPTH_TOL = 1e-6


def build_framework(
    assay: pd.DataFrame,
    litho: pd.DataFrame,
    domain_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Issue]]:
    """Split assay intervals at lithology boundaries and attach the geomet domain."""
    issues: list[Issue] = []
    litho_by_hole = {
        hid: g.sort_values(M.FROM_M) for hid, g in litho.groupby(M.HOLE_ID, sort=False)
    }

    pieces: list[pd.DataFrame] = []
    holes_without_litho: list[str] = []

    for hole_id, group in assay.groupby(M.HOLE_ID, sort=False):
        group = group.sort_values(M.FROM_M)
        litho_h = litho_by_hole.get(hole_id)
        if litho_h is None:
            holes_without_litho.append(hole_id)
            pieces.append(_no_split(group))
            continue
        pieces.append(_split_group(group, litho_h))

    if not pieces:
        return _empty_framework(assay), issues

    framework = pd.concat(pieces, ignore_index=True)
    framework[M.LENGTH_M] = framework[M.TO_M] - framework[M.FROM_M]
    framework = framework[framework[M.LENGTH_M] > DEPTH_TOL].reset_index(drop=True)

    framework, domain_issues = _attach_domain(framework, domain_lookup)
    issues += domain_issues

    if holes_without_litho:
        issues.append(
            Issue(
                Severity.WARN,
                "no_lithology",
                f"{len(holes_without_litho)} holes have assays but no logged lithology; "
                "their intervals carry no geomet domain and cannot supply candidates",
                source="litho",
                count=len(holes_without_litho),
                detail={"holes": sorted(holes_without_litho)[:20]},
            )
        )

    unlogged = int(framework[M.LOGGED_CODE].isna().sum())
    if unlogged:
        issues.append(
            Issue(
                Severity.WARN,
                "interval_not_logged",
                f"{unlogged} assay intervals fall outside every logged lithology interval; "
                "they carry no domain and are excluded from candidates",
                source="litho",
                count=unlogged,
            )
        )

    framework[M.INTERVAL_ID] = _interval_ids(framework)
    return framework.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True), issues


# ------------------------------------------------------------------- splitting


def _split_group(assay_h: pd.DataFrame, litho_h: pd.DataFrame) -> pd.DataFrame:
    """Cut each assay interval at every logged boundary that falls strictly inside it."""
    boundaries = np.unique(
        np.concatenate([litho_h[M.FROM_M].to_numpy(float), litho_h[M.TO_M].to_numpy(float)])
    )
    boundaries = boundaries[np.isfinite(boundaries)]

    from_m = assay_h[M.FROM_M].to_numpy(float)
    to_m = assay_h[M.TO_M].to_numpy(float)

    row_index: list[int] = []
    new_from: list[float] = []
    new_to: list[float] = []
    for i in range(len(assay_h)):
        lo, hi = from_m[i], to_m[i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo <= DEPTH_TOL:
            row_index.append(i)
            new_from.append(lo)
            new_to.append(hi)
            continue
        inner = boundaries[(boundaries > lo + DEPTH_TOL) & (boundaries < hi - DEPTH_TOL)]
        edges = np.concatenate([[lo], inner, [hi]])
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            row_index.append(i)
            new_from.append(a)
            new_to.append(b)

    out = assay_h.iloc[row_index].reset_index(drop=True)
    out = _rename_sample_id(out)
    out[M.FROM_M] = new_from
    out[M.TO_M] = new_to
    return _attach_litho(out, litho_h)


def _no_split(assay_h: pd.DataFrame) -> pd.DataFrame:
    out = _rename_sample_id(assay_h.reset_index(drop=True))
    out[M.LOGGED_CODE] = pd.Series([None] * len(out), dtype=object)
    return out


def _rename_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={M.SAMPLE_ID: M.PARENT_SAMPLE_ID})
    if M.PARENT_SAMPLE_ID not in out.columns:
        out[M.PARENT_SAMPLE_ID] = None
    return out


def _attach_litho(intervals: pd.DataFrame, litho_h: pd.DataFrame) -> pd.DataFrame:
    """Assign the logged interval containing each framework midpoint."""
    starts = litho_h[M.FROM_M].to_numpy(float)
    ends = litho_h[M.TO_M].to_numpy(float)
    codes = litho_h[M.LOGGED_CODE].to_numpy(object)

    mid = (intervals[M.FROM_M].to_numpy(float) + intervals[M.TO_M].to_numpy(float)) / 2.0
    slot = np.searchsorted(starts, mid, side="right") - 1
    valid = (slot >= 0) & (slot < len(starts))
    inside = np.zeros(len(mid), dtype=bool)
    inside[valid] = mid[valid] < ends[slot[valid]]

    assigned = np.full(len(mid), None, dtype=object)
    assigned[inside] = codes[slot[inside]]
    intervals[M.LOGGED_CODE] = assigned

    for col in litho_h.columns:
        if col.startswith("dh_attr_") and col not in intervals.columns:
            values = litho_h[col].to_numpy(object)
            carried = np.full(len(mid), None, dtype=object)
            carried[inside] = values[slot[inside]]
            intervals[col] = carried
    return intervals


def _attach_domain(
    framework: pd.DataFrame, domain_lookup: pd.DataFrame
) -> tuple[pd.DataFrame, list[Issue]]:
    lookup = dict(
        zip(
            domain_lookup[M.LOGGED_CODE],
            domain_lookup[M.GEOMET_DOMAIN],
            strict=True,
        )
    )
    codes = framework[M.LOGGED_CODE]
    framework[M.GEOMET_DOMAIN] = codes.map(lookup)

    logged = codes.notna()
    unmapped = sorted(set(codes[logged & framework[M.GEOMET_DOMAIN].isna()].astype(str)))
    issues: list[Issue] = []
    if unmapped:
        n = int((logged & framework[M.GEOMET_DOMAIN].isna()).sum())
        issues.append(
            Issue(
                Severity.ERROR,
                "logged_code_unmapped",
                f"logged codes absent from the domain lookup: {unmapped}. "
                "Dropping them silently would distort the allocation, so add every code "
                "to the lookup, mapping any you want excluded to an explicit domain.",
                source="domain_lookup",
                count=n,
                detail={"codes": unmapped},
            )
        )
    return framework, issues


def _interval_ids(framework: pd.DataFrame) -> pd.Series:
    """Unique id per framework row; suffixed only where a sample was actually split."""
    parent = framework[M.PARENT_SAMPLE_ID].astype(str)
    order = framework.groupby(parent.values, sort=False).cumcount()
    counts = parent.map(parent.value_counts())
    return np.where(counts.to_numpy() > 1, parent + "_" + (order + 1).astype(str), parent)


def _empty_framework(assay: pd.DataFrame) -> pd.DataFrame:
    out = _rename_sample_id(assay.iloc[0:0].copy())
    for col in (M.LOGGED_CODE, M.GEOMET_DOMAIN, M.INTERVAL_ID):
        out[col] = pd.Series(dtype=object)
    out[M.LENGTH_M] = pd.Series(dtype=float)
    return out


# ------------------------------------------------------------- planned holes


def discretise_planned(traces: dict, hole_ids: list[str], step_m: float) -> pd.DataFrame:
    """Cut planned hole traces into fixed-length intervals.

    Planned holes have no assays, so grades and density come from the block model later
    and every interval here is a placeholder carrying only geometry.
    """
    rows = []
    for hole_id in hole_ids:
        trace = traces.get(hole_id)
        if trace is None:
            continue
        edges = np.arange(0.0, trace.total_depth + step_m, step_m)
        edges = edges[edges <= trace.total_depth + DEPTH_TOL]
        if len(edges) < 2:
            continue
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            rows.append(
                {
                    M.HOLE_ID: hole_id,
                    M.PARENT_SAMPLE_ID: f"{hole_id}_PLAN_{a:.2f}",
                    M.FROM_M: float(a),
                    M.TO_M: float(b),
                    M.LENGTH_M: float(b - a),
                    M.LOGGED_CODE: None,
                    M.GEOMET_DOMAIN: None,
                    M.LOW_RECOVERY: False,
                    M.IS_PLANNED: True,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out[M.INTERVAL_ID] = out[M.PARENT_SAMPLE_ID]
    return out
