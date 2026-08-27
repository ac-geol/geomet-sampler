"""Assign block model attributes and mine plan period to drillhole intervals.

Nearest centroid via a KD-tree, then an explicit containment test against the block's
own DX/DY/DZ. Nearest-neighbour alone would happily attach a block a hundred metres
away to an interval that is off the model entirely, which is exactly the failure a
coordinate or unit mismatch produces.

For coarse blocks, midpoint assignment is adequate. If interval lengths approach block
height, sub-sampling the interval and taking the modal block would be the next step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .. import models as M
from ..models import Issue, Severity


def assign_blocks(
    intervals: pd.DataFrame,
    block_model: pd.DataFrame,
    *,
    roles: list[str],
    elements: list[str],
    tolerance_m: float = 1e-6,
) -> tuple[pd.DataFrame, list[Issue]]:
    """Attach period, block attributes and block grades to each interval midpoint.

    ``roles`` and ``elements`` come from the user's config; nothing here enumerates a
    known list of rock types, weathering states or elements.
    """
    issues: list[Issue] = []
    out = intervals.copy()

    carried = [M.PERIOD, M.DENSITY] + [M.bm_attr_col(r) for r in roles]
    carried += [M.bm_elem_col(e) for e in elements]
    carried = [c for c in carried if c in block_model.columns]

    bm_density = M.DENSITY in block_model.columns
    for col in carried:
        target = "bm_density" if col == M.DENSITY else col
        out[target] = (
            pd.Series([None] * len(out), dtype=object) if _is_object(block_model[col]) else np.nan
        )
    out[M.OUTSIDE_MODEL] = True

    if block_model.empty or out.empty:
        issues.append(
            Issue(
                Severity.WARN,
                "block_model_empty",
                "no blocks available, every interval is flagged outside_model",
                source="block_model",
            )
        )
        return out, issues

    mid = out[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy(dtype=float)
    finite = np.isfinite(mid).all(axis=1)
    if not finite.any():
        issues.append(
            Issue(
                Severity.WARN,
                "no_desurveyed_midpoints",
                "no interval has a desurveyed midpoint, block assignment skipped",
                source="survey",
            )
        )
        return out, issues

    centroids = block_model[[M.BLOCK_X, M.BLOCK_Y, M.BLOCK_Z]].to_numpy(dtype=float)
    tree = cKDTree(centroids)
    _, nearest = tree.query(mid[finite], k=1)

    half = block_model[[M.BLOCK_DX, M.BLOCK_DY, M.BLOCK_DZ]].to_numpy(dtype=float)[nearest] / 2.0
    offset = np.abs(mid[finite] - centroids[nearest])
    contained = (offset <= half + tolerance_m).all(axis=1)

    rows = out.index[finite][contained]
    picked = block_model.iloc[nearest[contained]]
    for col in carried:
        target = "bm_density" if col == M.DENSITY else col
        out.loc[rows, target] = picked[col].to_numpy()
    out.loc[rows, M.OUTSIDE_MODEL] = False

    if not bm_density:
        out["bm_density"] = np.nan

    off_model = int(out[M.OUTSIDE_MODEL].sum())
    if off_model:
        issues.append(
            Issue(
                Severity.INFO,
                "outside_model",
                f"{off_model} of {len(out)} intervals fall outside every block; "
                "they carry no period and cannot supply candidates",
                source="block_model",
                count=off_model,
            )
        )
    return out, issues


def off_model_fraction(intervals: pd.DataFrame) -> float:
    """Fraction of intervals that found no containing block. Feeds the extent check."""
    if intervals.empty:
        return 0.0
    return float(intervals[M.OUTSIDE_MODEL].mean())


def record_domain_match(
    intervals: pd.DataFrame, roles: list[str], *, domain_match_attribute: str | None = None
) -> pd.DataFrame:
    """Compare logged against modelled values wherever both are available.

    Two things are compared: any attribute role logged on both sides, and, if the user
    has named one, the block model role whose vocabulary should agree with the logged
    ``geomet_domain``. That correspondence is never guessed, because two vocabularies
    that happen to share words are not necessarily the same classification.

    This is a QC output and never a filter. Where they disagree the sample is still
    physically valid, but it may not represent the domain the mine plan assumes. Where
    nothing can be compared, ``domain_match`` is null rather than True.
    """
    out = intervals.copy()
    pairs: list[tuple[str, str]] = [
        (M.dh_attr_col(r), M.bm_attr_col(r))
        for r in roles
        if M.dh_attr_col(r) in out.columns and M.bm_attr_col(r) in out.columns
    ]
    if domain_match_attribute is not None:
        modelled = M.bm_attr_col(domain_match_attribute)
        if M.GEOMET_DOMAIN in out.columns and modelled in out.columns:
            pairs.append((M.GEOMET_DOMAIN, modelled))

    if not pairs:
        out[M.DOMAIN_MATCH] = pd.NA
        return out

    agree = pd.Series(True, index=out.index)
    known = pd.Series(False, index=out.index)
    for logged_col, modelled_col in pairs:
        logged = out[logged_col].astype("string").str.upper()
        modelled = out[modelled_col].astype("string").str.upper()
        pair_known = logged.notna() & modelled.notna()
        known |= pair_known
        agree &= ~pair_known | (logged == modelled)

    out[M.DOMAIN_MATCH] = agree.astype("boolean").where(known, pd.NA)
    return out


def resolve_attributes(intervals: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    """Resolve each attribute role to a single value per interval.

    Logged geology wins where it exists, because compositing boundaries must follow the
    material actually in the tray. The block model fills in the rest. Allocation domains
    are built separately, from block model values only, so that targets and achieved
    counts are always expressed in the same vocabulary.
    """
    out = intervals.copy()
    for role in roles:
        logged_col = M.dh_attr_col(role)
        modelled_col = M.bm_attr_col(role)
        if logged_col in out.columns and modelled_col in out.columns:
            logged = out[logged_col].astype("object")
            out[M.attr_col(role)] = logged.where(logged.notna(), out[modelled_col])
        elif logged_col in out.columns:
            out[M.attr_col(role)] = out[logged_col]
        elif modelled_col in out.columns:
            out[M.attr_col(role)] = out[modelled_col]
        else:
            out[M.attr_col(role)] = pd.NA
    return out


def _is_object(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)
