"""The global objective the swap pass tries to improve.

Kept separate from the greedy loop so the two can be reasoned about independently: the
greedy pass is a local rule about the next best sample, this is a statement about what a
good programme looks like overall.

    objective = -sum(|achieved_d - target_d|) * W1   allocation fit
              + mean_min_separation           * W2   spatial spread
              - sum(cv_primary)               * W3   sample homogeneity
              + domain_match_rate             * W4   logged/modelled agreement
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..allocate.targets import DEFICIT
from ..composite.candidates import CV_PRIMARY
from ..config import Config


def mean_min_separation(selected: pd.DataFrame) -> float:
    """Mean 3D distance from each selected composite to its nearest neighbour."""
    if len(selected) < 2:
        return 0.0
    points = selected[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy(dtype=float)
    if not np.isfinite(points).all():
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).mean())


def allocation_fit(selected: pd.DataFrame, allocation: pd.DataFrame) -> float:
    """Total absolute shortfall and overshoot against the per-domain deficits."""
    targets = allocation.set_index(M.ALLOCATION_DOMAIN)[DEFICIT].astype(float)
    achieved = (
        selected[M.ALLOCATION_DOMAIN].value_counts().astype(float)
        if not selected.empty
        else pd.Series(dtype=float)
    )
    domains = targets.index.union(achieved.index)
    return float(
        (achieved.reindex(domains).fillna(0.0) - targets.reindex(domains).fillna(0.0)).abs().sum()
    )


def domain_match_rate(selected: pd.DataFrame) -> float:
    """Fraction of selected composites whose logged and modelled domains agree.

    Composites where the comparison is impossible are excluded rather than counted as
    agreement, so an unlogged dataset does not score as a perfect match.
    """
    if selected.empty or M.DOMAIN_MATCH not in selected.columns:
        return 0.0
    known = selected[M.DOMAIN_MATCH].dropna()
    return float(known.astype(bool).mean()) if len(known) else 0.0


def objective(selected: pd.DataFrame, allocation: pd.DataFrame, cfg: Config) -> float:
    """Score a whole programme. Higher is better."""
    w = cfg.selection.objective_weights
    cv_total = float(selected[CV_PRIMARY].sum()) if not selected.empty else 0.0
    return (
        -allocation_fit(selected, allocation) * w.get("allocation", 10.0)
        + mean_min_separation(selected) * w.get("separation", 0.01)
        - cv_total * w.get("cv", 1.0)
        + domain_match_rate(selected) * w.get("match", 5.0)
    )


def components(selected: pd.DataFrame, allocation: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """The objective broken out, for the run summary."""
    return {
        "allocation_absolute_error": allocation_fit(selected, allocation),
        "mean_min_separation_m": mean_min_separation(selected),
        "total_cv_primary": float(selected[CV_PRIMARY].sum()) if not selected.empty else 0.0,
        "domain_match_rate": domain_match_rate(selected),
        "objective": objective(selected, allocation, cfg),
    }
