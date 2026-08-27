"""Greedy selection with a swap improvement pass.

Greedy is interpretable, needs no solver, runs in seconds, and for a coverage problem of
this shape usually lands within a few percent of optimal. Every pick records why it was
made, which matters more here than the last few percent: the output is a recommendation
a geologist has to review and be able to overrule.

A MILP formulation could be dropped in behind :func:`select` without changing anything
upstream or downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..allocate.targets import DEFICIT
from ..composite.candidates import (
    CANDIDATE_ID,
    CV_PRIMARY,
    EST_MASS_KG,
    INTERVAL_IDS,
    QUALITY,
)
from ..config import Config
from ..models import Issue, Severity
from .objective import objective

COMPOSITE_ID = "composite_id"
SCORE = "score"
REASON = "reason"
SELECTION_ORDER = "selection_order"


def select(
    candidates: pd.DataFrame, allocation: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, list[Issue]]:
    """Choose composites to fill the per-domain deficits.

    Returns the selected composites, the full candidate table with its final scores for
    audit, and any issues raised along the way.
    """
    issues: list[Issue] = []
    scored = candidates.copy()
    scored[SCORE] = np.nan
    scored[SELECTION_ORDER] = pd.NA
    scored[REASON] = ""

    if candidates.empty:
        issues.append(
            Issue(
                Severity.ERROR,
                "no_candidates",
                "no candidate composite met the length, mass, grade bin and availability "
                "constraints, so nothing can be selected",
                source="composite",
            )
        )
        return _empty_selection(scored), scored, issues

    deficits = {
        str(row[M.ALLOCATION_DOMAIN]): float(row[DEFICIT])
        for _, row in allocation.iterrows()
        if float(row[DEFICIT]) > 0
    }
    if not deficits:
        issues.append(
            Issue(
                Severity.WARN,
                "no_deficit",
                "every domain deficit is zero, so no sample is selected",
                source="allocation",
            )
        )
        return _empty_selection(scored), scored, issues

    state = _State(cfg)
    chosen: list[int] = []
    total_wanted = int(sum(deficits.values()))

    for step in range(total_wanted):
        scores, reasons = _score_all(scored, deficits, state, cfg)
        best = _best_index(scores, scored)
        if best is None:
            issues.append(
                Issue(
                    Severity.WARN,
                    "selection_exhausted",
                    f"stopped after {step} of {total_wanted} samples: no remaining candidate "
                    "scores above zero. Remaining deficits are in the gap register.",
                    source="selection",
                    count=total_wanted - step,
                )
            )
            break
        chosen.append(best)
        row = scored.loc[best]
        domain = str(row[M.ALLOCATION_DOMAIN])
        scored.at[best, SCORE] = scores[best]
        scored.at[best, SELECTION_ORDER] = len(chosen)
        scored.at[best, REASON] = reasons[best]
        deficits[domain] = max(deficits.get(domain, 0.0) - 1.0, 0.0)
        state.take(row)

    selected = scored.loc[chosen].copy()
    selected, swaps = _swap_pass(selected, scored, allocation, deficits, state, cfg)
    if swaps:
        issues.append(
            Issue(
                Severity.INFO,
                "swap_improvements",
                f"the swap pass replaced {swaps} composites and improved the objective",
                source="selection",
                count=swaps,
            )
        )

    selected = selected.sort_values([M.ALLOCATION_DOMAIN, M.HOLE_ID, M.FROM_M]).reset_index(
        drop=True
    )
    selected.insert(0, COMPOSITE_ID, [f"GM-{i + 1:03d}" for i in range(len(selected))])
    return selected, scored, issues


# ------------------------------------------------------------------- scoring


class _State:
    """What has been selected so far, in the forms the penalties need."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.per_hole: dict[str, int] = {}
        self.spans: dict[str, list[tuple[float, float]]] = {}
        self.points: list[np.ndarray] = []

    def take(self, row: pd.Series) -> None:
        hole = str(row[M.HOLE_ID])
        self.per_hole[hole] = self.per_hole.get(hole, 0) + 1
        self.spans.setdefault(hole, []).append((float(row[M.FROM_M]), float(row[M.TO_M])))
        self.points.append(row[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy(dtype=float))

    def release(self, row: pd.Series) -> None:
        hole = str(row[M.HOLE_ID])
        self.per_hole[hole] = max(self.per_hole.get(hole, 1) - 1, 0)
        span = (float(row[M.FROM_M]), float(row[M.TO_M]))
        if span in self.spans.get(hole, []):
            self.spans[hole].remove(span)
        point = row[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy(dtype=float)
        for i, existing in enumerate(self.points):
            if np.allclose(existing, point, equal_nan=True):
                self.points.pop(i)
                break

    def hole_penalty(self, row: pd.Series) -> float:
        return (
            0.0
            if self.per_hole.get(str(row[M.HOLE_ID]), 0) >= self.cfg.selection.max_samples_per_hole
            else 1.0
        )

    def overlaps(self, row: pd.Series) -> bool:
        lo, hi = float(row[M.FROM_M]), float(row[M.TO_M])
        return any(
            lo < end and hi > start for start, end in self.spans.get(str(row[M.HOLE_ID]), [])
        )

    def separation_penalty(self, row: pd.Series) -> float:
        limit = self.cfg.selection.min_separation_m
        if limit <= 0 or not self.points:
            return 1.0
        point = row[[M.X_MID, M.Y_MID, M.Z_MID]].to_numpy(dtype=float)
        if not np.isfinite(point).all():
            return 1.0
        distances = np.linalg.norm(np.vstack(self.points) - point, axis=1)
        return 0.0 if float(distances.min()) < limit else 1.0


def _score_all(
    candidates: pd.DataFrame, deficits: dict[str, float], state: _State, cfg: Config
) -> tuple[np.ndarray, dict[int, str]]:
    scores = np.zeros(len(candidates), dtype=float)
    reasons: dict[int, str] = {}
    positions = {idx: i for i, idx in enumerate(candidates.index)}

    for idx, row in candidates.iterrows():
        if pd.notna(row[SELECTION_ORDER]):
            continue
        domain = str(row[M.ALLOCATION_DOMAIN])
        remaining = deficits.get(domain, 0.0)
        if remaining <= 0:
            continue
        if state.overlaps(row):
            continue
        hole_factor = state.hole_penalty(row)
        if hole_factor == 0.0:
            continue
        separation = state.separation_penalty(row)
        if separation == 0.0:
            continue
        period_weight = cfg.allocation.period_weight(row.get(M.PERIOD))
        planned = cfg.selection.planned_penalty if bool(row.get(M.IS_PLANNED, False)) else 1.0
        score = remaining * period_weight * float(row[QUALITY]) * separation * hole_factor * planned
        scores[positions[idx]] = score
        if score > 0:
            reasons[idx] = _reason(row, domain, remaining, period_weight)
    return scores, reasons


def _reason(row: pd.Series, domain: str, remaining: float, period_weight: float) -> str:
    parts = [
        f"fills {domain} deficit ({int(remaining)} remaining)",
        f"period {row.get(M.PERIOD)} (weight {period_weight:g})",
        f"{float(row[EST_MASS_KG]):.1f} kg",
        f"CV {float(row[CV_PRIMARY]):.2f}",
    ]
    if row.get(M.DOMAIN_MATCH) is False:
        parts.append("logged domain disagrees with the block model")
    if bool(row.get(M.IS_PLANNED, False)):
        parts.append("planned hole, core not yet available")
    return ", ".join(parts)


def _best_index(scores: np.ndarray, candidates: pd.DataFrame) -> int | None:
    """Highest score, ties broken by lowest CV then hole ID, for determinism."""
    if not scores.any():
        return None
    best = scores.max()
    tied = candidates.index[np.isclose(scores, best) & (scores > 0)]
    if len(tied) == 1:
        return int(tied[0])
    frame = candidates.loc[tied]
    order = frame.sort_values([CV_PRIMARY, M.HOLE_ID, M.FROM_M], kind="stable")
    return int(order.index[0])


# ----------------------------------------------------------------- swap pass


def _swap_pass(
    selected: pd.DataFrame,
    scored: pd.DataFrame,
    allocation: pd.DataFrame,
    deficits: dict[str, float],
    state: _State,
    cfg: Config,
) -> tuple[pd.DataFrame, int]:
    """Try replacing selected composites with unselected ones, keeping any improvement."""
    if selected.empty or cfg.selection.swap_iterations <= 0:
        return selected, 0

    rng = np.random.default_rng(cfg.selection.random_seed)
    pool = scored.index[scored[SELECTION_ORDER].isna()]
    if len(pool) == 0:
        return selected, 0

    current = list(selected.index)
    best_value = objective(scored.loc[current], allocation, cfg)
    improvements = 0

    for _ in range(cfg.selection.swap_iterations):
        out_pos = int(rng.integers(len(current)))
        out_idx = current[out_pos]
        in_idx = int(pool[int(rng.integers(len(pool)))])
        if in_idx in current:
            continue

        state.release(scored.loc[out_idx])
        candidate_row = scored.loc[in_idx]
        feasible = (
            not state.overlaps(candidate_row)
            and state.hole_penalty(candidate_row) > 0
            and state.separation_penalty(candidate_row) > 0
        )
        if not feasible:
            state.take(scored.loc[out_idx])
            continue

        trial = current.copy()
        trial[out_pos] = in_idx
        value = objective(scored.loc[trial], allocation, cfg)
        if value > best_value + 1e-12:
            current = trial
            best_value = value
            improvements += 1
            state.take(candidate_row)
            scored.at[in_idx, SELECTION_ORDER] = scored.at[out_idx, SELECTION_ORDER]
            scored.at[in_idx, REASON] = (
                _reason(
                    candidate_row,
                    str(candidate_row[M.ALLOCATION_DOMAIN]),
                    max(deficits.get(str(candidate_row[M.ALLOCATION_DOMAIN]), 1.0), 1.0),
                    cfg.allocation.period_weight(candidate_row.get(M.PERIOD)),
                )
                + ", kept by the swap pass as it improved the overall programme"
            )
            scored.at[out_idx, SELECTION_ORDER] = pd.NA
            scored.at[out_idx, REASON] = ""
        else:
            state.take(scored.loc[out_idx])

    return scored.loc[current].copy(), improvements


def _empty_selection(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.iloc[0:0].copy()
    out.insert(0, COMPOSITE_ID, pd.Series(dtype=object))
    return out


def achieved_counts(selected: pd.DataFrame) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    return selected[M.ALLOCATION_DOMAIN].value_counts().astype(float)


def check_no_interval_reused(selected: pd.DataFrame) -> list[Issue]:
    """No sample interval may appear in two composites: it can only be cut once."""
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for row in selected.itertuples(index=False):
        for interval_id in getattr(row, INTERVAL_IDS):
            if interval_id in seen:
                clashes.append(
                    f"{interval_id} in {seen[interval_id]} and {getattr(row, COMPOSITE_ID)}"
                )
            seen[interval_id] = getattr(row, COMPOSITE_ID)
    if not clashes:
        return []
    return [
        Issue(
            Severity.ERROR,
            "interval_reused",
            f"{len(clashes)} sample intervals appear in more than one composite: {clashes[:5]}",
            source="selection",
            count=len(clashes),
        )
    ]


__all__ = [
    "CANDIDATE_ID",
    "COMPOSITE_ID",
    "REASON",
    "SCORE",
    "SELECTION_ORDER",
    "achieved_counts",
    "check_no_interval_reused",
    "select",
]
