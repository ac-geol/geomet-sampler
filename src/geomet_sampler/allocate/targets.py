"""Per-domain sample targets from mine plan tonnage.

Targets come from the block model, never from the drilling: how much of each domain is
scheduled to be mined, weighted toward earlier periods, is the question the programme has
to answer. What drilling happens to be available is a supply constraint handled later, in
selection, and it is deliberately kept out of the target calculation so that a domain
with no available core still shows as an unmet deficit rather than quietly vanishing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..config import Config
from ..domains import is_scheduled
from ..models import Issue, Severity

TONNES = "tonnes"
TONNES_PCT = "tonnes_pct"
WEIGHTED_TONNES = "weighted_tonnes"
WEIGHTED_PCT = "weighted_pct"
GRADE_SD = "grade_sd"
RISK = "risk_multiplier"
TARGET = "target"
EXISTING = "existing"
DEFICIT = "deficit"
DEFICIT_RAW = "deficit_raw"


def domain_tonnage(blocks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Scheduled tonnage and weighted tonnage per allocation domain.

    Blocks with a blank or excluded period are not scheduled and take no part in the
    allocation, which is what makes the targets reflect the mine plan rather than the
    resource.
    """
    scheduled = blocks[is_scheduled(blocks[M.PERIOD], cfg.block_model_options.excluded_periods)]
    if scheduled.empty:
        return pd.DataFrame(
            columns=[M.ALLOCATION_DOMAIN, TONNES, WEIGHTED_TONNES, TONNES_PCT, WEIGHTED_PCT]
        )

    weights = scheduled[M.PERIOD].map(cfg.allocation.period_weight)
    frame = pd.DataFrame(
        {
            M.ALLOCATION_DOMAIN: scheduled[M.ALLOCATION_DOMAIN],
            TONNES: scheduled[M.BLOCK_TONNES],
            WEIGHTED_TONNES: scheduled[M.BLOCK_TONNES] * weights,
        }
    )
    out = frame.groupby(M.ALLOCATION_DOMAIN, sort=True, as_index=False).sum()
    out[TONNES_PCT] = 100.0 * out[TONNES] / out[TONNES].sum()
    out[WEIGHTED_PCT] = 100.0 * out[WEIGHTED_TONNES] / out[WEIGHTED_TONNES].sum()
    return out


def tonnage_by_period(blocks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Scheduled tonnage split by domain and period, for reporting."""
    scheduled = blocks[is_scheduled(blocks[M.PERIOD], cfg.block_model_options.excluded_periods)]
    if scheduled.empty:
        return pd.DataFrame(columns=[M.ALLOCATION_DOMAIN, M.PERIOD, TONNES])
    return (
        scheduled.groupby([M.ALLOCATION_DOMAIN, M.PERIOD], sort=True)[M.BLOCK_TONNES]
        .sum()
        .reset_index(name=TONNES)
    )


def _grade_dispersion(blocks: pd.DataFrame, cfg: Config) -> tuple[pd.Series, list[Issue]]:
    """Standard deviation of the primary grade within each domain, from the block model."""
    column = M.bm_elem_col(cfg.primary_element)
    scheduled = blocks[is_scheduled(blocks[M.PERIOD], cfg.block_model_options.excluded_periods)]
    if column not in blocks.columns:
        return pd.Series(dtype=float), [
            Issue(
                Severity.WARN,
                "neyman_no_block_grade",
                f"allocation.method is neyman but the primary element "
                f"{cfg.primary_element!r} has no block_model field, so within-domain "
                "variability is unknown. Falling back to proportional allocation.",
                source="block_model",
            )
        ]
    sd = scheduled.groupby(M.ALLOCATION_DOMAIN, sort=True)[column].std(ddof=0)
    return sd, []


def compute_targets(
    blocks: pd.DataFrame,
    cfg: Config,
    existing: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[Issue]]:
    """Full allocation table: tonnage, target, existing coverage and deficit.

    Deficits are rescaled so that exactly ``allocation.total_samples`` samples are
    allocated, then rounded by largest remainder so the rounded total is exact rather
    than approximately right.
    """
    issues: list[Issue] = []
    table = domain_tonnage(blocks, cfg)
    if table.empty:
        return table, [
            Issue(
                Severity.ERROR,
                "no_scheduled_tonnage",
                "no scheduled block tonnage, so no allocation target can be computed",
                source="block_model",
            )
        ]

    method = cfg.allocation.method
    table[RISK] = (
        table[M.ALLOCATION_DOMAIN].map(cfg.allocation.risk_multipliers).fillna(1.0).astype(float)
    )

    if method == "neyman":
        sd, sd_issues = _grade_dispersion(blocks, cfg)
        issues += sd_issues
        if sd.empty:
            table[GRADE_SD] = 1.0
        else:
            # An unknown dispersion is filled with the average: we have no reason to
            # treat that domain as either uniform or variable. A dispersion that is
            # genuinely zero is left at zero, because a uniform domain really does need
            # fewer samples per tonne. Coverage of such a domain is the job of
            # min_samples_per_domain, not of a fabricated variance.
            values = table[M.ALLOCATION_DOMAIN].map(sd)
            table[GRADE_SD] = values.fillna(values.mean() if values.notna().any() else 1.0)
            if (table[GRADE_SD] > 0).sum() == 0:
                issues.append(
                    Issue(
                        Severity.WARN,
                        "neyman_no_dispersion",
                        f"the block model shows no within-domain variability in "
                        f"{cfg.primary_element}, so Neyman weighting has nothing to work "
                        "with and the result reduces to the per-domain floors.",
                        source="block_model",
                    )
                )
        basis = table[WEIGHTED_TONNES] * table[GRADE_SD] * table[RISK]
    elif method == "proportional":
        table[GRADE_SD] = np.nan
        basis = table[WEIGHTED_TONNES] * table[RISK]
    else:  # manual
        table[GRADE_SD] = np.nan
        basis = table[M.ALLOCATION_DOMAIN].map(cfg.allocation.manual_targets).astype(float)
        if basis.isna().all():
            issues.append(
                Issue(
                    Severity.ERROR,
                    "manual_targets_missing",
                    "allocation.method is manual but allocation.manual_targets names no domain "
                    f"present in the model. Domains: {sorted(table[M.ALLOCATION_DOMAIN])[:10]}",
                    source="allocation",
                )
            )
        basis = basis.fillna(0.0)

    total = float(cfg.allocation.total_samples)
    table[TARGET] = total * basis / basis.sum() if basis.sum() > 0 else 0.0

    floor_applies = table[TONNES_PCT] >= cfg.allocation.min_domain_tonnage_pct
    table[TARGET] = np.where(
        floor_applies,
        np.maximum(table[TARGET], cfg.allocation.min_samples_per_domain),
        table[TARGET],
    )

    table[EXISTING] = (
        table[M.ALLOCATION_DOMAIN].map(existing).fillna(0.0).astype(float)
        if existing is not None
        else 0.0
    )
    table[DEFICIT_RAW] = np.maximum(table[TARGET] - table[EXISTING], 0.0)

    if table[DEFICIT_RAW].sum() > 0:
        scaled = table[DEFICIT_RAW] * total / table[DEFICIT_RAW].sum()
    else:
        scaled = pd.Series(0.0, index=table.index)
        issues.append(
            Issue(
                Severity.WARN,
                "no_deficit",
                "existing testwork already covers every domain target, so nothing is allocated",
                source="allocation",
            )
        )
    table[DEFICIT] = largest_remainder(scaled, int(cfg.allocation.total_samples))

    table = table.sort_values(WEIGHTED_TONNES, ascending=False).reset_index(drop=True)
    return table, issues


def largest_remainder(values: pd.Series, total: int) -> pd.Series:
    """Round to integers summing exactly to ``total``.

    Ties on the fractional part are broken by position, which is stable because the
    caller sorts domains by name, so the same input always gives the same answer.
    """
    if values.empty or total <= 0:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    floors = np.floor(values.to_numpy(dtype=float)).astype(int)
    shortfall = total - int(floors.sum())
    if shortfall <= 0:
        return pd.Series(floors, index=values.index)
    remainders = values.to_numpy(dtype=float) - floors
    order = np.lexsort((np.arange(len(values)), -remainders))
    floors[order[:shortfall]] += 1
    return pd.Series(floors, index=values.index)


def existing_coverage(
    existing_testwork: pd.DataFrame | None, intervals: pd.DataFrame
) -> tuple[pd.Series, list[Issue]]:
    """Count prior testwork per allocation domain.

    Where the record does not name a domain, it is derived by matching the recorded
    depths back onto the interval framework, which is the same pipeline the candidates
    went through, so prior and proposed samples are counted in one vocabulary.
    """
    if existing_testwork is None or existing_testwork.empty:
        return pd.Series(dtype=float), []
    issues: list[Issue] = []

    known_domains = (
        set(intervals[M.ALLOCATION_DOMAIN].dropna()) if M.ALLOCATION_DOMAIN in intervals else set()
    )
    resolved: list[str | None] = []
    for row in existing_testwork.itertuples(index=False):
        stated = getattr(row, M.GEOMET_DOMAIN, None)
        if stated is not None and pd.notna(stated) and str(stated) in known_domains:
            resolved.append(str(stated))
            continue
        resolved.append(_derive_domain(row, intervals))

    unresolved = sum(1 for r in resolved if r is None)
    if unresolved:
        issues.append(
            Issue(
                Severity.WARN,
                "existing_testwork_undomained",
                f"{unresolved} existing testwork records could not be matched to an interval "
                "and are not counted against any domain target",
                source="existing_testwork",
                count=unresolved,
            )
        )
    counts = pd.Series([r for r in resolved if r is not None]).value_counts().astype(float)
    return counts, issues


def _derive_domain(row, intervals: pd.DataFrame) -> str | None:
    """Modal allocation domain of the framework intervals the record overlaps."""
    if M.ALLOCATION_DOMAIN not in intervals.columns:
        return None
    hole = getattr(row, M.HOLE_ID, None)
    start, end = getattr(row, M.FROM_M, np.nan), getattr(row, M.TO_M, np.nan)
    if hole is None or not (pd.notna(start) and pd.notna(end)):
        return None
    overlapping = intervals[
        (intervals[M.HOLE_ID] == hole) & (intervals[M.TO_M] > start) & (intervals[M.FROM_M] < end)
    ]
    domains = overlapping[M.ALLOCATION_DOMAIN].dropna()
    return str(domains.mode().iloc[0]) if len(domains) else None


def unmet_gaps(allocation: pd.DataFrame, achieved: pd.Series) -> list:
    """Domains whose deficit was not filled, with the count still outstanding."""
    from ..models import GapEntry

    entries = []
    for row in allocation.itertuples(index=False):
        domain = getattr(row, M.ALLOCATION_DOMAIN)
        got = float(achieved.get(domain, 0.0))
        want = float(getattr(row, DEFICIT))
        if got >= want:
            continue
        entries.append(
            GapEntry(
                scope="domain",
                key=domain,
                reason="deficit not filled: no remaining candidate met the mass, grade bin, "
                "separation or availability constraints",
                detail={
                    "target": round(float(getattr(row, TARGET)), 2),
                    "existing": float(getattr(row, EXISTING)),
                    "deficit": want,
                    "achieved": got,
                    "shortfall": want - got,
                    "tonnes_pct": round(float(getattr(row, TONNES_PCT)), 2),
                },
            )
        )
    return entries
