"""Data quality checks producing structured issues.

Nothing here modifies data and nothing is dropped. Every check produces Issue records
that end up in the validation report, so anything the pipeline later ignores has already
been named and counted.

All checks work on canonical field names. Where a message has to name a source column
for the user's benefit, the reader's mapping table translates it back.
"""

from __future__ import annotations

import pandas as pd

from .. import models as M
from ..config import Config
from ..domains import is_scheduled
from ..io.readers import Dataset, source_column
from ..models import Issue, Severity
from ..units import DENSITY_RANGE_T_M3

#: Below this, an interval gap is normal core handling; above it, worth reporting.
GAP_INFO_TOLERANCE_M = 0.10


def validate_inputs(dataset: Dataset, cfg: Config) -> list[Issue]:
    """Every check that can run before desurvey."""
    issues: list[Issue] = []
    issues += _check_unique_collars(dataset)
    issues += _check_orphans(dataset)
    issues += _check_intervals(dataset.assay, "assay", dataset, cfg)
    issues += _check_intervals(dataset.litho, "litho", dataset, cfg)
    issues += _check_beyond_total_depth(dataset)
    issues += _check_survey(dataset, cfg)
    issues += _check_density(dataset)
    issues += _check_block_model(dataset, cfg)
    issues += _check_domain_lookup(dataset)
    return issues


# ----------------------------------------------------------------- collars/ids


def _check_unique_collars(dataset: Dataset) -> list[Issue]:
    duplicated = dataset.collar[dataset.collar.duplicated(M.HOLE_ID, keep=False)]
    if duplicated.empty:
        return []
    holes = sorted(set(duplicated[M.HOLE_ID]))
    return [
        Issue(
            Severity.ERROR,
            "collar_duplicate",
            f"{len(holes)} hole IDs appear more than once in the collar file: {holes[:10]}",
            source="collar",
            count=len(duplicated),
        )
    ]


def _check_orphans(dataset: Dataset) -> list[Issue]:
    known = set(dataset.collar[M.HOLE_ID])
    issues = []
    for name, frame in (
        ("survey", dataset.survey),
        ("assay", dataset.assay),
        ("litho", dataset.litho),
    ):
        if frame is None or frame.empty:
            continue
        orphans = sorted(set(frame[M.HOLE_ID]) - known)
        if orphans:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "orphan_hole",
                    f"{len(orphans)} hole IDs in {name} have no collar record: {orphans[:10]}",
                    source=name,
                    count=int(frame[M.HOLE_ID].isin(orphans).sum()),
                    detail={"holes": orphans[:50]},
                )
            )
    return issues


# ------------------------------------------------------------------- intervals


def _check_intervals(frame: pd.DataFrame, name: str, dataset: Dataset, cfg: Config) -> list[Issue]:
    if frame is None or frame.empty:
        return []
    issues: list[Issue] = []
    from_col = source_column(dataset, name, M.FROM_M)
    to_col = source_column(dataset, name, M.TO_M)

    missing = frame[M.FROM_M].isna() | frame[M.TO_M].isna()
    if missing.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "interval_depth_missing",
                f"{int(missing.sum())} rows have a blank or non-numeric {from_col} or {to_col}",
                source=name,
                count=int(missing.sum()),
            )
        )

    bad_order = (frame[M.FROM_M] >= frame[M.TO_M]) & ~missing
    if bad_order.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "interval_order",
                f"{int(bad_order.sum())} rows have {from_col} at or beyond {to_col}",
                source=name,
                count=int(bad_order.sum()),
                detail={"holes": sorted(set(frame.loc[bad_order, M.HOLE_ID]))[:20]},
            )
        )

    ordered = frame[~missing & ~bad_order].sort_values([M.HOLE_ID, M.FROM_M])
    same_hole = ordered[M.HOLE_ID].eq(ordered[M.HOLE_ID].shift())
    delta = ordered[M.FROM_M] - ordered[M.TO_M].shift()

    overlap = same_hole & (delta < -1e-9)
    if overlap.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "interval_overlap",
                f"{int(overlap.sum())} intervals overlap the previous interval in the same hole",
                source=name,
                count=int(overlap.sum()),
                detail={"holes": sorted(set(ordered.loc[overlap, M.HOLE_ID]))[:20]},
            )
        )

    gaps = same_hole & (delta > 1e-9)
    small = gaps & (delta <= cfg.domaining.max_interval_gap_m)
    large = gaps & (delta > cfg.domaining.max_interval_gap_m)
    if small.any():
        issues.append(
            Issue(
                Severity.INFO,
                "interval_gap",
                f"{int(small.sum())} gaps below max_interval_gap_m "
                f"({cfg.domaining.max_interval_gap_m} m); composites may span them",
                source=name,
                count=int(small.sum()),
            )
        )
    if large.any():
        issues.append(
            Issue(
                Severity.WARN,
                "interval_gap",
                f"{int(large.sum())} gaps exceed max_interval_gap_m "
                f"({cfg.domaining.max_interval_gap_m} m) and will break runs",
                source=name,
                count=int(large.sum()),
            )
        )
    return issues


def _check_beyond_total_depth(dataset: Dataset) -> list[Issue]:
    # duplicate hole IDs are reported by their own check; here they must simply not
    # stop the rest of the report from being produced
    collar = dataset.collar.drop_duplicates(M.HOLE_ID, keep="first")
    depths = collar.set_index(M.HOLE_ID)[M.TOTAL_DEPTH]
    if depths.isna().all():
        return []
    issues = []
    for name, frame in (("assay", dataset.assay), ("litho", dataset.litho)):
        if frame is None or frame.empty:
            continue
        limit = frame[M.HOLE_ID].map(depths)
        beyond = frame[M.TO_M] > limit + 1e-6
        if beyond.any():
            issues.append(
                Issue(
                    Severity.WARN,
                    "beyond_total_depth",
                    f"{int(beyond.sum())} {name} intervals end below the collar total depth",
                    source=name,
                    count=int(beyond.sum()),
                    detail={"holes": sorted(set(frame.loc[beyond, M.HOLE_ID]))[:20]},
                )
            )
    return issues


# ---------------------------------------------------------------------- survey


def _check_survey(dataset: Dataset, cfg: Config) -> list[Issue]:
    survey = dataset.survey
    if survey is None or survey.empty:
        return [
            Issue(Severity.ERROR, "survey_empty", "the survey file has no rows", source="survey")
        ]
    issues: list[Issue] = []

    ordered = survey.sort_values([M.HOLE_ID, M.DEPTH])
    same_hole = ordered[M.HOLE_ID].eq(ordered[M.HOLE_ID].shift())
    repeated = same_hole & ordered[M.DEPTH].eq(ordered[M.DEPTH].shift())
    if repeated.any():
        issues.append(
            Issue(
                Severity.WARN,
                "survey_duplicate_depth",
                f"{int(repeated.sum())} survey stations repeat a depth already surveyed; "
                "the first reading at each depth is used",
                source="survey",
                count=int(repeated.sum()),
            )
        )

    azimuth_col = source_column(dataset, "survey", M.AZIMUTH)
    bad_azimuth = survey[M.AZIMUTH].notna() & ((survey[M.AZIMUTH] < 0) | (survey[M.AZIMUTH] >= 360))
    if bad_azimuth.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "azimuth_range",
                f"{int(bad_azimuth.sum())} rows have {azimuth_col} outside [0, 360)",
                source="survey",
                count=int(bad_azimuth.sum()),
            )
        )

    dip_col = source_column(dataset, "survey", M.DIP)
    bad_dip = survey[M.DIP].notna() & (survey[M.DIP].abs() > 90.0)
    if bad_dip.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "dip_range",
                f"{int(bad_dip.sum())} rows have {dip_col} outside [-90, 90] after "
                f"normalising from dip_convention={cfg.conventions.dip_convention.value}",
                source="survey",
                count=int(bad_dip.sum()),
            )
        )

    # dips are normalised to negative-down: an all-positive file is the classic
    # convention error, and it would otherwise only show up as holes flying upward
    valid = survey[M.DIP].dropna()
    if len(valid) and (valid > 0).mean() > 0.9:
        issues.append(
            Issue(
                Severity.WARN,
                "dip_sign",
                f"{(valid > 0).mean():.0%} of dips point upward after normalisation. "
                f"Check conventions.dip_convention, currently "
                f"{cfg.conventions.dip_convention.value}.",
                source="survey",
                count=int((valid > 0).sum()),
            )
        )

    missing = survey[[M.DEPTH, M.DIP, M.AZIMUTH]].isna().any(axis=1)
    if missing.any():
        issues.append(
            Issue(
                Severity.WARN,
                "survey_incomplete",
                f"{int(missing.sum())} survey stations have a missing depth, dip or azimuth "
                "and are ignored when building the trace",
                source="survey",
                count=int(missing.sum()),
            )
        )
    return issues


# --------------------------------------------------------------------- density


def _check_density(dataset: Dataset) -> list[Issue]:
    issues = []
    low, high = DENSITY_RANGE_T_M3
    for name, frame in (("assay", dataset.assay), ("block_model", dataset.block_model)):
        if frame is None or M.DENSITY not in frame.columns:
            continue
        values = pd.to_numeric(frame[M.DENSITY], errors="coerce")
        blank = values.isna()
        if blank.any():
            issues.append(
                Issue(
                    Severity.WARN,
                    "density_missing",
                    f"{int(blank.sum())} {name} rows have no density; mass estimates for those "
                    "intervals fall back to the configured constant",
                    source=name,
                    count=int(blank.sum()),
                )
            )
        implausible = values.notna() & ((values < low) | (values > high))
        if implausible.any():
            issues.append(
                Issue(
                    Severity.ERROR,
                    "density_range",
                    f"{int(implausible.sum())} {name} densities fall outside {low}-{high} t/m3 "
                    "after unit normalisation. Check conventions.density_units.",
                    source=name,
                    count=int(implausible.sum()),
                )
            )
    return issues


# ----------------------------------------------------------------- block model


def _check_block_model(dataset: Dataset, cfg: Config) -> list[Issue]:
    blocks = dataset.block_model
    if blocks is None or blocks.empty:
        return [
            Issue(
                Severity.ERROR,
                "block_model_empty",
                "the block model has no rows, so no allocation target can be computed",
                source="block_model",
            )
        ]
    issues: list[Issue] = []

    centroids = blocks[[M.BLOCK_X, M.BLOCK_Y, M.BLOCK_Z]]
    duplicated = centroids.duplicated(keep=False)
    if duplicated.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "block_duplicate_centroid",
                f"{int(duplicated.sum())} blocks share a centroid with another block. "
                "Nearest-centroid assignment would be arbitrary between them.",
                source="block_model",
                count=int(duplicated.sum()),
            )
        )

    scheduled = is_scheduled(blocks[M.PERIOD], cfg.block_model_options.excluded_periods)
    if not scheduled.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "period_empty",
                f"no block carries a scheduled period in "
                f"{source_column(dataset, 'block_model', M.PERIOD)}. "
                "Allocation targets come from scheduled tonnage, so nothing can be allocated.",
                source="block_model",
            )
        )
    else:
        unscheduled = int((~scheduled).sum())
        if unscheduled:
            issues.append(
                Issue(
                    Severity.INFO,
                    "period_unscheduled",
                    f"{unscheduled} of {len(blocks)} blocks have no period or an excluded "
                    "period; they are not counted toward allocation targets",
                    source="block_model",
                    count=unscheduled,
                )
            )

    zero_volume = (blocks[[M.BLOCK_DX, M.BLOCK_DY, M.BLOCK_DZ]] <= 0).any(axis=1)
    if zero_volume.any():
        issues.append(
            Issue(
                Severity.ERROR,
                "block_dimension",
                f"{int(zero_volume.sum())} blocks have a zero or negative dimension",
                source="block_model",
                count=int(zero_volume.sum()),
            )
        )
    return issues


def _check_domain_lookup(dataset: Dataset) -> list[Issue]:
    codes = set(dataset.litho[M.LOGGED_CODE].dropna().unique())
    known = set(dataset.domain_lookup[M.LOGGED_CODE])
    unmapped = sorted(codes - known)
    if not unmapped:
        return []
    return [
        Issue(
            Severity.ERROR,
            "logged_code_unmapped",
            f"{len(unmapped)} logged codes are absent from the domain lookup: {unmapped[:20]}. "
            "Silently dropping them would distort the allocation.",
            source="domain_lookup",
            count=len(unmapped),
            detail={"codes": unmapped},
        )
    ]


# ---------------------------------------------------- post-geometry sanity


def check_toe_above_collar(traces: dict, collar: pd.DataFrame) -> list[Issue]:
    """A hole whose toe is above its collar means the dip convention is inverted.

    This is the check that catches a `positive_down` file read as `negative_down` before
    the error reaches the block model, where it would look like ordinary off-model drilling.
    """
    offenders = [
        hole_id for hole_id, trace in traces.items() if trace.toe[2] > trace.collar[2] + 1e-6
    ]
    if not offenders:
        return []
    return [
        Issue(
            Severity.ERROR,
            "toe_above_collar",
            f"{len(offenders)} holes desurvey to a toe above their collar RL: {offenders[:10]}. "
            "This is almost always an inverted conventions.dip_convention.",
            source="survey",
            count=len(offenders),
            detail={"holes": offenders[:50]},
        )
    ]


def check_model_extent(intervals: pd.DataFrame, *, error_fraction: float) -> list[Issue]:
    """Too much drilling outside the model means a coordinate or unit mismatch."""
    if intervals.empty or M.OUTSIDE_MODEL not in intervals.columns:
        return []
    fraction = float(intervals[M.OUTSIDE_MODEL].mean())
    if fraction <= error_fraction:
        return []
    return [
        Issue(
            Severity.ERROR,
            "off_model_extent",
            f"{fraction:.0%} of intervals fall outside the block model extents, above the "
            f"{error_fraction:.0%} threshold. This usually means the drilling and the model "
            "are in different coordinate systems or different length units, not that the "
            "drilling is genuinely off-model.",
            source="block_model",
            count=int(intervals[M.OUTSIDE_MODEL].sum()),
        )
    ]


def report_frame(issues: list[Issue]) -> pd.DataFrame:
    """Validation issues as a table, ordered most severe first."""
    if not issues:
        return pd.DataFrame(columns=["severity", "check", "source", "hole_id", "count", "message"])
    frame = pd.DataFrame([i.as_row() for i in issues])
    order = {Severity.ERROR.value: 0, Severity.WARN.value: 1, Severity.INFO.value: 2}
    return (
        frame.assign(_o=frame["severity"].map(order))
        .sort_values(["_o", "check", "source"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def summarise(issues: list[Issue]) -> dict[str, int]:
    counts = {s.value: 0 for s in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    return counts


__all__ = [
    "check_model_extent",
    "check_toe_above_collar",
    "report_frame",
    "summarise",
    "validate_inputs",
]
