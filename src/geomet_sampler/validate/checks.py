"""Data quality checks producing structured issues.

Nothing here modifies data and nothing is dropped. Every check produces Issue records
that end up in the validation report, so anything the pipeline later ignores has already
been named and counted.

All checks work on canonical field names. Where a message has to name a source column
for the user's benefit, the reader's mapping table translates it back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models as M
from ..config import Config
from ..domains import is_scheduled
from ..io.readers import Dataset, source_column
from ..models import Issue, Severity
from ..units import DENSITY_RANGE_T_M3, FEET_TO_METRES

#: Below this, an interval gap is normal core handling; above it, worth reporting.
GAP_INFO_TOLERANCE_M = 0.10


#: Supplied coordinates are compared with the depths they belong to. A discrepancy
#: larger than this fraction of the interval length, or than the absolute floor for short
#: intervals, is reported. The floor absorbs coordinates rounded to a few centimetres.
GEOMETRY_REL_TOL = 0.05
GEOMETRY_ABS_TOL_M = 0.05
#: A coordinate-to-depth length ratio within this fraction of 1/0.3048 or 0.3048 is
#: read as one side in feet and the other in metres, rather than a general mismatch.
UNIT_RATIO_TOL = 0.05
#: Above this fraction of holes rising with depth, the orientation is probably wrong.
UPWARD_MAJORITY_FRACTION = 0.5
#: Where the section view of collars and toes is written, relative to output_dir.
HOLE_ORIENTATION_PLOT = "plots/hole_orientation.png"


def validate_inputs(dataset: Dataset, cfg: Config) -> list[Issue]:
    """Every check that can run before desurvey."""
    issues: list[Issue] = []
    if dataset.samples is not None:
        issues += _checks_not_applicable_to_samples()
        issues += _check_intervals(dataset.samples, "samples", dataset, cfg)
        issues += check_supplied_geometry(dataset.samples)
    else:
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
    for name, frame in (
        ("assay", dataset.assay),
        ("samples", dataset.samples),
        ("block_model", dataset.block_model),
    ):
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
    logged = dataset.samples if dataset.samples is not None else dataset.litho
    codes = set(logged[M.LOGGED_CODE].dropna().unique())
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


# ---------------------------------------------------- supplied geometry


def _checks_not_applicable_to_samples() -> list[Issue]:
    """Say which checks did not run, so their absence is visible in the report."""
    return [
        Issue(
            Severity.INFO,
            "geometry_supplied",
            "coordinates taken from the desurveyed samples table. Collar, survey, orphan "
            "and total-depth checks do not apply and were not run; the supplied-geometry "
            "checks below replace them.",
            source="samples",
        )
    ]


def check_supplied_geometry(samples: pd.DataFrame) -> list[Issue]:
    """Test supplied coordinates against the depths they claim to belong to.

    Without a survey there is no dip convention to invert, so the silent failures here
    are different: depths and coordinates in different units, an inverted Z, or rows
    whose coordinates belong to another interval. Each one shows up as a disagreement
    between the along-hole length and the 3D geometry.
    """
    issues: list[Issue] = []
    coords = [*M.FROM_XYZ, *M.TO_XYZ, *M.MID_XYZ]
    missing = samples[coords].isna().any(axis=1)
    if missing.any():
        issues.append(
            Issue(
                Severity.WARN,
                "coordinates_missing",
                f"{int(missing.sum())} sample rows have a blank or non-numeric from, to or mid "
                "coordinate. They cannot be placed in the block model and will be flagged "
                "outside_model.",
                source="samples",
                count=int(missing.sum()),
                detail={"holes": sorted(set(samples.loc[missing, M.HOLE_ID]))[:20]},
            )
        )
    frame = samples[~missing & (samples[M.TO_M] > samples[M.FROM_M])]
    if frame.empty:
        return issues
    issues += _check_row_lengths(frame)
    issues += _check_midpoints(frame)
    issues += _check_row_continuity(frame)
    issues += _check_vertical_direction(frame)
    return issues


def _xyz(frame: pd.DataFrame, cols: tuple[str, str, str]) -> np.ndarray:
    return frame[list(cols)].to_numpy(dtype=float)


def _tolerance(length: np.ndarray) -> np.ndarray:
    return np.maximum(GEOMETRY_ABS_TOL_M, GEOMETRY_REL_TOL * length)


def _check_row_lengths(frame: pd.DataFrame) -> list[Issue]:
    """The straight line from `from` to `to` should match the depth interval.

    On a curved hole the chord is slightly shorter than the arc, but over a single
    sample the difference is far inside the tolerance.
    """
    along = (frame[M.TO_M] - frame[M.FROM_M]).to_numpy(dtype=float)
    chord = np.linalg.norm(_xyz(frame, M.TO_XYZ) - _xyz(frame, M.FROM_XYZ), axis=1)
    bad = np.abs(chord - along) > _tolerance(along)
    if not bad.any():
        return []

    ratio = chord / along
    coords_feet = bad & (np.abs(ratio - 1 / FEET_TO_METRES) < UNIT_RATIO_TOL / FEET_TO_METRES)
    depths_feet = bad & (np.abs(ratio - FEET_TO_METRES) < UNIT_RATIO_TOL * FEET_TO_METRES)
    other = bad & ~coords_feet & ~depths_feet
    issues = []
    for mask, which in (
        (coords_feet, "coordinates look like feet while depths look like metres"),
        (depths_feet, "depths look like feet while coordinates look like metres"),
    ):
        if mask.any():
            issues.append(
                Issue(
                    Severity.ERROR,
                    "geometry_unit_mismatch",
                    f"{int(mask.sum())} rows: the 3D distance from the from-point to the "
                    f"to-point is about {np.median(ratio[mask]):.3f} times the depth interval, "
                    f"so {which}. conventions.length_units applies to both; convert one of "
                    "them in the source table.",
                    source="samples",
                    count=int(mask.sum()),
                    detail={"holes": sorted(set(frame.loc[mask, M.HOLE_ID]))[:20]},
                )
            )
    if other.any():
        issues.append(
            Issue(
                Severity.WARN,
                "geometry_length_mismatch",
                f"{int(other.sum())} rows: the 3D distance from the from-point to the to-point "
                f"differs from to_m - from_m by more than {GEOMETRY_REL_TOL:.0%} "
                f"(or {GEOMETRY_ABS_TOL_M} m). The coordinates may belong to a different "
                "interval, or the table was desurveyed with different depths.",
                source="samples",
                count=int(other.sum()),
                detail={"holes": sorted(set(frame.loc[other, M.HOLE_ID]))[:20]},
            )
        )
    return issues


def _check_midpoints(frame: pd.DataFrame) -> list[Issue]:
    """The midpoint should sit halfway between the from- and to-points."""
    along = (frame[M.TO_M] - frame[M.FROM_M]).to_numpy(dtype=float)
    halfway = (_xyz(frame, M.FROM_XYZ) + _xyz(frame, M.TO_XYZ)) / 2.0
    offset = np.linalg.norm(_xyz(frame, M.MID_XYZ) - halfway, axis=1)
    bad = offset > _tolerance(along)
    if not bad.any():
        return []
    return [
        Issue(
            Severity.WARN,
            "geometry_midpoint_offset",
            f"{int(bad.sum())} rows have a midpoint more than {GEOMETRY_REL_TOL:.0%} of the "
            f"interval length (or {GEOMETRY_ABS_TOL_M} m) away from halfway between their "
            "from and to points. The midpoint places the row in the block model, so check "
            "which columns are mapped to x_mid, y_mid and z_mid.",
            source="samples",
            count=int(bad.sum()),
            detail={"holes": sorted(set(frame.loc[bad, M.HOLE_ID]))[:20]},
        )
    ]


def _check_row_continuity(frame: pd.DataFrame) -> list[Issue]:
    """Where one row ends at the depth the next begins, their points should coincide."""
    ordered = frame.sort_values([M.HOLE_ID, M.FROM_M])
    same_hole = ordered[M.HOLE_ID].eq(ordered[M.HOLE_ID].shift()).to_numpy()
    touching = same_hole & np.isclose(
        ordered[M.FROM_M].to_numpy(dtype=float), ordered[M.TO_M].shift().to_numpy(dtype=float)
    )
    if not touching.any():
        return []
    jump = np.linalg.norm(
        _xyz(ordered, M.FROM_XYZ) - np.roll(_xyz(ordered, M.TO_XYZ), 1, axis=0), axis=1
    )
    bad = touching & (jump > GEOMETRY_ABS_TOL_M)
    if not bad.any():
        return []
    return [
        Issue(
            Severity.WARN,
            "geometry_discontinuous",
            f"{int(bad.sum())} rows start at the depth where the previous row ends, but their "
            f"from-point is more than {GEOMETRY_ABS_TOL_M} m from its to-point. Rows may be "
            "attached to the wrong hole, or two desurveys have been mixed in one table.",
            source="samples",
            count=int(bad.sum()),
            detail={"holes": sorted(set(ordered.loc[bad, M.HOLE_ID]))[:20]},
        )
    ]


def _check_vertical_direction(frame: pd.DataFrame) -> list[Issue]:
    """Flag holes whose rows both rise and fall with depth.

    Holes that rise throughout are handled by :func:`check_hole_direction`, which runs
    for both input layouts.
    """
    along = (frame[M.TO_M] - frame[M.FROM_M]).to_numpy(dtype=float)
    dz = (frame[M.Z_TO] - frame[M.Z_FROM]).to_numpy(dtype=float)
    # near-horizontal rows say nothing about direction
    steep = np.abs(dz) > 0.01 * along
    rising = pd.Series(steep & (dz > 0), index=frame.index).groupby(frame[M.HOLE_ID]).any()
    falling = pd.Series(steep & (dz < 0), index=frame.index).groupby(frame[M.HOLE_ID]).any()
    reversing = sorted(rising[rising & falling].index)
    if not reversing:
        return []
    return [
        Issue(
            Severity.WARN,
            "hole_changes_vertical_direction",
            f"{len(reversing)} holes both rise and fall with depth: {reversing[:10]}. "
            "Rare in a real hole; check for mixed-up rows or a partly inverted Z.",
            source="samples",
            count=len(reversing),
            detail={"holes": reversing[:50]},
        )
    ]


# ---------------------------------------------------- post-geometry sanity


def hole_ends(intervals: pd.DataFrame, traces: dict) -> pd.DataFrame:
    """Collar and toe coordinates of every hole, and whether it rises with depth.

    Desurveyed holes use their trace, which runs from the collar to total depth. Holes
    with supplied coordinates use the top of their shallowest row and the bottom of
    their deepest, which is the collar and toe only if the table covers the whole hole.
    """
    rows = []
    for hole_id, trace in traces.items():
        rows.append((hole_id, *trace.collar, *trace.toe))
    placed = intervals[~intervals[M.HOLE_ID].isin(set(traces))]
    if set(M.FROM_XYZ + M.TO_XYZ) <= set(placed.columns):
        placed = placed.dropna(subset=[*M.FROM_XYZ, *M.TO_XYZ])
    else:
        placed = placed.iloc[0:0]
    for hole_id, group in placed.groupby(M.HOLE_ID, sort=True):
        top = group.loc[group[M.FROM_M].idxmin()]
        bottom = group.loc[group[M.TO_M].idxmax()]
        rows.append(
            (
                hole_id,
                *top[list(M.FROM_XYZ)].to_numpy(float),
                *bottom[list(M.TO_XYZ)].to_numpy(float),
            )
        )
    ends = pd.DataFrame(rows, columns=[M.HOLE_ID, *M.COLLAR_XYZ, *M.TOE_XYZ])
    ends[M.RISES] = ends[M.TOE_XYZ[2]] > ends[M.COLLAR_XYZ[2]] + 1e-6
    return ends.sort_values(M.HOLE_ID).reset_index(drop=True)


def check_hole_direction(ends: pd.DataFrame) -> list[Issue]:
    """List the holes that rise with depth; warn if they are the majority.

    Upward holes are legitimate from underground, and only the user knows which holes
    were drilled that way, so a single rising hole is never an error. What a wrong
    dip convention or an inverted Z looks like is most of the holes rising at once.
    The section view written with the run is how the user confirms either way.
    """
    if ends.empty:
        return []
    upward = sorted(ends.loc[ends[M.RISES], M.HOLE_ID])
    if not upward:
        return []
    fraction = len(upward) / len(ends)
    issues = [
        Issue(
            Severity.INFO,
            "hole_rises_with_depth",
            f"{len(upward)} of {len(ends)} holes end above their collar: {upward[:10]}. "
            f"Correct for holes drilled upward. Confirm on {HOLE_ORIENTATION_PLOT}.",
            source="geometry",
            count=len(upward),
            detail={"holes": upward[:50]},
        )
    ]
    if fraction > UPWARD_MAJORITY_FRACTION:
        issues.append(
            Issue(
                Severity.WARN,
                "most_holes_rise",
                f"{fraction:.0%} of holes end above their collar. Unless most of this "
                "programme was drilled upward, the dip convention is inverted "
                "(conventions.dip_convention) or Z is inverted in the source data. Every "
                f"downstream result depends on this: check {HOLE_ORIENTATION_PLOT} "
                "before using the output.",
                source="geometry",
                count=len(upward),
            )
        )
    return issues


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
    "check_hole_direction",
    "check_supplied_geometry",
    "hole_ends",
    "report_frame",
    "summarise",
    "validate_inputs",
]
