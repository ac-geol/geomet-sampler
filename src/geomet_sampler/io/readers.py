"""CSV loading, encoding handling and column mapping onto the canonical schema.

This module is the boundary. Above it, user column names, units and conventions.
Below it, only canonical names in normalised units. If a user column name appears
anywhere else in the library, it is a bug.

Everything is read as text first, then parsed explicitly by :mod:`.numbers`, which
reports every value that is not a plain number. Fully quoted files, BOMs and CRLF line
endings are absorbed here rather than fixed by hand in the source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import models as M
from ..config import AssaySource, BlockModelSource, Config, GradeUnits, SourceBase
from ..errors import MappingError
from ..models import Availability, Issue, Severity
from ..units import (
    azimuth_to_grid,
    density_to_t_m3,
    dip_to_negative_down,
    grade_to_ppm,
    length_to_metres,
)
from .numbers import NumberRules, parse_grades, parse_numbers

ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1252")


@dataclass
class Dataset:
    """Every input, in canonical names and normalised units."""

    domain_lookup: pd.DataFrame
    block_model: pd.DataFrame
    availability: pd.DataFrame
    #: separate-file layout; all four are None when a samples table is supplied
    collar: pd.DataFrame | None = None
    survey: pd.DataFrame | None = None
    assay: pd.DataFrame | None = None
    litho: pd.DataFrame | None = None
    #: pre-desurveyed layout: assay fields, logged code and coordinates in one table
    samples: pd.DataFrame | None = None
    existing_testwork: pd.DataFrame | None = None
    planned_collar: pd.DataFrame | None = None
    planned_survey: pd.DataFrame | None = None
    #: canonical name -> user column, per source, for translating messages back
    mappings: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Tier 2 fallbacks that were actually used, reported in the run summary
    fallbacks: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ raw reading


def _read_text(path: Path, encoding: str) -> pd.DataFrame:
    """Read a CSV entirely as strings, so quoting and mixed types cannot mislead us."""
    encodings = ENCODING_CANDIDATES if encoding == "auto" else (encoding,)
    last: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError as exc:  # pragma: no cover - depends on input bytes
            last = exc
            continue
        # a BOM survives a plain utf-8 read and corrupts the first header
        if df.columns[0].startswith("﻿"):
            if enc != encodings[-1]:
                continue
            df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("﻿")})
        df.columns = [str(c).strip() for c in df.columns]
        return df
    raise last if last else RuntimeError(f"could not read {path}")


def _rules(src: SourceBase, source_name: str) -> NumberRules:
    return NumberRules(
        source=source_name,
        file=src.path.name,
        null_values=tuple(src.null_values),
        negative_is_below_detection=getattr(src, "negative_is_below_detection", False),
    )


def _num(
    df: pd.DataFrame, col: str, src: SourceBase, rules: NumberRules, issues: list[Issue]
) -> pd.Series:
    """Parse one numeric column, adding what was found to ``issues``.

    ``col`` is a canonical name in a renamed frame, or a user column; messages always
    name the user's column.
    """
    values, found = parse_numbers(df[col], src.user_column(col) or col, rules)
    issues.extend(found)
    return values


def _grade(df: pd.DataFrame, col: str, rules: NumberRules, issues: list[Issue]) -> pd.Series:
    """Parse one grade column (a user column), adding what was found to ``issues``."""
    values, found = parse_grades(df[col], col, rules)
    issues.extend(found)
    return values


def _clean_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _check_mapping(df: pd.DataFrame, src: SourceBase, source_name: str) -> list[str]:
    """Return a human-readable problem per configured column that is not in the file."""
    problems = []
    present = set(df.columns)
    for canonical, user_col in src.mapped().items():
        if user_col not in present:
            problems.append(
                f"{source_name}.columns.{canonical}: {user_col!r} not found in {src.path.name}. "
                f"Columns present: {sorted(present)}"
            )
    for extra in src.carry_through:
        if extra not in present:
            problems.append(
                f"{source_name}.carry_through: {extra!r} not found in {src.path.name}. "
                f"Columns present: {sorted(present)}"
            )
    return problems


def _extra_column_problems(
    df: pd.DataFrame, wanted: dict[str, str | None], label: str, path: Path
) -> list[str]:
    """Check non-``columns`` references (density, flags, elements, attributes)."""
    problems = []
    present = set(df.columns)
    for what, user_col in wanted.items():
        if user_col is not None and user_col not in present:
            problems.append(
                f"{label}.{what}: {user_col!r} not found in {path.name}. "
                f"Columns present: {sorted(present)}"
            )
    return problems


def _rename(df: pd.DataFrame, src: SourceBase) -> pd.DataFrame:
    return df.rename(columns={user: canon for canon, user in src.mapped().items()})


# ------------------------------------------------------------- source readers


def read_collar(
    cfg: Config, src=None, *, planned: bool = False
) -> tuple[pd.DataFrame, list[Issue]]:
    src = src or cfg.sources.collar
    name = "planned_collar" if planned else "collar"
    raw = _read_text(src.path, src.encoding)
    _raise_if(_check_mapping(raw, src, f"sources.{name}"))
    df = _rename(raw, src)
    rules = _rules(src, name)
    issues: list[Issue] = []
    conv = cfg.conventions
    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    for col in (M.EAST, M.NORTH, M.RL):
        out[col] = length_to_metres(_num(df, col, src, rules, issues), conv.length_units)

    if M.TOTAL_DEPTH in df.columns:
        out[M.TOTAL_DEPTH] = length_to_metres(
            _num(df, M.TOTAL_DEPTH, src, rules, issues), conv.length_units
        )
    else:
        out[M.TOTAL_DEPTH] = np.nan
        issues.append(
            Issue(
                Severity.INFO,
                "tier2_fallback",
                "collar total_depth not mapped, derived from survey and interval depths",
                source="collar",
            )
        )
    for extra in src.carry_through:
        out[extra] = raw[extra]
    out[M.IS_PLANNED] = planned
    return out, issues


def read_survey(
    cfg: Config, src=None, *, planned: bool = False
) -> tuple[pd.DataFrame, list[Issue]]:
    src = src or cfg.sources.survey
    name = "planned_survey" if planned else "survey"
    raw = _read_text(src.path, src.encoding)
    _raise_if(_check_mapping(raw, src, f"sources.{name}"))
    df = _rename(raw, src)
    rules = _rules(src, name)
    issues: list[Issue] = []
    conv = cfg.conventions
    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    out[M.DEPTH] = length_to_metres(_num(df, M.DEPTH, src, rules, issues), conv.length_units)
    out[M.DIP] = dip_to_negative_down(_num(df, M.DIP, src, rules, issues), conv)
    out[M.AZIMUTH] = azimuth_to_grid(_num(df, M.AZIMUTH, src, rules, issues), conv)
    return out.sort_values([M.HOLE_ID, M.DEPTH]).reset_index(drop=True), issues


def read_assay(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    src = cfg.sources.assay
    raw = _read_text(src.path, src.encoding)
    _raise_if(_interval_mapping_problems(raw, src, "sources.assay", cfg))
    out, issues = _interval_fields(raw, src, "assay", cfg)
    return out.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True), issues


def read_samples(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    """Read a pre-desurveyed samples table: assay fields, logged code and coordinates.

    Coordinates go through the same length conversion as depths, once, here. Rows with a
    blank logged code keep ``None`` rather than being dropped, so they reach the
    framework and the validation report exactly as unlogged assay intervals do.
    """
    src = cfg.sources.samples
    raw = _read_text(src.path, src.encoding)
    problems = _interval_mapping_problems(raw, src, "sources.samples", cfg)
    missing = [c for c in M.SAMPLES_REQUIRED if src.user_column(c) is None]
    if missing:
        problems.append(
            f"sources.samples.columns: {', '.join(missing)} must be mapped. A desurveyed "
            "samples table needs the logged code and the from, to and mid coordinates."
        )
    _raise_if(problems)

    out, issues = _interval_fields(raw, src, "samples", cfg)
    df = _rename(raw, src)
    rules = _rules(src, "samples")
    code = df[M.LOGGED_CODE].astype(str).str.strip()
    out[M.LOGGED_CODE] = code.where(code != "", None)
    for col in (*M.FROM_XYZ, *M.TO_XYZ, *M.MID_XYZ):
        out[col] = length_to_metres(_num(df, col, src, rules, issues), cfg.conventions.length_units)
    return out.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True), issues


def _interval_mapping_problems(
    raw: pd.DataFrame, src: AssaySource, label: str, cfg: Config
) -> list[str]:
    """Every configured column an assay-like source refers to must be in the file."""
    problems = _check_mapping(raw, src, label)
    problems += _extra_column_problems(
        raw, {"source": src.density.source}, f"{label}.density", src.path
    )
    problems += _extra_column_problems(
        raw,
        {f"{name}.source": spec.source for name, spec in src.flags.items()},
        f"{label}.flags",
        src.path,
    )
    problems += _extra_column_problems(
        raw,
        {f"{n}.drillhole.field": s.drillhole.field for n, s in cfg.elements.items()},
        "elements",
        src.path,
    )
    return problems


def _interval_fields(
    raw: pd.DataFrame, src: AssaySource, source_name: str, cfg: Config
) -> tuple[pd.DataFrame, list[Issue]]:
    """Depths, sample ID, density, flags, grades and attributes, in canonical form."""
    df = _rename(raw, src)
    conv = cfg.conventions
    rules = _rules(src, source_name)
    issues: list[Issue] = []

    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    for col in (M.FROM_M, M.TO_M):
        out[col] = length_to_metres(_num(df, col, src, rules, issues), conv.length_units)

    generated = (
        out[M.HOLE_ID] + "_" + out[M.FROM_M].map(lambda v: f"{v:.2f}" if pd.notna(v) else "NA")
    )
    if M.SAMPLE_ID in df.columns:
        ids = _clean_id(df[M.SAMPLE_ID])
        blank = ids == ""
        out[M.SAMPLE_ID] = ids.where(~blank, generated)
        if blank.any():
            issues.append(
                Issue(
                    Severity.WARN,
                    "sample_id_blank",
                    f"{int(blank.sum())} rows have a blank sample ID, usually unsampled core. "
                    "IDs generated as {hole_id}_{from_m} so each row stays traceable; these "
                    "rows have no lab number for the core shed to match.",
                    source=source_name,
                    count=int(blank.sum()),
                )
            )
    else:
        out[M.SAMPLE_ID] = generated
        issues.append(
            Issue(
                Severity.WARN,
                "tier2_fallback",
                "no sample_id column mapped: IDs generated as {hole_id}_{from_m}. "
                "The pick list is degraded, core shed staff cannot match a lab number.",
                source=source_name,
            )
        )

    out[M.DENSITY], dens_issues = _density_column(df, src, conv, rules)
    issues += dens_issues

    flag = src.flags.get(M.LOW_RECOVERY)
    if flag is not None and flag.source is not None:
        out[M.LOW_RECOVERY] = _resolve_flag(df, flag, src, rules, issues)
    else:
        out[M.LOW_RECOVERY] = False
        issues.append(
            Issue(
                Severity.INFO,
                "tier2_fallback",
                "no low_recovery flag mapped: no recovery-based compositing breaks",
                source=source_name,
            )
        )

    for name, spec in cfg.elements.items():
        if spec.drillhole.field is not None:
            values = _grade(df, spec.drillhole.field, rules, issues)
            out[M.elem_col(name)] = grade_to_ppm(values, spec.drillhole.units)

    for role, spec in cfg.attributes.items():
        if spec.drillhole is not None and spec.drillhole in raw.columns:
            out[M.dh_attr_col(role)] = raw[spec.drillhole].astype(str).str.strip()

    for extra in src.carry_through:
        out[extra] = raw[extra]
    return out, issues


def read_litho(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    src = cfg.sources.litho
    raw = _read_text(src.path, src.encoding)
    problems = _check_mapping(raw, src, "sources.litho")
    litho_attrs = {
        role: spec.drillhole
        for role, spec in cfg.attributes.items()
        if spec.drillhole is not None and spec.drillhole in raw.columns
    }
    _raise_if(problems)

    df = _rename(raw, src)
    conv = cfg.conventions
    rules = _rules(src, "litho")
    issues: list[Issue] = []
    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    for col in (M.FROM_M, M.TO_M):
        out[col] = length_to_metres(_num(df, col, src, rules, issues), conv.length_units)
    out[M.LOGGED_CODE] = df[M.LOGGED_CODE].astype(str).str.strip()
    for role, user_col in litho_attrs.items():
        out[M.dh_attr_col(role)] = raw[user_col].astype(str).str.strip()
    for extra in src.carry_through:
        out[extra] = raw[extra]
    return out.sort_values([M.HOLE_ID, M.FROM_M]).reset_index(drop=True), issues


def read_domain_lookup(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    src = cfg.sources.domain_lookup
    raw = _read_text(src.path, src.encoding)
    _raise_if(_check_mapping(raw, src, "sources.domain_lookup"))
    df = _rename(raw, src)
    out = pd.DataFrame(
        {
            M.LOGGED_CODE: df[M.LOGGED_CODE].astype(str).str.strip(),
            M.GEOMET_DOMAIN: df[M.GEOMET_DOMAIN].astype(str).str.strip(),
        }
    )
    issues: list[Issue] = []
    dupes = out[out.duplicated(M.LOGGED_CODE, keep=False)]
    if not dupes.empty:
        issues.append(
            Issue(
                Severity.ERROR,
                "domain_lookup_duplicate",
                f"logged codes mapped more than once: {sorted(set(dupes[M.LOGGED_CODE]))}",
                source="domain_lookup",
                count=len(dupes),
            )
        )
    return out.drop_duplicates(M.LOGGED_CODE), issues


def read_block_model(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    src = cfg.sources.block_model
    raw = _read_text(src.path, src.encoding)
    problems = _check_mapping(raw, src, "sources.block_model")
    problems += _extra_column_problems(
        raw, {"source": src.density.source}, "sources.block_model.density", src.path
    )
    problems += _extra_column_problems(
        raw,
        {f"{n}.block_model.field": s.block_model.field for n, s in cfg.elements.items()},
        "elements",
        src.path,
    )
    problems += _extra_column_problems(
        raw,
        {f"{r}.block_model": s.block_model for r, s in cfg.attributes.items()},
        "attributes",
        src.path,
    )
    _raise_if(problems)

    df = _rename(raw, src)
    conv = cfg.conventions
    rules = _rules(src, "block_model")
    issues: list[Issue] = []
    out = pd.DataFrame(index=df.index)
    for col in (M.BLOCK_X, M.BLOCK_Y, M.BLOCK_Z, M.BLOCK_DX, M.BLOCK_DY, M.BLOCK_DZ):
        out[col] = length_to_metres(_num(df, col, src, rules, issues), conv.length_units)
    out[M.PERIOD] = df[M.PERIOD].astype(str).str.strip().replace({"": None, "nan": None})
    out[M.DENSITY], dens_issues = _density_column(df, src, conv, rules)
    issues += dens_issues

    for role, spec in cfg.attributes.items():
        if spec.block_model is not None:
            out[M.bm_attr_col(role)] = raw[spec.block_model].astype(str).str.strip()
    for name, spec in cfg.elements.items():
        if spec.block_model.field is not None:
            out[M.bm_elem_col(name)] = grade_to_ppm(
                _grade(raw, spec.block_model.field, rules, issues), spec.block_model.units
            )
    out[M.BLOCK_TONNES] = out[M.BLOCK_DX] * out[M.BLOCK_DY] * out[M.BLOCK_DZ] * out[M.DENSITY]
    return out.reset_index(drop=True), issues


def read_availability(cfg: Config) -> tuple[pd.DataFrame, list[Issue]]:
    src = cfg.sources.availability
    raw = _read_text(src.path, src.encoding)
    _raise_if(_check_mapping(raw, src, "sources.availability"))
    df = _rename(raw, src)
    issues: list[Issue] = []

    lookup: dict[str, str] = {}
    for state, vocabulary in src.value_map.items():
        for word in vocabulary:
            lookup[str(word).strip().upper()] = state
    for state in Availability:
        lookup.setdefault(state.value, state.value)

    raw_state = df[M.AVAILABILITY].astype(str).str.strip().str.upper()
    mapped = raw_state.map(lookup)
    unresolved = sorted(set(raw_state[mapped.isna()]))
    if unresolved:
        issues.append(
            Issue(
                Severity.ERROR,
                "availability_unmapped",
                f"availability values not covered by value_map: {unresolved}. "
                "Unmapped holes would fail closed and silently shrink the candidate pool.",
                source="availability",
                count=int(mapped.isna().sum()),
            )
        )

    rules = _rules(src, "availability")
    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    out[M.AVAILABILITY] = mapped.fillna(Availability.UNAVAILABLE.value)
    for col in (M.CORE_DIAMETER_MM, M.REMAINING_FRACTION):
        out[col] = (
            _num(df, col, src, rules, issues)
            if col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
    out[M.CORE_DIAMETER_MM] = out[M.CORE_DIAMETER_MM].fillna(cfg.mass.default_core_diameter_mm)
    out[M.REMAINING_FRACTION] = out[M.REMAINING_FRACTION].fillna(
        cfg.mass.default_remaining_fraction
    )
    return out.drop_duplicates(M.HOLE_ID).reset_index(drop=True), issues


def read_existing_testwork(cfg: Config) -> tuple[pd.DataFrame | None, list[Issue]]:
    src = cfg.sources.existing_testwork
    if src is None:
        return None, []
    raw = _read_text(src.path, src.encoding)
    _raise_if(_check_mapping(raw, src, "sources.existing_testwork"))
    df = _rename(raw, src)
    conv = cfg.conventions
    rules = _rules(src, "existing_testwork")
    issues: list[Issue] = []
    out = pd.DataFrame({M.HOLE_ID: _clean_id(df[M.HOLE_ID])})
    for col in (M.FROM_M, M.TO_M):
        out[col] = (
            length_to_metres(_num(df, col, src, rules, issues), conv.length_units)
            if col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
    out[M.SAMPLE_ID] = _clean_id(df[M.SAMPLE_ID]) if M.SAMPLE_ID in df.columns else out[M.HOLE_ID]
    out[M.GEOMET_DOMAIN] = (
        df[M.GEOMET_DOMAIN].astype(str).str.strip().replace({"": None})
        if M.GEOMET_DOMAIN in df.columns
        else None
    )
    out["test_package"] = (
        df["test_package"].astype(str).str.strip() if "test_package" in df.columns else ""
    )
    return out, issues


# ---------------------------------------------------------------- shared parts


def _density_column(
    df: pd.DataFrame, src: AssaySource | BlockModelSource, conv, rules: NumberRules
) -> tuple[pd.Series, list[Issue]]:
    spec, source_name = src.density, rules.source
    issues: list[Issue] = []
    if spec.source is not None:
        values = density_to_t_m3(_num(df, spec.source, src, rules, issues), conv.density_units)
        if spec.fallback_constant is not None and values.isna().any():
            n = int(values.isna().sum())
            values = values.fillna(spec.fallback_constant)
            issues.append(
                Issue(
                    Severity.WARN,
                    "density_fallback",
                    f"{n} rows had no density, filled with fallback_constant "
                    f"{spec.fallback_constant}",
                    source=source_name,
                    count=n,
                )
            )
    else:
        values = pd.Series(float(spec.fallback_constant), index=df.index)
        issues.append(
            Issue(
                Severity.INFO,
                "tier2_fallback",
                f"no density column mapped, using constant {spec.fallback_constant} t/m3",
                source=source_name,
            )
        )
    return values, issues


def _resolve_flag(
    df: pd.DataFrame, spec, src: SourceBase, rules: NumberRules, issues: list[Issue]
) -> pd.Series:
    raw = df[spec.source]
    if spec.type == "boolean":
        truthy = {str(v).strip().upper() for v in spec.true_values}
        return raw.astype(str).str.strip().str.upper().isin(truthy)
    values = _num(df, spec.source, src, rules, issues)
    if spec.direction == "below":
        return values < spec.threshold
    return values > spec.threshold


def _raise_if(problems: list[str]) -> None:
    if problems:
        raise MappingError("\n".join(problems))


# -------------------------------------------------------------------- load all


def load_all(cfg: Config) -> tuple[Dataset, list[Issue]]:
    """Read every configured source, reporting all mapping problems in one go."""
    issues: list[Issue] = []
    problems: list[str] = []
    frames: dict[str, object] = {}

    if cfg.sources.uses_samples_table:
        readers = [("samples", lambda: read_samples(cfg))]
    else:
        readers = [
            ("collar", lambda: read_collar(cfg)),
            ("survey", lambda: read_survey(cfg)),
            ("assay", lambda: read_assay(cfg)),
            ("litho", lambda: read_litho(cfg)),
        ]
    readers += [
        ("domain_lookup", lambda: read_domain_lookup(cfg)),
        ("block_model", lambda: read_block_model(cfg)),
        ("availability", lambda: read_availability(cfg)),
        ("existing_testwork", lambda: read_existing_testwork(cfg)),
    ]
    if cfg.sources.planned_collar is not None:
        readers.append(
            ("planned_collar", lambda: read_collar(cfg, cfg.sources.planned_collar, planned=True))
        )
    if cfg.sources.planned_survey is not None:
        readers.append(
            ("planned_survey", lambda: read_survey(cfg, cfg.sources.planned_survey, planned=True))
        )

    for name, reader in readers:
        try:
            frame, frame_issues = reader()
        except MappingError as exc:
            problems.append(str(exc))
            frames[name] = None
            continue
        frames[name] = frame
        issues.extend(frame_issues)

    if problems:
        raise MappingError("\n".join(problems))

    mappings = {
        name: getattr(cfg.sources, name).mapped()
        for name in type(cfg.sources).model_fields
        if getattr(cfg.sources, name) is not None
    }
    dataset = Dataset(
        collar=frames.get("collar"),
        survey=frames.get("survey"),
        assay=frames.get("assay"),
        litho=frames.get("litho"),
        samples=frames.get("samples"),
        domain_lookup=frames["domain_lookup"],
        block_model=frames["block_model"],
        availability=frames["availability"],
        existing_testwork=frames["existing_testwork"],
        planned_collar=frames.get("planned_collar"),
        planned_survey=frames.get("planned_survey"),
        mappings=mappings,
        fallbacks=[i.message for i in issues if i.check == "tier2_fallback"],
    )
    return dataset, issues


def source_column(dataset: Dataset, source: str, canonical: str) -> str:
    """Translate a canonical field back to the user's column name, for messages."""
    return dataset.mappings.get(source, {}).get(canonical, canonical)


__all__ = [
    "Dataset",
    "GradeUnits",
    "load_all",
    "read_assay",
    "read_availability",
    "read_block_model",
    "read_collar",
    "read_domain_lookup",
    "read_existing_testwork",
    "read_litho",
    "read_samples",
    "read_survey",
    "source_column",
]
