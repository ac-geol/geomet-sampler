"""Draft a project config by inspecting the user's files.

This is the only place in the tool that guesses. The pipeline itself either has an
explicit mapping or errors, because a mapping guessed at runtime is a mapping nobody
checked. Everything inferred here is marked ``# CONFIRM`` and the result is a starting
point for review, never a config to run unedited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .io.readers import _read_text, _to_num
from .units import DENSITY_RANGE_T_M3

CONFIRM = "  # CONFIRM"

#: Name synonyms for the structural fields, checked after lowercasing and stripping
#: separators, so `Hole_ID`, `holeid` and `BHID` all resolve.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "hole_id": ("holeid", "bhid", "dhid", "drillhole", "hole", "holenumber", "borehole"),
    "east": ("east", "easting", "x", "xcollar", "gridx", "utme"),
    "north": ("north", "northing", "y", "ycollar", "gridy", "utmn"),
    "rl": ("rl", "elevation", "elev", "z", "zcollar", "collarrl"),
    "total_depth": ("totaldepth", "depth", "eoh", "holelength", "length", "lengthm", "maxdepth"),
    "depth": ("depth", "depthm", "at", "md", "measureddepth", "station"),
    "dip": ("dip", "inclination", "incl", "plunge"),
    "azimuth": ("azimuth", "azi", "bearing", "brg"),
    "from_m": ("from", "fromm", "depthfrom", "start", "top", "fromdepth"),
    "to_m": ("to", "tom", "depthto", "end", "bottom", "todepth"),
    "sample_id": ("sampleid", "sample", "sampleno", "samplenumber", "labid"),
    "logged_code": ("code", "lith", "litho", "lithology", "rocktype", "logcode", "unit"),
    "x": ("x", "xc", "xcentre", "xcenter", "east", "easting"),
    "y": ("y", "yc", "ycentre", "ycenter", "north", "northing"),
    "z": ("z", "zc", "zcentre", "zcenter", "rl", "elevation"),
    "dx": ("dx", "xinc", "xsize", "blockx"),
    "dy": ("dy", "yinc", "ysize", "blocky"),
    "dz": ("dz", "zinc", "zsize", "blockz"),
    "period": ("period", "mineperiod", "schedule", "stage", "year", "phase"),
}

#: Suffixes that mark a numeric column as a grade, mapped to their declared units.
ELEMENT_SUFFIXES = {
    "pct": "pct",
    "perc": "pct",
    "ppm": "ppm",
    "gpt": "gpt",
    "gt": "gpt",
    "ozt": "oz_t",
}

DENSITY_HINTS = ("density", "sg", "bd", "bulkdensity", "tonnesm3", "specificgravity")
#: A string column with fewer distinct values than this is a candidate attribute.
MAX_ATTRIBUTE_CARDINALITY = 50


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass
class Inference:
    """What was guessed for one file, and what was left unresolved."""

    label: str
    path: Path
    columns: list[str]
    mapping: dict[str, str | None] = field(default_factory=dict)
    elements: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)
    density: str | None = None
    unmapped: list[str] = field(default_factory=list)


def match_column(canonical: str, columns: list[str]) -> str | None:
    """Best column for a canonical field: exact synonym first, then containment."""
    wanted = SYNONYMS.get(canonical, (canonical,))
    normalised = {normalise(c): c for c in columns}
    for synonym in wanted:
        if synonym in normalised:
            return normalised[synonym]
    for synonym in wanted:
        for norm, original in normalised.items():
            if norm.startswith(synonym) or synonym in norm:
                return original
    return None


def inspect_file(label: str, path: Path, wanted: tuple[str, ...]) -> Inference:
    frame = _read_text(path, "auto")
    columns = list(frame.columns)
    inference = Inference(label=label, path=path, columns=columns)
    for canonical in wanted:
        inference.mapping[canonical] = match_column(canonical, columns)
    taken = {v for v in inference.mapping.values() if v}

    for column in columns:
        if column in taken:
            continue
        values = _to_num(frame, column)
        numeric_fraction = float(values.notna().mean())
        if numeric_fraction > 0.8:
            element, units = _element_from_name(column)
            if element:
                inference.elements[element] = f"{column}|{units}"
                taken.add(column)
                continue
            if _looks_like_density(column, values):
                inference.density = inference.density or column
                taken.add(column)
                continue
        else:
            distinct = frame[column].astype(str).str.strip().nunique()
            if 1 < distinct <= MAX_ATTRIBUTE_CARDINALITY:
                inference.attributes[_role_from_name(column)] = column
                taken.add(column)
                continue
        inference.unmapped.append(column)
    return inference


def _element_from_name(column: str) -> tuple[str | None, str | None]:
    match = re.match(r"^([A-Za-z]{1,2})[_\- ]?([A-Za-z]+)$", str(column).strip())
    if not match:
        return None, None
    symbol, suffix = match.group(1), normalise(match.group(2))
    units = ELEMENT_SUFFIXES.get(suffix)
    if units is None:
        return None, None
    return symbol.capitalize(), units


def _looks_like_density(column: str, values: pd.Series) -> bool:
    """Name hint plus a plausible value range: neither alone is enough."""
    if not any(hint in normalise(column) for hint in DENSITY_HINTS):
        return False
    valid = values.dropna()
    if valid.empty:
        return False
    low, high = DENSITY_RANGE_T_M3
    return bool(valid.between(low, high).mean() > 0.8)


def _role_from_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") or "attribute"


def guess_dip_convention(survey_path: Path) -> tuple[str, str]:
    """Guess from the sign of the majority of dips. Always marked for confirmation."""
    frame = _read_text(survey_path, "auto")
    column = match_column("dip", list(frame.columns))
    if column is None:
        return "negative_down", "no dip column found"
    values = _to_num(frame, column).dropna()
    if values.empty:
        return "negative_down", "dip column is empty"
    negative = float((values < 0).mean())
    if negative > 0.6:
        return "negative_down", f"{negative:.0%} of dips are negative"
    return "positive_down", f"{1 - negative:.0%} of dips are positive"


def build_draft(
    paths: dict[str, Path], project_name: str = "New project"
) -> tuple[str, list[Inference], list[str]]:
    """Return the draft YAML text, what was inferred, and notes for the user."""
    wanted = {
        "collar": ("hole_id", "east", "north", "rl", "total_depth"),
        "survey": ("hole_id", "depth", "dip", "azimuth"),
        "assay": ("hole_id", "from_m", "to_m", "sample_id"),
        "litho": ("hole_id", "from_m", "to_m", "logged_code"),
        "block_model": ("x", "y", "z", "dx", "dy", "dz", "period"),
    }
    inferences = [
        inspect_file(label, paths[label], fields)
        for label, fields in wanted.items()
        if label in paths
    ]
    by_label = {i.label: i for i in inferences}
    notes: list[str] = []

    dip_convention, dip_reason = (
        guess_dip_convention(paths["survey"])
        if "survey" in paths
        else ("negative_down", "no survey file")
    )
    notes.append(f"dip_convention guessed as {dip_convention}: {dip_reason}")

    assay = by_label.get("assay")
    elements = dict(assay.elements) if assay else {}
    primary = next(iter(elements), None)
    if primary:
        notes.append(f"primary element guessed as {primary}: first grade column found")
    else:
        notes.append("no grade column found: elements must be filled in by hand")

    block = by_label.get("block_model")
    attributes = dict(block.attributes) if block else {}
    for inference in inferences:
        if inference.label in {"assay", "litho"}:
            for role, column in inference.attributes.items():
                attributes.setdefault(role, None)
                notes.append(f"{inference.label} column {column!r} may be an attribute role")

    text = _render(paths, by_label, attributes, elements, primary, dip_convention, project_name)
    for inference in inferences:
        missing = [k for k, v in inference.mapping.items() if v is None]
        if missing:
            notes.append(f"{inference.label}: could not guess {', '.join(missing)}")
        if inference.unmapped:
            notes.append(
                f"{inference.label}: {len(inference.unmapped)} columns left unmapped "
                f"({', '.join(inference.unmapped[:6])})"
            )
    return text, inferences, notes


def _mapping_lines(inference: Inference | None, indent: str = "      ") -> str:
    if inference is None:
        return f"{indent}# file not supplied\n"
    lines = []
    for canonical, column in inference.mapping.items():
        value = f"{column}{CONFIRM}" if column else f"null{CONFIRM} <- not found, map by hand"
        lines.append(f"{indent}{canonical}: {value}")
    return "\n".join(lines) + "\n"


def _render(
    paths: dict[str, Path],
    by_label: dict[str, Inference],
    attributes: dict[str, str | None],
    elements: dict[str, str],
    primary: str | None,
    dip_convention: str,
    project_name: str,
) -> str:
    def path_of(label: str) -> str:
        return str(paths[label]) if label in paths else "PATH_REQUIRED"

    element_lines = []
    for name, spec in elements.items():
        column, units = spec.split("|")
        flag = "\n    primary: true" + CONFIRM if name == primary else ""
        element_lines.append(
            f"  {name}:{flag}\n"
            f"    drillhole: {{field: {column}, units: {units}}}{CONFIRM}\n"
            f"    block_model: {{field: null, units: null}}"
        )
    if not element_lines:
        element_lines.append(
            "  # REQUIRED: declare at least one element, exactly one marked primary: true\n"
            "  # Zn:\n"
            "  #   primary: true\n"
            "  #   drillhole: {field: Zn_pct, units: pct}\n"
            "  #   block_model: {field: null, units: null}"
        )

    attribute_lines = [
        f"  {role}:\n    block_model: {column or 'null'}{CONFIRM}\n    drillhole: null"
        for role, column in attributes.items()
    ] or ["  # REQUIRED: declare at least one attribute role used by allocation_key"]

    allocation_key = [r for r in list(attributes)[:2]] or ["ROLE_REQUIRED"]
    density = by_label.get("assay").density if "assay" in by_label else None
    block_density = by_label.get("block_model").density if "block_model" in by_label else None

    return f"""# Draft config generated by `geomet-sampler init`.
#
# Every line marked # CONFIRM was guessed by inspecting your files. Check each one
# before running the pipeline: a wrong column mapping fails loudly, but a wrong
# convention fails silently and is the more expensive mistake.
#
# You must still supply by hand:
#   - sources.domain_lookup: logged_code -> geomet_domain, required
#   - sources.availability:  which holes still have core, required
#   - domaining.allocation_key and grade_bins
#   - compositing and mass parameters for your testwork package

project:
  name: {project_name}
  crs: null{CONFIRM}
  output_dir: ./output

conventions:
  length_units: m{CONFIRM}
  dip_convention: {dip_convention}{CONFIRM}
  azimuth_reference: grid{CONFIRM}
  magnetic_declination_deg: 0.0
  density_units: t_m3{CONFIRM}

sources:

  collar:
    path: {path_of("collar")}
    encoding: auto
    columns:
{_mapping_lines(by_label.get("collar"))}
  survey:
    path: {path_of("survey")}
    columns:
{_mapping_lines(by_label.get("survey"))}
  assay:
    path: {path_of("assay")}
    columns:
{_mapping_lines(by_label.get("assay"))}    density:
      source: {density or "null"}{CONFIRM}
      fallback_constant: 2.70{CONFIRM}
    flags:
      low_recovery:
        source: null{CONFIRM} <- map a recovery column if you have one
        type: boolean
        true_values: [Y, "1"]

  litho:
    path: {path_of("litho")}
    columns:
{_mapping_lines(by_label.get("litho"))}
  domain_lookup:
    path: PATH_REQUIRED  # a two-column CSV: logged_code, geomet_domain
    columns:
      logged_code: logged_code
      geomet_domain: geomet_domain

  block_model:
    path: {path_of("block_model")}
    columns:
{_mapping_lines(by_label.get("block_model"))}    density:
      source: {block_density or "null"}{CONFIRM}
      fallback_constant: 2.70{CONFIRM}

  availability:
    path: PATH_REQUIRED  # hole_id, availability, core_diameter_mm, remaining_fraction
    columns:
      hole_id: hole_id
      availability: availability
      core_diameter_mm: core_diameter_mm
      remaining_fraction: remaining_fraction
    value_map:
      AVAILABLE: [AVAILABLE, YES, Y, IN STORAGE]
      UNAVAILABLE: [UNAVAILABLE, NO, N, DISPOSED, CONSUMED]
      PLANNED: [PLANNED, PROPOSED, FUTURE]

  existing_testwork: null
  planned_collar: null
  planned_survey: null

# Role names below are yours to choose. hard_break_on and allocation_key refer to them.
attributes:
{chr(10).join(attribute_lines)}

elements:
{chr(10).join(element_lines)}

desurvey:
  method: minimum_curvature
  fill_collar_survey: true

domaining:
  domain_match_attribute: null
  hard_break_on:
    - geomet_domain
  max_interval_gap_m: 0.10
  break_on_low_recovery: true
  break_on_missing_primary_grade: true

  allocation_key:
{chr(10).join(f"    - {role}" for role in allocation_key)}
    - grade_bin
  grade_bins:
    element: {primary or "ELEMENT_REQUIRED"}{CONFIRM}
    source: drillhole{CONFIRM}
    edges: [0.0, 2.0, 5.0, 999.0]{CONFIRM} <- placeholder, set from your own grade distribution
    labels: [LOW, MED, HIGH]

compositing:
  min_length_m: 6.0{CONFIRM}
  max_length_m: 30.0{CONFIRM}
  target_mass_kg: 50.0{CONFIRM} <- set from your testwork package requirement
  min_mass_kg: 40.0{CONFIRM}
  allow_multi_hole: false
  max_cv_primary: 1.0{CONFIRM}
  require_grade_bin_match: true

mass:
  default_core_diameter_mm: 63.5{CONFIRM} <- HQ 63.5, NQ 47.6, BQ 36.5, PQ 85.0
  default_remaining_fraction: 0.5{CONFIRM} <- 0.5 assumes half core already assayed
  loss_factor: 0.90{CONFIRM} <- PLACEHOLDER, calibrate against a past programme

allocation:
  method: neyman
  total_samples: 40{CONFIRM}
  min_samples_per_domain: 2{CONFIRM}
  min_domain_tonnage_pct: 1.0{CONFIRM}
  risk_multipliers: {{}}
  period_weights:
    1: 3.0{CONFIRM} <- placeholder weighting toward early production
    2: 2.5
    3: 2.0
    default: 1.0

selection:
  method: greedy_with_swap
  max_samples_per_hole: 3{CONFIRM}
  min_separation_m: 50.0{CONFIRM}
  swap_iterations: 200
  random_seed: 42

reporting:
  formats: [xlsx, csv]
  include_gap_register: true
  include_plots: true
"""
