"""Pydantic models for the project YAML.

The config plus the input CSVs is a complete, reproducible record of how a sample list
was produced, so a hash of the raw config text is carried into every output.

All cross-field rules from the specification are enforced here, before any data file is
opened. Checks that need the actual file headers happen in the readers.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import BUILTIN_KEYS


class Strict(BaseModel):
    """Reject unknown keys: a typo in config must not be silently ignored."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ conventions


class LengthUnits(StrEnum):
    M = "m"
    FT = "ft"


class DipConvention(StrEnum):
    NEGATIVE_DOWN = "negative_down"
    POSITIVE_DOWN = "positive_down"


class AzimuthReference(StrEnum):
    GRID = "grid"
    TRUE = "true"


class DensityUnits(StrEnum):
    T_M3 = "t_m3"
    G_CM3 = "g_cm3"
    LB_FT3 = "lb_ft3"


class GradeUnits(StrEnum):
    PCT = "pct"
    PPM = "ppm"
    GPT = "gpt"
    OZ_T = "oz_t"


class Conventions(Strict):
    length_units: LengthUnits
    dip_convention: DipConvention
    azimuth_reference: AzimuthReference = AzimuthReference.GRID
    magnetic_declination_deg: float = 0.0
    density_units: DensityUnits = DensityUnits.T_M3


# ---------------------------------------------------------------------- sources


class DensitySpec(Strict):
    source: str | None = None
    fallback_constant: float | None = None

    @model_validator(mode="after")
    def _need_one(self) -> DensitySpec:
        if self.source is None and self.fallback_constant is None:
            raise ValueError("density: either `source` or `fallback_constant` must be set")
        return self


def _as_text(values: Any) -> Any:
    """Site vocabulary lists are plain text, whatever YAML made of them."""
    if isinstance(values, list):
        return [str(v) for v in values]
    return values


class FlagSpec(Strict):
    source: str | None = None
    type: Literal["boolean", "threshold"] = "boolean"
    true_values: list[str] = Field(default_factory=lambda: ["Y", "1", "TRUE", "true"])

    @field_validator("true_values", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        return _as_text(v)

    threshold: float | None = None
    direction: Literal["below", "above"] = "below"

    @model_validator(mode="after")
    def _threshold_needs_value(self) -> FlagSpec:
        if self.type == "threshold" and self.source is not None and self.threshold is None:
            raise ValueError("flag of type `threshold` requires `threshold`")
        return self


class SourceBase(Strict):
    path: Path
    encoding: str = "auto"
    columns: dict[str, str | None] = Field(default_factory=dict)
    carry_through: list[str] = Field(default_factory=list)

    def mapped(self) -> dict[str, str]:
        """Canonical name -> user column, for the mappings that are actually set."""
        return {k: v for k, v in self.columns.items() if v is not None}

    def user_column(self, canonical: str) -> str | None:
        return self.columns.get(canonical)


class CollarSource(SourceBase):
    pass


class SurveySource(SourceBase):
    pass


class AssaySource(SourceBase):
    density: DensitySpec
    flags: dict[str, FlagSpec] = Field(default_factory=dict)


class LithoSource(SourceBase):
    pass


class DomainLookupSource(SourceBase):
    pass


class BlockModelSource(SourceBase):
    density: DensitySpec


class AvailabilitySource(SourceBase):
    value_map: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("value_map", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _as_text(values) for k, values in v.items()}
        return v


class ExistingTestworkSource(SourceBase):
    pass


class Sources(Strict):
    collar: CollarSource
    survey: SurveySource
    assay: AssaySource
    litho: LithoSource
    domain_lookup: DomainLookupSource
    block_model: BlockModelSource
    availability: AvailabilitySource
    existing_testwork: ExistingTestworkSource | None = None
    planned_collar: CollarSource | None = None
    planned_survey: SurveySource | None = None


# -------------------------------------------------------- user-declared fields


class AttributeSpec(Strict):
    """One user-invented attribute role and where its values come from."""

    block_model: str | None = None
    drillhole: str | None = None

    @model_validator(mode="after")
    def _need_one(self) -> AttributeSpec:
        if self.block_model is None and self.drillhole is None:
            raise ValueError("attribute must declare a `block_model` or `drillhole` source")
        return self


class ElementFieldSpec(Strict):
    field: str | None = None
    units: GradeUnits | None = None

    @model_validator(mode="after")
    def _units_with_field(self) -> ElementFieldSpec:
        if self.field is not None and self.units is None:
            raise ValueError("element field requires explicit `units`")
        return self


class ElementSpec(Strict):
    primary: bool = False
    drillhole: ElementFieldSpec = Field(default_factory=ElementFieldSpec)
    block_model: ElementFieldSpec = Field(default_factory=ElementFieldSpec)


# ------------------------------------------------------------------ processing


class DesurveyConfig(Strict):
    method: Literal["minimum_curvature"] = "minimum_curvature"
    fill_collar_survey: bool = True


class GradeBins(Strict):
    element: str
    source: Literal["block_model", "drillhole"]
    edges: list[float]
    labels: list[str]

    @model_validator(mode="after")
    def _shape(self) -> GradeBins:
        if len(self.edges) != len(self.labels) + 1:
            raise ValueError("grade_bins: len(edges) must equal len(labels) + 1")
        if list(self.edges) != sorted(self.edges):
            raise ValueError("grade_bins: edges must be increasing")
        return self


class Domaining(Strict):
    hard_break_on: list[str]
    #: Block model attribute role whose vocabulary should agree with the logged
    #: `geomet_domain`. Set it to enable the domain_match QC comparison. Left null, the
    #: tool does not guess a correspondence between two vocabularies it cannot verify.
    domain_match_attribute: str | None = None
    max_interval_gap_m: float = 0.10
    break_on_low_recovery: bool = True
    break_on_missing_primary_grade: bool = True
    allocation_key: list[str]
    grade_bins: GradeBins


class Compositing(Strict):
    min_length_m: float
    max_length_m: float
    target_mass_kg: float
    min_mass_kg: float
    allow_multi_hole: bool = False
    max_cv_primary: float = 1.0
    require_grade_bin_match: bool = True

    @model_validator(mode="after")
    def _ranges(self) -> Compositing:
        if self.min_length_m >= self.max_length_m:
            raise ValueError("compositing: min_length_m must be less than max_length_m")
        if self.min_mass_kg > self.target_mass_kg:
            raise ValueError("compositing: min_mass_kg must not exceed target_mass_kg")
        return self


class MassConfig(Strict):
    default_core_diameter_mm: float
    default_remaining_fraction: float = 0.5
    loss_factor: float = 0.90


class AllocationConfig(Strict):
    method: Literal["proportional", "neyman", "manual"] = "neyman"
    total_samples: int
    min_samples_per_domain: int = 0
    min_domain_tonnage_pct: float = 0.0
    risk_multipliers: dict[str, float] = Field(default_factory=dict)
    period_weights: dict[str, float] = Field(default_factory=dict)
    manual_targets: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _stringify_period_keys(cls, data: Any) -> Any:
        """Periods are written as integers in YAML but keyed as strings internally."""
        if isinstance(data, dict) and isinstance(data.get("period_weights"), dict):
            data = dict(data)
            data["period_weights"] = {str(k): float(v) for k, v in data["period_weights"].items()}
        return data

    def period_weight(self, period: Any) -> float:
        if period is None:
            return float(self.period_weights.get("default", 1.0))
        key = str(period)
        if key not in self.period_weights:
            # integral floats ("2.0") should match a "2" key
            with suppress(TypeError, ValueError):
                key = str(int(float(period)))
        return float(self.period_weights.get(key, self.period_weights.get("default", 1.0)))


class SelectionConfig(Strict):
    method: Literal["greedy_with_swap"] = "greedy_with_swap"
    max_samples_per_hole: int = 3
    min_separation_m: float = 0.0
    swap_iterations: int = 200
    random_seed: int = 42
    planned_penalty: float = 0.5
    #: Credit for mass above target is capped here so a very heavy composite cannot
    #: outweigh domain fit.
    mass_margin_cap: float = 1.25
    #: Quality multiplier applied when logged and modelled domains disagree.
    domain_mismatch_factor: float = 0.6
    objective_weights: dict[str, float] = Field(
        default_factory=lambda: {"allocation": 10.0, "separation": 0.01, "cv": 1.0, "match": 5.0}
    )


class ReportingConfig(Strict):
    formats: list[Literal["xlsx", "csv"]] = Field(default_factory=lambda: ["xlsx", "csv"])
    include_gap_register: bool = True
    include_plots: bool = True


class ProjectInfo(Strict):
    name: str
    crs: str | None = None
    output_dir: Path = Path("./output")


class BlockModelOptions(Strict):
    excluded_periods: list[str] = Field(default_factory=list)
    off_model_error_fraction: float = 0.5
    planned_discretisation_m: float = 1.0


class Config(Strict):
    """Root config object. Everything the pipeline needs, fully validated."""

    project: ProjectInfo
    conventions: Conventions
    sources: Sources
    attributes: dict[str, AttributeSpec] = Field(default_factory=dict)
    elements: dict[str, ElementSpec]
    desurvey: DesurveyConfig = Field(default_factory=DesurveyConfig)
    domaining: Domaining
    compositing: Compositing
    mass: MassConfig
    allocation: AllocationConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    block_model_options: BlockModelOptions = Field(default_factory=BlockModelOptions)

    # populated by load_config; not part of the YAML
    config_hash: str = ""
    config_path: Path | None = None

    # -------------------------------------------------------------- accessors

    @property
    def primary_element(self) -> str:
        return next(name for name, spec in self.elements.items() if spec.primary)

    def elements_with_drillhole(self) -> list[str]:
        return [n for n, s in self.elements.items() if s.drillhole.field is not None]

    def elements_with_block_model(self) -> list[str]:
        return [n for n, s in self.elements.items() if s.block_model.field is not None]

    def attributes_with_drillhole(self) -> list[str]:
        return [n for n, s in self.attributes.items() if s.drillhole is not None]

    def attributes_with_block_model(self) -> list[str]:
        return [n for n, s in self.attributes.items() if s.block_model is not None]

    # ------------------------------------------------------- cross-field rules

    @model_validator(mode="after")
    def _validate(self) -> Config:
        problems: list[str] = []

        primaries = [n for n, s in self.elements.items() if s.primary]
        if len(primaries) != 1:
            problems.append(
                f"exactly one element must be flagged primary, found {len(primaries)}: {primaries}"
            )
        else:
            primary = primaries[0]
            if self.elements[primary].drillhole.field is None:
                problems.append(f"primary element {primary!r} needs a drillhole field")

        gb = self.domaining.grade_bins
        if gb.element not in self.elements:
            problems.append(f"grade_bins.element {gb.element!r} is not a declared element")
        else:
            spec = self.elements[gb.element]
            side = spec.block_model if gb.source == "block_model" else spec.drillhole
            if side.field is None:
                problems.append(
                    f"grade_bins.source is {gb.source!r} but element {gb.element!r} "
                    f"has no {gb.source} field"
                )

        role = self.domaining.domain_match_attribute
        if role is not None:
            if role not in self.attributes:
                problems.append(
                    f"domaining.domain_match_attribute: {role!r} is not a declared attribute role"
                )
            elif self.attributes[role].block_model is None:
                problems.append(
                    f"domaining.domain_match_attribute: {role!r} must have a block_model source"
                )

        for key in self.domaining.hard_break_on:
            if key in BUILTIN_KEYS:
                continue
            if key not in self.attributes:
                problems.append(f"hard_break_on: {key!r} is not a declared attribute role")

        for key in self.domaining.allocation_key:
            if key == "geomet_domain":
                problems.append(
                    "allocation_key: `geomet_domain` comes from logging only and cannot be "
                    "aggregated over block model tonnage. Use an attribute role with a "
                    "block_model source instead."
                )
                continue
            if key in BUILTIN_KEYS:
                continue
            spec = self.attributes.get(key)
            if spec is None:
                problems.append(f"allocation_key: {key!r} is not a declared attribute role")
            elif spec.block_model is None:
                problems.append(
                    f"allocation_key: {key!r} must have a block_model source, "
                    "allocation is computed from block model tonnage"
                )

        if problems:
            raise ValueError("; ".join(problems))
        return self


class ConfigLoader(yaml.SafeLoader):
    """YAML loader that does not turn site vocabulary into booleans.

    YAML 1.1 resolves ``Y``, ``YES``, ``NO``, ``ON`` and ``OFF`` to booleans, which
    silently mangles availability vocabularies and flag true-values. This loader uses
    the YAML 1.2 core schema rule instead: only ``true`` and ``false`` are boolean.
    """


ConfigLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def config_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    """Parse and fully validate a project YAML. Paths are resolved against its directory."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = yaml.load(text, Loader=ConfigLoader) or {}
    cfg = Config.model_validate(raw)
    cfg.config_hash = config_hash(text)
    cfg.config_path = path
    _resolve_paths(cfg, path.parent)
    return cfg


def _resolve_paths(cfg: Config, base: Path) -> None:
    """Make every relative path in the config relative to the config file itself."""
    for name in type(cfg.sources).model_fields:
        src = getattr(cfg.sources, name)
        if src is not None and not src.path.is_absolute():
            src.path = (base / src.path).resolve()
    if not cfg.project.output_dir.is_absolute():
        cfg.project.output_dir = (base / cfg.project.output_dir).resolve()
