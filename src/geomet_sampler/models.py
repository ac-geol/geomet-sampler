"""Canonical schema, naming helpers and structured records passed between stages.

Every name in this module is fixed in code. User column names are mapped onto these
at load time by :mod:`geomet_sampler.io.readers` and never appear downstream.

Attributes and elements are open namespaces: their members are declared by the user in
config, so they are addressed through the ``*_col`` helpers below rather than being
enumerated anywhere in the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------- canonical fields

HOLE_ID = "hole_id"
EAST = "east"
NORTH = "north"
RL = "rl"
TOTAL_DEPTH = "total_depth"

DEPTH = "depth"
DIP = "dip"
AZIMUTH = "azimuth"

SAMPLE_ID = "sample_id"
FROM_M = "from_m"
TO_M = "to_m"
LENGTH_M = "length_m"
DENSITY = "density"
LOW_RECOVERY = "low_recovery"

LOGGED_CODE = "logged_code"
GEOMET_DOMAIN = "geomet_domain"

BLOCK_X = "x"
BLOCK_Y = "y"
BLOCK_Z = "z"
BLOCK_DX = "dx"
BLOCK_DY = "dy"
BLOCK_DZ = "dz"
PERIOD = "period"

# derived geometry
X_FROM, Y_FROM, Z_FROM = "x_from", "y_from", "z_from"
X_TO, Y_TO, Z_TO = "x_to", "y_to", "z_to"
X_MID, Y_MID, Z_MID = "x_mid", "y_mid", "z_mid"

# derived pipeline fields
INTERVAL_ID = "interval_id"
INTERVAL_IDS_COL = "interval_ids"
PARENT_SAMPLE_ID = "parent_sample_id"
RUN_ID = "run_id"
OUTSIDE_MODEL = "outside_model"
DOMAIN_MATCH = "domain_match"
GRADE_BIN = "grade_bin"
MASS_KG = "mass_kg"
CORE_DIAMETER_MM = "core_diameter_mm"
REMAINING_FRACTION = "remaining_fraction"
AVAILABILITY = "availability"
IS_PLANNED = "is_planned"
ALLOCATION_DOMAIN = "allocation_domain"
BLOCK_TONNES = "tonnes"

COLLAR_REQUIRED = (HOLE_ID, EAST, NORTH, RL)
SURVEY_REQUIRED = (HOLE_ID, DEPTH, DIP, AZIMUTH)
INTERVAL_REQUIRED = (HOLE_ID, FROM_M, TO_M)
LITHO_REQUIRED = (HOLE_ID, FROM_M, TO_M, LOGGED_CODE)
BLOCK_REQUIRED = (BLOCK_X, BLOCK_Y, BLOCK_Z, BLOCK_DX, BLOCK_DY, BLOCK_DZ, PERIOD)

#: Domaining keys that are not user-declared attribute roles.
BUILTIN_KEYS = (GEOMET_DOMAIN, GRADE_BIN, PERIOD)


# ------------------------------------------------------- open-namespace helpers


def attr_col(role: str) -> str:
    """Resolved value of a user-declared attribute role."""
    return f"attr_{role}"


def dh_attr_col(role: str) -> str:
    """Attribute role as logged in the drillhole data."""
    return f"dh_attr_{role}"


def bm_attr_col(role: str) -> str:
    """Attribute role as carried on the block model."""
    return f"bm_attr_{role}"


def elem_col(name: str) -> str:
    """Drillhole grade for a user-declared element, in normalised units."""
    return f"elem_{name}"


def bm_elem_col(name: str) -> str:
    """Block model grade for a user-declared element, in normalised units."""
    return f"bm_elem_{name}"


def wtd_elem_col(name: str) -> str:
    """Length-weighted mean grade of an element over a composite."""
    return f"wtd_{name}"


def domain_key_col(key: str) -> str:
    """Resolve a domaining key name (built-in or attribute role) to its column."""
    return key if key in BUILTIN_KEYS else attr_col(key)


# ------------------------------------------------------------------- records


class Severity(StrEnum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class Issue:
    """One validation finding. Never raised on its own; collected into a report."""

    severity: Severity
    check: str
    message: str
    source: str = ""
    hole_id: str | None = None
    count: int = 1
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "check": self.check,
            "source": self.source,
            "hole_id": self.hole_id,
            "count": self.count,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def extend(self, issues: list[Issue]) -> None:
        self.issues.extend(issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARN]

    def has_errors(self) -> bool:
        return bool(self.errors)


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    PLANNED = "PLANNED"


@dataclass
class GapEntry:
    """A reason some material could not supply a candidate composite."""

    scope: str  # "run" | "domain"
    key: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = {"scope": self.scope, "key": self.key, "reason": self.reason}
        row.update(self.detail)
        return row
