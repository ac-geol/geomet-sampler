"""Estimate the mass of core an interval can actually supply.

A recommendation the user cannot physically collect is worse than no recommendation, so
mass is computed for every interval before any candidate is considered valid.

    area_m2 = pi * (core_diameter_mm / 2000) ** 2
    mass_kg = length_m * area_m2 * remaining_fraction * density_t_m3 * 1000 * loss_factor
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import models as M


def core_area_m2(core_diameter_mm: float | pd.Series) -> float | pd.Series:
    """Cross-sectional area of whole core, in square metres."""
    return math.pi * (np.asarray(core_diameter_mm, dtype=float) / 2000.0) ** 2


def mass_per_metre_kg(
    core_diameter_mm: float | pd.Series,
    density_t_m3: float | pd.Series,
    remaining_fraction: float | pd.Series = 1.0,
    loss_factor: float = 1.0,
) -> float | pd.Series:
    """Kilograms of recoverable core per metre. The number that forces compositing."""
    return (
        core_area_m2(core_diameter_mm)
        * np.asarray(density_t_m3, dtype=float)
        * np.asarray(remaining_fraction, dtype=float)
        * 1000.0
        * loss_factor
    )


def estimate_interval_mass(
    intervals: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    default_core_diameter_mm: float,
    default_remaining_fraction: float,
    loss_factor: float,
    default_density_t_m3: float | None = None,
) -> pd.DataFrame:
    """Attach core diameter, remaining fraction, availability and mass to each interval.

    Holes absent from the availability file default to UNAVAILABLE: the tool fails
    closed rather than recommending core nobody has confirmed still exists.
    """
    out = intervals.copy()
    lookup = availability.set_index(M.HOLE_ID)

    for column, default in (
        (M.CORE_DIAMETER_MM, default_core_diameter_mm),
        (M.REMAINING_FRACTION, default_remaining_fraction),
    ):
        mapped = out[M.HOLE_ID].map(lookup[column]) if column in lookup.columns else np.nan
        out[column] = pd.to_numeric(mapped, errors="coerce").fillna(default)

    state = out[M.HOLE_ID].map(lookup[M.AVAILABILITY]) if M.AVAILABILITY in lookup.columns else None
    out[M.AVAILABILITY] = (
        pd.Series(M.Availability.UNAVAILABLE.value, index=out.index)
        if state is None
        else state.fillna(M.Availability.UNAVAILABLE.value)
    )

    density = pd.to_numeric(out.get(M.DENSITY), errors="coerce")
    if "bm_density" in out.columns:
        density = density.fillna(pd.to_numeric(out["bm_density"], errors="coerce"))
    if default_density_t_m3 is not None:
        density = density.fillna(default_density_t_m3)
    out[M.DENSITY] = density

    out[M.MASS_KG] = out[M.LENGTH_M] * mass_per_metre_kg(
        out[M.CORE_DIAMETER_MM], out[M.DENSITY], out[M.REMAINING_FRACTION], loss_factor
    )
    return out
