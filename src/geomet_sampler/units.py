"""Unit and convention normalisation. Applied once, at load, by the readers.

Conventions fail silently, which makes them a larger risk than column names. Every
conversion the tool performs lives here so there is one place to audit, and no module
downstream of the reader does any convention handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AzimuthReference, Conventions, DensityUnits, GradeUnits, LengthUnits

FEET_TO_METRES = 0.3048
#: Plausible dry bulk density range in t/m3 for rock. Used as a sanity check, not a filter.
DENSITY_RANGE_T_M3 = (1.5, 6.0)

_DENSITY_TO_T_M3 = {
    DensityUnits.T_M3: 1.0,
    DensityUnits.G_CM3: 1.0,  # 1 g/cm3 == 1 t/m3
    DensityUnits.LB_FT3: 0.0160184634,
}

#: Grades are normalised to ppm internally; ppm and g/t are equivalent for rock.
_GRADE_TO_PPM = {
    GradeUnits.PCT: 10_000.0,
    GradeUnits.PPM: 1.0,
    GradeUnits.GPT: 1.0,
    GradeUnits.OZ_T: 34.2857142857,  # troy oz/short ton -> g/t
}


def length_to_metres(values: pd.Series, units: LengthUnits) -> pd.Series:
    if units is LengthUnits.FT:
        return values * FEET_TO_METRES
    return values


def density_to_t_m3(values: pd.Series, units: DensityUnits) -> pd.Series:
    return values * _DENSITY_TO_T_M3[units]


def grade_to_ppm(values: pd.Series, units: GradeUnits) -> pd.Series:
    return values * _GRADE_TO_PPM[units]


def ppm_to_units(values: pd.Series | float, units: GradeUnits) -> pd.Series | float:
    """Inverse of :func:`grade_to_ppm`, for reporting grades back in declared units."""
    return values / _GRADE_TO_PPM[units]


def dip_to_negative_down(values: pd.Series, conventions: Conventions) -> pd.Series:
    """Normalise dips so that a downward-pointing hole has a negative dip."""
    from .config import DipConvention

    if conventions.dip_convention is DipConvention.POSITIVE_DOWN:
        return -values
    return values


def azimuth_to_grid(values: pd.Series, conventions: Conventions) -> pd.Series:
    """Apply magnetic declination if azimuths are quoted against true north."""
    if conventions.azimuth_reference is AzimuthReference.TRUE:
        values = values + conventions.magnetic_declination_deg
    return np.mod(values, 360.0)
