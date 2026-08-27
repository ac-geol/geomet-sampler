"""Mass estimation, checked against the reference kg/m table.

These numbers decide whether a recommendation is physically collectable, and they are
what makes compositing unavoidable, so they are asserted against the table in the
specification rather than against the implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.mass.estimate import core_area_m2, estimate_interval_mass, mass_per_metre_kg
from geomet_sampler.models import Availability

# core size, diameter mm, kg/m of half core at SG 2.7, metres needed for 50 kg
REFERENCE = [
    ("BQ", 36.5, 1.41, 35.4),
    ("NQ", 47.6, 2.40, 20.8),
    ("HQ", 63.5, 4.27, 11.7),
    ("PQ", 85.0, 7.66, 6.5),
]


@pytest.mark.parametrize(("name", "diameter", "kg_per_m", "metres_for_50kg"), REFERENCE)
def test_kg_per_metre_matches_the_reference_table(name, diameter, kg_per_m, metres_for_50kg):
    computed = mass_per_metre_kg(diameter, 2.7, remaining_fraction=0.5, loss_factor=1.0)
    assert computed == pytest.approx(kg_per_m, abs=0.01), name
    assert 50.0 / computed == pytest.approx(metres_for_50kg, abs=0.1), name


def test_whole_core_is_twice_half_core():
    half = mass_per_metre_kg(63.5, 2.7, remaining_fraction=0.5)
    whole = mass_per_metre_kg(63.5, 2.7, remaining_fraction=1.0)
    assert whole == pytest.approx(2 * half)


def test_loss_factor_scales_mass_linearly():
    full = mass_per_metre_kg(63.5, 2.7, 0.5, loss_factor=1.0)
    lossy = mass_per_metre_kg(63.5, 2.7, 0.5, loss_factor=0.9)
    assert lossy == pytest.approx(0.9 * full)


def test_core_area_is_the_circle_area_in_square_metres():
    assert core_area_m2(2000.0) == pytest.approx(3.141592653589793, rel=1e-9)


# ------------------------------------------------------------ interval level


def _intervals():
    return pd.DataFrame(
        {
            M.HOLE_ID: ["H1", "H2", "H3"],
            M.LENGTH_M: [10.0, 10.0, 10.0],
            M.DENSITY: [2.7, 2.7, 2.7],
        }
    )


def _availability():
    return pd.DataFrame(
        {
            M.HOLE_ID: ["H1", "H2"],
            M.AVAILABILITY: [Availability.AVAILABLE.value, Availability.AVAILABLE.value],
            M.CORE_DIAMETER_MM: [63.5, 85.0],
            M.REMAINING_FRACTION: [0.5, 1.0],
        }
    )


def test_interval_mass_uses_the_holes_own_core_size_and_remaining_fraction():
    out = estimate_interval_mass(
        _intervals(),
        _availability(),
        default_core_diameter_mm=63.5,
        default_remaining_fraction=0.5,
        loss_factor=1.0,
    )
    assert out[M.MASS_KG].iloc[0] == pytest.approx(42.75, abs=0.05)  # HQ half core, 10 m
    assert out[M.MASS_KG].iloc[1] == pytest.approx(153.2, abs=0.1)  # PQ whole core, 10 m


def test_hole_absent_from_the_availability_file_fails_closed():
    """Absent means unavailable: never recommend core nobody has confirmed exists."""
    out = estimate_interval_mass(
        _intervals(),
        _availability(),
        default_core_diameter_mm=63.5,
        default_remaining_fraction=0.5,
        loss_factor=1.0,
    )
    assert out[M.AVAILABILITY].iloc[2] == Availability.UNAVAILABLE.value


def test_missing_interval_density_falls_back_to_the_block_model_then_the_constant():
    frame = _intervals()
    frame[M.DENSITY] = [2.7, None, None]
    frame["bm_density"] = [None, 3.1, None]
    out = estimate_interval_mass(
        frame,
        _availability(),
        default_core_diameter_mm=63.5,
        default_remaining_fraction=0.5,
        loss_factor=1.0,
        default_density_t_m3=2.5,
    )
    assert list(out[M.DENSITY]) == [2.7, 3.1, 2.5]
