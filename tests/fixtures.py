"""Synthetic fixture project, written as code rather than committed CSVs.

Generating the files from a seeded function keeps the fixture and the schema in one
place, and reproduces the awkward parts of a real export deliberately: a UTF-8 BOM, a
fully quoted file, CRLF endings and inconsistent header casing.

The same template also produces a disguised copy of the project, used by the smoke test
in ``test_end_to_end``: every column renamed, lengths in feet, dips positive-down. Both
variants come from one template so the disguise cannot silently drift out of step with
the original.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

SEED = 7
HOLE_COUNT = 8
HOLE_LENGTH = 60.0
SAMPLE_LENGTH = 1.0
FEET_PER_METRE = 1.0 / 0.3048

DOMAIN_BY_CODE = {"VOLC": "VOLCANIC", "TUFF": "VOLCANIC", "SDST": "SEDIMENT"}

#: Which column each file holds, by internal token. Nothing outside this module knows
#: the tokens; they exist only so the original and disguised configs stay in step.
FIELDS: dict[str, list[str]] = {
    "collar.csv": ["c_hole", "c_east", "c_north", "c_rl", "c_td"],
    "survey.csv": ["s_hole", "s_depth", "s_dip", "s_azi"],
    "assay.csv": ["a_hole", "a_from", "a_to", "a_zn", "a_bd", "a_rec"],
    "litho.csv": ["l_hole", "l_from", "l_to", "l_code"],
    "domain_lookup.csv": ["d_code", "d_domain"],
    "availability.csv": ["v_hole", "v_avail", "v_dia", "v_frac"],
    "block_model.csv": [
        "b_x",
        "b_y",
        "b_z",
        "b_dx",
        "b_dy",
        "b_dz",
        "b_rock",
        "b_weath",
        "b_zn",
        "b_dens",
        "b_period",
    ],
}

#: The column names as a real export might spell them: inconsistent casing, unit
#: suffixes, an awkward character or two.
ORIGINAL = {
    "c_hole": "HoleID",
    "c_east": "Easting",
    "c_north": "Northing",
    "c_rl": "Elevation",
    "c_td": "Length_m",
    "s_hole": "HoleID",
    "s_depth": "Depth_m",
    "s_dip": "Dip",
    "s_azi": "Azimuth",
    "a_hole": "HoleID",
    "a_from": "From_m",
    "a_to": "To_m",
    "a_zn": "Zn_pct",
    "a_bd": "BD_tonnes_m3",
    "a_rec": "LowRec",
    "l_hole": "holeid",
    "l_from": "from",
    "l_to": "to",
    "l_code": "Code",
    "d_code": "logged_code",
    "d_domain": "geomet_domain",
    "v_hole": "hole_id",
    "v_avail": "availability",
    "v_dia": "core_diameter_mm",
    "v_frac": "remaining_fraction",
    "b_x": "X",
    "b_y": "Y",
    "b_z": "Z",
    "b_dx": "DX",
    "b_dy": "DY",
    "b_dz": "DZ",
    "b_rock": "ROCKTYPE",
    "b_weath": "WEATHERING",
    "b_zn": "ZN_PCT",
    "b_dens": "DENSITY",
    "b_period": "PERIOD",
}

#: Arbitrary names carrying no hint of meaning. If any of these produces a different
#: answer from the originals, the library is reading names it should never see.
DISGUISED = {
    "c_hole": "BH",
    "c_east": "C1",
    "c_north": "C2",
    "c_rl": "C3",
    "c_td": "C4",
    "s_hole": "Q1",
    "s_depth": "Q2",
    "s_dip": "Q3",
    "s_azi": "Q4",
    "a_hole": "A1",
    "a_from": "A2",
    "a_to": "A3",
    "a_zn": "A4",
    "a_bd": "A5",
    "a_rec": "A6",
    "l_hole": "L1",
    "l_from": "L2",
    "l_to": "L3",
    "l_code": "L4",
    "d_code": "D1",
    "d_domain": "D2",
    "v_hole": "V1",
    "v_avail": "V2",
    "v_dia": "V3",
    "v_frac": "V4",
    "b_x": "B1",
    "b_y": "B2",
    "b_z": "B3",
    "b_dx": "B4",
    "b_dy": "B5",
    "b_dz": "B6",
    "b_rock": "B7",
    "b_weath": "B8",
    "b_zn": "B9",
    "b_dens": "B10",
    "b_period": "B11",
}

#: Tokens holding a length, which the disguised copy quotes in feet.
LENGTH_TOKENS = {
    "c_east",
    "c_north",
    "c_rl",
    "c_td",
    "s_depth",
    "a_from",
    "a_to",
    "l_from",
    "l_to",
    "b_x",
    "b_y",
    "b_z",
    "b_dx",
    "b_dy",
    "b_dz",
}


def header(filename: str, names: dict[str, str]) -> list[str]:
    return [names[token] for token in FIELDS[filename]]


def _write(path: Path, header_row: list[str], rows: list[list], *, quote_all=False, bom=False):
    buffer = io.StringIO()
    writer = csv.writer(
        buffer, lineterminator="\r\n", quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    )
    writer.writerow(header_row)
    writer.writerows(rows)
    path.write_bytes((("﻿" if bom else "") + buffer.getvalue()).encode("utf-8"))


def write_fixture_project(root: Path, *, total_samples: int = 6) -> Path:
    """Write a complete tiny project and return the path to its config."""
    rng = np.random.default_rng(SEED)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    collars, surveys, lithos, assays, availability = [], [], [], [], []
    for i in range(HOLE_COUNT):
        hole = f"DDH{i + 1:03d}"
        east = 1000.0 + 40.0 * i
        north = 2000.0 + 25.0 * (i % 3)
        collars.append([hole, f"{east:.2f}", f"{north:.2f}", "500.00", f"{HOLE_LENGTH:.1f}"])
        surveys.append([hole, "0.00", "-90.00", "0.00"])
        surveys.append([hole, f"{HOLE_LENGTH:.2f}", "-90.00", "0.00"])

        # two logged units per hole, contact halfway down, so every hole has a hard break
        contact = 30.0
        lithos.append([hole, "0.00", f"{contact:.2f}", "VOLC" if i % 2 == 0 else "TUFF"])
        lithos.append([hole, f"{contact:.2f}", f"{HOLE_LENGTH:.2f}", "SDST"])

        depth = 0.0
        while depth < HOLE_LENGTH:
            end = min(depth + SAMPLE_LENGTH, HOLE_LENGTH)
            # high grade in the upper unit, low in the lower, so the bins are separable
            zn = (6.0 if depth < contact else 0.8) + float(rng.normal(0, 0.15))
            assays.append(
                [hole, f"{depth:.2f}", f"{end:.2f}", f"{max(zn, 0.01):.3f}", "2.700", "N"]
            )
            depth = end

        # one hole has no core left, so the supply constraint is exercised
        availability.append([hole, "IN STORAGE" if i < 7 else "DISPOSED", "63.5", "0.5"])

    _write(data / "collar.csv", header("collar.csv", ORIGINAL), collars)
    _write(data / "survey.csv", header("survey.csv", ORIGINAL), surveys, quote_all=True)
    _write(data / "litho.csv", header("litho.csv", ORIGINAL), lithos)
    _write(data / "assay.csv", header("assay.csv", ORIGINAL), assays, bom=True)
    _write(
        data / "domain_lookup.csv",
        header("domain_lookup.csv", ORIGINAL),
        [[code, domain] for code, domain in sorted(DOMAIN_BY_CODE.items())],
    )
    _write(data / "availability.csv", header("availability.csv", ORIGINAL), availability)

    blocks = []
    for bx in np.arange(960.0, 1320.0, 40.0):
        for by in np.arange(1980.0, 2080.0, 40.0):
            for bz in np.arange(440.0, 500.0, 10.0):
                depth = 500.0 - float(bz)
                upper = depth < 30.0
                blocks.append(
                    [
                        f"{bx:.1f}",
                        f"{by:.1f}",
                        f"{bz:.1f}",
                        "40.0",
                        "40.0",
                        "10.0",
                        "VOLCANIC" if upper else "SEDIMENT",
                        "OXIDE" if depth < 25.0 else "FRESH",
                        f"{6.0 if upper else 0.8:.3f}",
                        "2.700",
                        "1" if upper else "2",
                    ]
                )
    _write(data / "block_model.csv", header("block_model.csv", ORIGINAL), blocks)

    config = root / "project.yaml"
    config.write_text(config_text(total_samples, ORIGINAL), encoding="utf-8")
    return config


def write_disguised_project(root: Path, source: Path, *, total_samples: int = 6) -> Path:
    """Rewrite the fixture with renamed columns, feet, and positive-down dips.

    The tool must produce the same answer from this as from the original. If it does
    not, a column name or a convention has leaked out of the readers and into the
    library.
    """
    import pandas as pd

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    for filename, tokens in FIELDS.items():
        frame = pd.read_csv(source / "data" / filename, dtype=str, encoding="utf-8-sig")
        frame.columns = [c.strip() for c in frame.columns]
        by_token = dict(zip(tokens, frame.columns, strict=True))

        for token in tokens:
            column = by_token[token]
            if token in LENGTH_TOKENS:
                frame[column] = (pd.to_numeric(frame[column]) * FEET_PER_METRE).map(
                    lambda v: f"{v:.10f}"
                )
            elif token == "s_dip":  # the file now quotes dips positive-down
                frame[column] = (-pd.to_numeric(frame[column])).map(lambda v: f"{v:.6f}")

        renames = {by_token[token]: DISGUISED[token] for token in tokens}
        frame.rename(columns=renames).to_csv(data / filename, index=False)

    config = root / "project.yaml"
    config.write_text(
        config_text(total_samples, DISGUISED, length_units="ft", dip_convention="positive_down"),
        encoding="utf-8",
    )
    return config


def config_text(
    total_samples: int,
    n: dict[str, str],
    *,
    length_units: str = "m",
    dip_convention: str = "negative_down",
) -> str:
    """One config template, rendered with whichever set of column names is in play."""
    return f"""
project:
  name: Fixture project
  output_dir: ./output

conventions:
  length_units: {length_units}
  dip_convention: {dip_convention}
  azimuth_reference: grid
  density_units: t_m3

sources:
  collar:
    path: data/collar.csv
    columns: {{hole_id: {n["c_hole"]}, east: {n["c_east"]}, north: {n["c_north"]},
               rl: {n["c_rl"]}, total_depth: {n["c_td"]}}}
  survey:
    path: data/survey.csv
    columns: {{hole_id: {n["s_hole"]}, depth: {n["s_depth"]}, dip: {n["s_dip"]},
               azimuth: {n["s_azi"]}}}
  assay:
    path: data/assay.csv
    columns: {{hole_id: {n["a_hole"]}, from_m: {n["a_from"]}, to_m: {n["a_to"]},
               sample_id: null}}
    density: {{source: {n["a_bd"]}, fallback_constant: 2.70}}
    flags:
      low_recovery: {{source: {n["a_rec"]}, type: boolean, true_values: [Y]}}
  litho:
    path: data/litho.csv
    columns: {{hole_id: {n["l_hole"]}, from_m: {n["l_from"]}, to_m: {n["l_to"]},
               logged_code: {n["l_code"]}}}
  domain_lookup:
    path: data/domain_lookup.csv
    columns: {{logged_code: {n["d_code"]}, geomet_domain: {n["d_domain"]}}}
  block_model:
    path: data/block_model.csv
    columns: {{x: {n["b_x"]}, y: {n["b_y"]}, z: {n["b_z"]}, dx: {n["b_dx"]},
               dy: {n["b_dy"]}, dz: {n["b_dz"]}, period: {n["b_period"]}}}
    density: {{source: {n["b_dens"]}, fallback_constant: 2.70}}
  availability:
    path: data/availability.csv
    columns: {{hole_id: {n["v_hole"]}, availability: {n["v_avail"]},
               core_diameter_mm: {n["v_dia"]}, remaining_fraction: {n["v_frac"]}}}
    value_map:
      AVAILABLE: [IN STORAGE]
      UNAVAILABLE: [DISPOSED]
      PLANNED: [PLANNED]
  existing_testwork: null

attributes:
  rock_type: {{block_model: {n["b_rock"]}, drillhole: null}}
  weathering: {{block_model: {n["b_weath"]}, drillhole: null}}

elements:
  Zn:
    primary: true
    drillhole: {{field: {n["a_zn"]}, units: pct}}
    block_model: {{field: {n["b_zn"]}, units: pct}}

domaining:
  domain_match_attribute: rock_type
  hard_break_on: [geomet_domain, weathering]
  max_interval_gap_m: 0.10
  break_on_low_recovery: true
  break_on_missing_primary_grade: true
  allocation_key: [rock_type, weathering, grade_bin]
  grade_bins:
    element: Zn
    source: drillhole
    edges: [0.0, 2.0, 999.0]
    labels: [LOW, HIGH]

compositing:
  min_length_m: 6.0
  max_length_m: 20.0
  target_mass_kg: 50.0
  min_mass_kg: 40.0
  max_cv_primary: 1.0
  require_grade_bin_match: true

mass:
  default_core_diameter_mm: 63.5
  default_remaining_fraction: 0.5
  loss_factor: 0.90

allocation:
  method: proportional
  total_samples: {total_samples}
  min_samples_per_domain: 1
  min_domain_tonnage_pct: 1.0
  period_weights: {{1: 2.0, default: 1.0}}

selection:
  method: greedy_with_swap
  max_samples_per_hole: 2
  min_separation_m: 0.0
  swap_iterations: 50
  random_seed: 42

reporting:
  formats: [csv]
  include_plots: false
"""
