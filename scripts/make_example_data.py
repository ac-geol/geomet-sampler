"""Generate the synthetic example dataset used by config/example_project.yaml.

The real MPA export is not distributed with this repository. These files stand in for
it: same shape, same messiness, different numbers. They deliberately reproduce the
quirks the readers must absorb, so running the example exercises the awkward paths:

- UTF-8 BOM on the assay ``HoleID`` header
- survey file fully quoted, so every field parses as a string
- CRLF line endings throughout
- lithology file with lowercase ``holeid`` / ``from`` / ``to`` headers
- partial coverage: not every collar has assays or lithology
- a block model carrying a PERIOD field, which the real example lacks

Run with:  uv run python scripts/make_example_data.py
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RNG = np.random.default_rng(20240227)

N_HOLES = 60
GRID_ORIGIN = (10_000.0, 20_000.0, 450.0)
LITHO_CODES = {
    "VOLCANIC": ["BZPZ", "BZTF", "VOLC"],
    "SEDIMENT": ["TSBF", "SDST", "MDST"],
    "INTRUSIVE": ["DIOR", "QFP"],
}
ROCK_BY_DOMAIN = {"VOLCANIC": "VOLCANIC", "SEDIMENT": "SEDIMENT", "INTRUSIVE": "INTRUSIVE"}


# ------------------------------------------------------------------- geology


def rock_type_at(x: float, y: float, z: float) -> str:
    """A simple layered and intruded geology, shared by blocks and drillholes."""
    fold = 30.0 * math.sin((x - GRID_ORIGIN[0]) / 120.0)
    horizon = GRID_ORIGIN[2] - 120.0 + fold
    if (x - 10_260.0) ** 2 / 90.0**2 + (y - 20_240.0) ** 2 / 70.0**2 < 1.0 and z < horizon + 60:
        return "INTRUSIVE"
    return "VOLCANIC" if z > horizon else "SEDIMENT"


def weathering_at(z: float, surface_rl: float) -> str:
    depth = surface_rl - z
    if depth < 35.0:
        return "OXIDE"
    if depth < 70.0:
        return "TRANSITION"
    return "FRESH"


def zn_pct_at(x: float, y: float, z: float) -> float:
    """A single steeply dipping lens, so grade bins are not uniformly distributed."""
    across = (x - 10_250.0) * 0.94 + (z - 330.0) * 0.34
    along = y - 20_250.0
    intensity = math.exp(-((across / 55.0) ** 2) - ((along / 160.0) ** 2))
    return max(0.02, 9.0 * intensity + RNG.normal(0.0, 0.35))


def surface_rl(x: float, y: float) -> float:
    return GRID_ORIGIN[2] + 12.0 * math.sin(x / 300.0) + 8.0 * math.cos(y / 260.0)


# --------------------------------------------------------------- file writing


def write_csv(path: Path, header: list[str], rows: list[list], *, quote_all=False, bom=False):
    """Write CRLF-terminated CSV, optionally fully quoted and with a UTF-8 BOM."""
    buffer = io.StringIO()
    writer = csv.writer(
        buffer, lineterminator="\r\n", quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
    )
    writer.writerow(header)
    writer.writerows(rows)
    text = buffer.getvalue()
    path.write_bytes((("﻿" if bom else "") + text).encode("utf-8"))
    print(f"  {path.relative_to(ROOT)}  ({len(rows)} rows)")


# ------------------------------------------------------------------ generation


def make_collars():
    collars = []
    for i in range(N_HOLES):
        row, col = divmod(i, 10)
        east = 10_120.0 + col * 40.0 + RNG.normal(0, 6)
        north = 20_120.0 + row * 55.0 + RNG.normal(0, 6)
        rl = surface_rl(east, north)
        dip = float(RNG.choice([-90.0, -75.0, -65.0, -60.0]))
        azimuth = float(RNG.choice([90.0, 270.0, 0.0])) if dip > -90 else 0.0
        length = float(RNG.integers(120, 320))
        collars.append(
            {
                "HoleID": f"BX{20 + i // 20}-{i + 1:03d}",
                "Easting": east,
                "Northing": north,
                "Elevation": rl,
                "Length_m": length,
                "Dip": dip,
                "Azimuth": azimuth,
                "HoleType": "DD",
                "Prospect": "MAIN" if col < 6 else "NORTH",
                "Year": 2020 + i % 5,
            }
        )
    return collars


def trace_point(collar, depth: float) -> tuple[float, float, float]:
    """Straight-line trace, adequate for placing synthetic geology."""
    inclination = math.radians(90.0 + collar["Dip"])
    bearing = math.radians(collar["Azimuth"])
    return (
        collar["Easting"] + depth * math.sin(inclination) * math.sin(bearing),
        collar["Northing"] + depth * math.sin(inclination) * math.cos(bearing),
        collar["Elevation"] - depth * math.cos(inclination),
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    collars = make_collars()
    print("writing example data:")

    write_csv(
        DATA / "MPA_Collar_20240227.csv",
        ["HoleID", "Easting", "Northing", "Elevation", "Length_m", "HoleType", "Prospect", "Year"],
        [
            [
                c["HoleID"],
                f"{c['Easting']:.2f}",
                f"{c['Northing']:.2f}",
                f"{c['Elevation']:.2f}",
                f"{c['Length_m']:.1f}",
                c["HoleType"],
                c["Prospect"],
                c["Year"],
            ]
            for c in collars
        ],
    )

    survey_rows = []
    for c in collars:
        depths = [0.0]
        while depths[-1] + 50.0 < c["Length_m"]:
            depths.append(depths[-1] + 50.0)
        depths.append(c["Length_m"])
        for k, depth in enumerate(depths):
            # gentle downhole flattening; never past vertical, which would
            # trip the dip range check
            survey_rows.append(
                [
                    c["HoleID"],
                    f"{depth:.2f}",
                    f"{min(c['Dip'] + 0.6 * k, -45.0):.2f}",
                    f"{(c['Azimuth'] + 0.8 * k) % 360:.2f}",
                ]
            )
    write_csv(
        DATA / "MPA_Survey_20240227.csv",
        ["HoleID", "Depth_m", "Dip", "Azimuth"],
        survey_rows,
        quote_all=True,
    )

    # partial coverage: some holes were never logged, some never assayed
    logged = [c for c in collars if RNG.random() < 0.88]
    assayed = [c for c in collars if RNG.random() < 0.92]

    litho_rows = []
    for c in logged:
        depth = 0.0
        while depth < c["Length_m"]:
            run = float(RNG.uniform(8.0, 35.0))
            end = min(depth + run, c["Length_m"])
            x, y, z = trace_point(c, (depth + end) / 2.0)
            domain = rock_type_at(x, y, z)
            if RNG.random() < 0.12:  # logging and modelling disagree sometimes
                domain = str(RNG.choice([d for d in LITHO_CODES if d != domain]))
            code = str(RNG.choice(LITHO_CODES[domain]))
            litho_rows.append([c["HoleID"], f"{depth:.2f}", f"{end:.2f}", code])
            depth = end
    write_csv(DATA / "MPA_Interp_20240227.csv", ["holeid", "from", "to", "Code"], litho_rows)

    assay_rows = []
    for c in assayed:
        depth = 0.0
        while depth < c["Length_m"]:
            run = float(RNG.choice([1.0, 1.0, 1.5, 2.0]))
            end = min(depth + run, c["Length_m"])
            x, y, z = trace_point(c, (depth + end) / 2.0)
            zn = zn_pct_at(x, y, z)
            assay_rows.append(
                [
                    c["HoleID"],
                    f"{depth:.2f}",
                    f"{end:.2f}",
                    f"{zn:.3f}",
                    f"{max(0.01, zn * 0.35 + RNG.normal(0, 0.1)):.3f}",
                    f"{max(0.5, zn * 6.0 + RNG.normal(0, 2.0)):.1f}",
                    f"{2.62 + 0.06 * min(zn, 8.0) + RNG.normal(0, 0.03):.3f}",
                    "Y" if RNG.random() < 0.03 else "N",
                ]
            )
            depth = end
    write_csv(
        DATA / "MPA_Samples_BD_20240227.csv",
        [
            "HoleID",
            "From_m",
            "To_m",
            "Zn_pct",
            "Pb_pct",
            "Ag_ppm",
            "BD_tonnes_m3",
            "LowRecovery_<=85pct",
        ],
        assay_rows,
        bom=True,
    )

    lookup_rows = [[code, domain] for domain, codes in LITHO_CODES.items() for code in codes]
    write_csv(DATA / "domain_lookup.csv", ["logged_code", "geomet_domain"], sorted(lookup_rows))

    availability_rows = []
    for i, c in enumerate(collars):
        if i % 11 == 0:
            availability_rows.append([c["HoleID"], "DISPOSED", "", "", "core disposed"])
        else:
            diameter = float(RNG.choice([63.5, 63.5, 47.6, 85.0]))
            availability_rows.append([c["HoleID"], "IN STORAGE", f"{diameter}", "0.5", ""])
    write_csv(
        DATA / "hole_availability.csv",
        ["hole_id", "availability", "core_diameter_mm", "remaining_fraction", "notes"],
        availability_rows,
    )

    block_rows = []
    dx = dy = 20.0
    dz = 10.0
    for bx in np.arange(10_060.0, 10_540.0, dx):
        for by in np.arange(20_060.0, 20_460.0, dy):
            for bz in np.arange(120.0, 470.0, dz):
                x, y, z = float(bx), float(by), float(bz)
                if z > surface_rl(x, y):
                    continue
                zn = zn_pct_at(x, y, z)
                rock = rock_type_at(x, y, z)
                weathering = weathering_at(z, surface_rl(x, y))
                # only the better material near the lens is scheduled
                period = "" if zn < 1.2 else str(int(np.clip(1 + (10_400 - x) // 90, 1, 6)))
                block_rows.append(
                    [
                        f"{x:.1f}",
                        f"{y:.1f}",
                        f"{z:.1f}",
                        f"{dx:.1f}",
                        f"{dy:.1f}",
                        f"{dz:.1f}",
                        ROCK_BY_DOMAIN[rock],
                        weathering,
                        "MEASURED" if abs(x - 10_250) < 120 else "INDICATED",
                        f"{zn:.3f}",
                        f"{zn * 6.0:.2f}",
                        f"{2.62 + 0.06 * min(zn, 8.0):.3f}",
                        period,
                    ]
                )
    write_csv(
        DATA / "block_model_periods.csv",
        [
            "X",
            "Y",
            "Z",
            "DX",
            "DY",
            "DZ",
            "ROCKTYPE",
            "WEATHERING",
            "CLASS",
            "ZN_PCT",
            "AG_GPT",
            "DENSITY",
            "PERIOD",
        ],
        block_rows,
    )

    testwork_rows = []
    for c in collars[:6]:
        testwork_rows.append(
            [f"TW-{c['HoleID']}", c["HoleID"], "40.00", "58.00", "", "COMMINUTION"]
        )
    write_csv(
        DATA / "existing_testwork.csv",
        ["sample_id", "HoleID", "from_m", "to_m", "geomet_domain", "test_package"],
        testwork_rows,
    )
    print("done")


if __name__ == "__main__":
    main()
