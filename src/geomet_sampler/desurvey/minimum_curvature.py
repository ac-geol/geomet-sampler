"""Minimum curvature desurvey.

Input is assumed already normalised: dips negative-down, lengths in metres, azimuths in
grid degrees. This module does no convention handling; that happens once, in the readers.

Angles follow the specification: inclination from vertical-down ``I = 90 + dip``, so a
vertical hole has ``I = 0`` and a horizontal hole ``I = 90``. The unit tangent is
``(east, north, down) = (sin I sin B, sin I cos B, cos I)`` and RL decreases as the
downward component accumulates.

Interpolation to an arbitrary depth uses spherical interpolation of the tangent, which is
exactly the circular arc that minimum curvature assumes, rather than an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import models as M
from ..models import Issue, Severity

#: Below this dogleg (radians) the arc is treated as straight; avoids 0/0 in the factor.
STRAIGHT_TOL = 1e-9


@dataclass
class HoleTrace:
    """Desurveyed survey stations for one hole, with tangents for interpolation."""

    hole_id: str
    depths: np.ndarray  # (n,) measured depth, ascending
    coords: np.ndarray  # (n, 3) east, north, rl
    tangents: np.ndarray  # (n, 3) unit vectors in (east, north, down)

    @property
    def collar(self) -> np.ndarray:
        return self.coords[0]

    @property
    def toe(self) -> np.ndarray:
        return self.coords[-1]

    @property
    def total_depth(self) -> float:
        return float(self.depths[-1])


def tangent(dip_deg: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    """Unit direction vector (east, north, down) for negative-down dips."""
    inclination = np.radians(90.0 + np.asarray(dip_deg, dtype=float))
    bearing = np.radians(np.asarray(azimuth_deg, dtype=float))
    return np.stack(
        [
            np.sin(inclination) * np.sin(bearing),
            np.sin(inclination) * np.cos(bearing),
            np.cos(inclination),
        ],
        axis=-1,
    )


def _dogleg(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.einsum("...i,...i->...", t1, t2), -1.0, 1.0))


def _ratio_factor(dogleg: np.ndarray) -> np.ndarray:
    """(2/DL) tan(DL/2), which tends to 1 as the dogleg tends to zero."""
    dogleg = np.asarray(dogleg, dtype=float)
    out = np.ones_like(dogleg)
    curved = dogleg > STRAIGHT_TOL
    out[curved] = (2.0 / dogleg[curved]) * np.tan(dogleg[curved] / 2.0)
    return out


def _step(t1: np.ndarray, t2: np.ndarray, md: np.ndarray) -> np.ndarray:
    """Minimum curvature displacement (east, north, down) over course length ``md``."""
    md = np.asarray(md, dtype=float)
    rf = _ratio_factor(_dogleg(t1, t2))
    return (md / 2.0)[..., None] * (t1 + t2) * rf[..., None]


def _slerp(t1: np.ndarray, t2: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Interpolate the tangent along the same great circle the arc follows.

    Minimum curvature assumes the direction turns at a constant rate, which is exactly
    a great-circle path on the direction sphere. Interpolating the tangent this way,
    rather than interpolating dip and azimuth separately, keeps an interpolated point
    on the same arc the two stations define.
    """
    dogleg = float(_dogleg(t1, t2))
    f = np.atleast_1d(np.asarray(f, dtype=float))
    if dogleg <= STRAIGHT_TOL:
        return np.repeat(t1[None, :], f.size, axis=0)
    scale = np.sin(dogleg)
    w1 = np.sin((1.0 - f) * dogleg) / scale
    w2 = np.sin(f * dogleg) / scale
    return w1[:, None] * t1 + w2[:, None] * t2


def _displacement_to_coords(origin: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply an (east, north, down) displacement to an (east, north, rl) point."""
    out = np.asarray(delta, dtype=float).copy()
    out[..., 2] *= -1.0  # down accumulates as falling RL
    return origin + out


def build_traces(
    collar: pd.DataFrame, survey: pd.DataFrame, *, fill_collar_survey: bool = True
) -> tuple[dict[str, HoleTrace], list[Issue]]:
    """Desurvey every collar that has usable survey data.

    A station at depth 0 is inserted from the shallowest surveyed orientation when
    ``fill_collar_survey`` is set, and the deepest station is extrapolated to total depth
    holding orientation constant.
    """
    issues: list[Issue] = []
    traces: dict[str, HoleTrace] = {}
    by_hole = {hid: g for hid, g in survey.groupby(M.HOLE_ID, sort=False)}

    for row in collar.itertuples(index=False):
        hole_id = getattr(row, M.HOLE_ID)
        stations = by_hole.get(hole_id)
        if stations is None or stations.empty:
            issues.append(
                Issue(
                    Severity.WARN,
                    "no_survey",
                    "hole has no survey stations and cannot be desurveyed",
                    source="survey",
                    hole_id=hole_id,
                )
            )
            continue

        stations = stations.dropna(subset=[M.DEPTH, M.DIP, M.AZIMUTH]).sort_values(M.DEPTH)
        stations = stations.drop_duplicates(M.DEPTH, keep="first")
        if stations.empty:
            issues.append(
                Issue(
                    Severity.WARN,
                    "no_survey",
                    "all survey stations for this hole have missing depth, dip or azimuth",
                    source="survey",
                    hole_id=hole_id,
                )
            )
            continue

        depths = stations[M.DEPTH].to_numpy(dtype=float)
        dips = stations[M.DIP].to_numpy(dtype=float)
        azimuths = stations[M.AZIMUTH].to_numpy(dtype=float)

        if fill_collar_survey and depths[0] > 0.0:
            depths = np.insert(depths, 0, 0.0)
            dips = np.insert(dips, 0, dips[0])
            azimuths = np.insert(azimuths, 0, azimuths[0])

        total_depth = getattr(row, M.TOTAL_DEPTH, np.nan)
        if pd.notna(total_depth) and total_depth > depths[-1] + 1e-9:
            depths = np.append(depths, float(total_depth))
            dips = np.append(dips, dips[-1])
            azimuths = np.append(azimuths, azimuths[-1])

        tangents = tangent(dips, azimuths)
        coords = np.empty((len(depths), 3), dtype=float)
        coords[0] = [getattr(row, M.EAST), getattr(row, M.NORTH), getattr(row, M.RL)]
        if len(depths) > 1:
            md = np.diff(depths)
            deltas = _step(tangents[:-1], tangents[1:], md)
            deltas[:, 2] *= -1.0
            coords[1:] = coords[0] + np.cumsum(deltas, axis=0)

        traces[hole_id] = HoleTrace(hole_id, depths, coords, tangents)

    return traces, issues


def interpolate(trace: HoleTrace, depths: np.ndarray) -> np.ndarray:
    """Position (east, north, rl) at arbitrary measured depths along a trace."""
    depths = np.atleast_1d(np.asarray(depths, dtype=float))
    out = np.empty((depths.size, 3), dtype=float)

    for i, depth in enumerate(depths):
        if not np.isfinite(depth):
            out[i] = np.nan
            continue
        if depth <= trace.depths[0]:
            delta = trace.tangents[0] * (depth - trace.depths[0])
            out[i] = _displacement_to_coords(trace.coords[0], delta)
            continue
        if depth >= trace.depths[-1]:
            delta = trace.tangents[-1] * (depth - trace.depths[-1])
            out[i] = _displacement_to_coords(trace.coords[-1], delta)
            continue

        j = int(np.searchsorted(trace.depths, depth, side="right") - 1)
        d1, d2 = trace.depths[j], trace.depths[j + 1]
        t1, t2 = trace.tangents[j], trace.tangents[j + 1]
        f = (depth - d1) / (d2 - d1)
        t_f = _slerp(t1, t2, np.array([f]))[0]
        delta = _step(t1, t_f, np.array(depth - d1))
        out[i] = _displacement_to_coords(trace.coords[j], delta)

    return out


def desurvey_intervals(
    intervals: pd.DataFrame, traces: dict[str, HoleTrace]
) -> tuple[pd.DataFrame, list[Issue]]:
    """Attach from, to and midpoint coordinates to an interval table."""
    issues: list[Issue] = []
    coord_cols = [
        (M.X_FROM, M.Y_FROM, M.Z_FROM),
        (M.X_TO, M.Y_TO, M.Z_TO),
        (M.X_MID, M.Y_MID, M.Z_MID),
    ]
    out = intervals.copy()
    for group in coord_cols:
        for col in group:
            out[col] = np.nan

    missing: set[str] = set()
    for hole_id, group in out.groupby(M.HOLE_ID, sort=False):
        trace = traces.get(hole_id)
        if trace is None:
            missing.add(hole_id)
            continue
        from_m = group[M.FROM_M].to_numpy(dtype=float)
        to_m = group[M.TO_M].to_numpy(dtype=float)
        mid = (from_m + to_m) / 2.0
        for depths, cols in zip((from_m, to_m, mid), coord_cols, strict=True):
            xyz = interpolate(trace, depths)
            for k, col in enumerate(cols):
                out.loc[group.index, col] = xyz[:, k]

    if missing:
        issues.append(
            Issue(
                Severity.WARN,
                "interval_not_desurveyed",
                f"{len(missing)} holes carry intervals but have no desurveyed trace; "
                "their intervals cannot be assigned to the block model",
                source="survey",
                count=len(missing),
                detail={"holes": sorted(missing)[:20]},
            )
        )
    return out, issues
