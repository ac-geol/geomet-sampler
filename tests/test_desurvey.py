"""Desurvey verified against hand calculations.

Desurvey errors propagate silently into every downstream result, so these are checked
against values computed by hand rather than against the implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomet_sampler import models as M
from geomet_sampler.desurvey import build_traces, desurvey_intervals, interpolate
from geomet_sampler.desurvey.minimum_curvature import _ratio_factor, _step, tangent


def collar_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def survey_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ hand cases


def test_vertical_hole_drops_straight_down():
    collar = collar_frame(
        [{M.HOLE_ID: "V1", M.EAST: 1000.0, M.NORTH: 2000.0, M.RL: 500.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "V1", M.DEPTH: 0.0, M.DIP: -90.0, M.AZIMUTH: 0.0},
            {M.HOLE_ID: "V1", M.DEPTH: 100.0, M.DIP: -90.0, M.AZIMUTH: 0.0},
        ]
    )
    traces, issues = build_traces(collar, survey)
    assert issues == []
    assert traces["V1"].collar == pytest.approx([1000.0, 2000.0, 500.0], abs=1e-9)
    toe = traces["V1"].toe
    assert toe == pytest.approx([1000.0, 2000.0, 400.0], abs=1e-9)

    mid = interpolate(traces["V1"], [50.0])[0]
    assert mid == pytest.approx([1000.0, 2000.0, 450.0], abs=1e-9)


def test_inclined_hole_without_deviation_matches_trigonometry():
    """A straight 45 degree hole toward 090 must be pure trigonometry."""
    collar = collar_frame(
        [{M.HOLE_ID: "I1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "I1", M.DEPTH: 0.0, M.DIP: -45.0, M.AZIMUTH: 90.0},
            {M.HOLE_ID: "I1", M.DEPTH: 100.0, M.DIP: -45.0, M.AZIMUTH: 90.0},
        ]
    )
    traces, _ = build_traces(collar, survey)
    expected_run = 100.0 * np.cos(np.radians(45.0))
    assert traces["I1"].toe == pytest.approx([expected_run, 0.0, -expected_run], abs=1e-9)


def test_azimuth_090_is_east_and_000_is_north():
    """Guards against an east/north swap, which no downstream check would catch."""
    east = tangent(np.array([0.0]), np.array([90.0]))[0]
    north = tangent(np.array([0.0]), np.array([0.0]))[0]
    assert east == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert north == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_published_dogleg_example():
    """Classic minimum curvature worked example.

    Station 1 at MD 3000, inclination 15 degrees, azimuth 020.
    Station 2 at MD 3100, inclination 25 degrees, azimuth 045.
    Hand calculation: dogleg 12.9459 degrees, ratio factor 1.004244,
    dNorth 27.2173, dEast 19.4503, dTVD 94.0091.
    """
    collar = collar_frame(
        [{M.HOLE_ID: "D1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: np.nan}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "D1", M.DEPTH: 3000.0, M.DIP: -75.0, M.AZIMUTH: 20.0},
            {M.HOLE_ID: "D1", M.DEPTH: 3100.0, M.DIP: -65.0, M.AZIMUTH: 45.0},
        ]
    )
    traces, _ = build_traces(collar, survey, fill_collar_survey=False)
    trace = traces["D1"]

    d_east = trace.coords[1, 0] - trace.coords[0, 0]
    d_north = trace.coords[1, 1] - trace.coords[0, 1]
    d_tvd = trace.coords[0, 2] - trace.coords[1, 2]

    assert d_north == pytest.approx(27.2173, abs=0.01)
    assert d_east == pytest.approx(19.4503, abs=0.01)
    assert d_tvd == pytest.approx(94.0091, abs=0.01)


def test_ratio_factor_tends_to_one_for_a_straight_segment():
    assert _ratio_factor(np.array([0.0]))[0] == pytest.approx(1.0)
    assert _ratio_factor(np.array([1e-12]))[0] == pytest.approx(1.0)
    assert _ratio_factor(np.array([np.radians(30.0)]))[0] > 1.0


# --------------------------------------------------------------- interpolation


def test_interpolation_reproduces_station_coordinates():
    collar = collar_frame(
        [{M.HOLE_ID: "D1", M.EAST: 500.0, M.NORTH: 900.0, M.RL: 250.0, M.TOTAL_DEPTH: 300.0}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "D1", M.DEPTH: 0.0, M.DIP: -60.0, M.AZIMUTH: 135.0},
            {M.HOLE_ID: "D1", M.DEPTH: 100.0, M.DIP: -65.0, M.AZIMUTH: 140.0},
            {M.HOLE_ID: "D1", M.DEPTH: 200.0, M.DIP: -75.0, M.AZIMUTH: 150.0},
        ]
    )
    traces, _ = build_traces(collar, survey)
    trace = traces["D1"]
    at_stations = interpolate(trace, trace.depths[:3])
    assert at_stations == pytest.approx(trace.coords[:3], abs=1e-9)


def test_interpolated_arc_length_matches_measured_depth():
    """Spherical tangent interpolation must trace the same arc, so chord sum -> MD."""
    collar = collar_frame(
        [{M.HOLE_ID: "D1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "D1", M.DEPTH: 0.0, M.DIP: -30.0, M.AZIMUTH: 0.0},
            {M.HOLE_ID: "D1", M.DEPTH: 100.0, M.DIP: -80.0, M.AZIMUTH: 90.0},
        ]
    )
    traces, _ = build_traces(collar, survey)
    points = interpolate(traces["D1"], np.linspace(0.0, 100.0, 2001))
    chord_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    assert chord_length == pytest.approx(100.0, rel=1e-6)


def test_last_station_extrapolates_to_total_depth():
    collar = collar_frame(
        [{M.HOLE_ID: "E1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 150.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "E1", M.DEPTH: 0.0, M.DIP: -90.0, M.AZIMUTH: 0.0}])
    traces, _ = build_traces(collar, survey)
    assert traces["E1"].total_depth == pytest.approx(150.0)
    assert traces["E1"].toe[2] == pytest.approx(-150.0)


def test_depth_beyond_last_station_extrapolates_along_final_tangent():
    collar = collar_frame(
        [{M.HOLE_ID: "E1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "E1", M.DEPTH: 0.0, M.DIP: 0.0, M.AZIMUTH: 0.0}])
    traces, _ = build_traces(collar, survey)
    beyond = interpolate(traces["E1"], [130.0])[0]
    assert beyond == pytest.approx([0.0, 130.0, 0.0], abs=1e-9)


def test_hole_without_survey_is_reported_not_dropped_silently():
    collar = collar_frame(
        [{M.HOLE_ID: "N1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 50.0}]
    )
    traces, issues = build_traces(collar, survey_frame([]).reindex(columns=[M.HOLE_ID, M.DEPTH]))
    assert traces == {}
    assert [i.check for i in issues] == ["no_survey"]


def test_step_is_symmetric_in_station_order():
    t1 = tangent(np.array([-70.0]), np.array([10.0]))
    t2 = tangent(np.array([-60.0]), np.array([40.0]))
    forward = _step(t1, t2, np.array([100.0]))
    backward = _step(t2, t1, np.array([100.0]))
    assert forward == pytest.approx(backward, abs=1e-12)


# ------------------------------------------------------- degenerate inputs


def test_hole_whose_stations_are_all_blank_is_reported_not_silently_dropped():
    collar = collar_frame(
        [{M.HOLE_ID: "B1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 50.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "B1", M.DEPTH: np.nan, M.DIP: np.nan, M.AZIMUTH: np.nan}])
    traces, issues = build_traces(collar, survey)
    assert traces == {}
    assert [i.check for i in issues] == ["no_survey"]
    assert "missing depth, dip or azimuth" in issues[0].message


def test_a_collar_station_is_inserted_when_the_first_survey_is_downhole():
    """Without it the trace would start at the shallowest station, not the collar."""
    collar = collar_frame(
        [{M.HOLE_ID: "F1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "F1", M.DEPTH: 20.0, M.DIP: -90.0, M.AZIMUTH: 0.0}])

    filled, _ = build_traces(collar, survey, fill_collar_survey=True)
    assert filled["F1"].depths[0] == pytest.approx(0.0)
    assert filled["F1"].coords[0] == pytest.approx([0.0, 0.0, 0.0])

    unfilled, _ = build_traces(collar, survey, fill_collar_survey=False)
    assert unfilled["F1"].depths[0] == pytest.approx(20.0)


def test_repeated_survey_depths_use_the_first_reading():
    collar = collar_frame(
        [{M.HOLE_ID: "R1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame(
        [
            {M.HOLE_ID: "R1", M.DEPTH: 0.0, M.DIP: -90.0, M.AZIMUTH: 0.0},
            {M.HOLE_ID: "R1", M.DEPTH: 0.0, M.DIP: -45.0, M.AZIMUTH: 180.0},
            {M.HOLE_ID: "R1", M.DEPTH: 100.0, M.DIP: -90.0, M.AZIMUTH: 0.0},
        ]
    )
    traces, _ = build_traces(collar, survey)
    assert traces["R1"].toe == pytest.approx([0.0, 0.0, -100.0], abs=1e-9)


def test_a_non_finite_depth_interpolates_to_nan_rather_than_guessing():
    collar = collar_frame(
        [{M.HOLE_ID: "N1", M.EAST: 0.0, M.NORTH: 0.0, M.RL: 0.0, M.TOTAL_DEPTH: 100.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "N1", M.DEPTH: 0.0, M.DIP: -90.0, M.AZIMUTH: 0.0}])
    traces, _ = build_traces(collar, survey)
    assert np.isnan(interpolate(traces["N1"], [np.nan])[0]).all()


def test_intervals_for_an_undesurveyed_hole_are_reported():
    intervals = pd.DataFrame({M.HOLE_ID: ["GHOST"], M.FROM_M: [0.0], M.TO_M: [1.0]})
    out, issues = desurvey_intervals(intervals, {})
    assert out[M.X_MID].isna().all()
    assert [i.check for i in issues] == ["interval_not_desurveyed"]


def test_desurveyed_intervals_carry_from_to_and_midpoint_coordinates():
    collar = collar_frame(
        [{M.HOLE_ID: "H1", M.EAST: 10.0, M.NORTH: 20.0, M.RL: 100.0, M.TOTAL_DEPTH: 50.0}]
    )
    survey = survey_frame([{M.HOLE_ID: "H1", M.DEPTH: 0.0, M.DIP: -90.0, M.AZIMUTH: 0.0}])
    traces, _ = build_traces(collar, survey)
    intervals = pd.DataFrame({M.HOLE_ID: ["H1"], M.FROM_M: [10.0], M.TO_M: [20.0]})
    out, issues = desurvey_intervals(intervals, traces)
    assert issues == []
    assert out[M.Z_FROM].iloc[0] == pytest.approx(90.0)
    assert out[M.Z_TO].iloc[0] == pytest.approx(80.0)
    assert out[M.Z_MID].iloc[0] == pytest.approx(85.0)
    assert out[M.X_MID].iloc[0] == pytest.approx(10.0)
