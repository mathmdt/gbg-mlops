"""Simulated route of the web app: one coloured segment per second, peak on the bump."""

from __future__ import annotations

import numpy as np

from app.route import CALM, MOST_AGITATED, ROUTE, point_along_route, simulated_route


def test_route_ends_are_the_first_and_last_points():
    assert np.allclose(point_along_route(0.0), ROUTE[0])
    assert np.allclose(point_along_route(1.0), ROUTE[-1])


def test_one_segment_per_second_and_peak_on_the_bump():
    rotation = np.ones(900)  # 90 s at 10 Hz
    rotation[600:620] = 20.0  # bump at 60 s
    route = simulated_route(rotation)
    assert len(route["segments"]) == 90
    assert 58 <= route["peak"]["second"] <= 62
    assert route["segments"][60]["colour"] == MOST_AGITATED
    assert route["segments"][10]["colour"] == CALM
