"""Simulated route of a ride on a map of Göteborg, for display only.

The IMU data has no GPS: the path is a fixed, illustrative route in central Göteborg
(Lorensberg → Avenyn → Centralstationen). Only the colours come from the ride: each second
is coloured by how agitated the rotation signal is around it.
"""

from __future__ import annotations

import json

import numpy as np

# (latitude, longitude) of the illustrative route, from start to end.
ROUTE = np.array(
    [
        [57.6969, 11.9797],
        [57.6990, 11.9770],
        [57.7012, 11.9741],
        [57.7031, 11.9715],
        [57.7044, 11.9697],
        [57.7052, 11.9688],
        [57.7066, 11.9679],
        [57.7076, 11.9700],
        [57.7086, 11.9727],
        [57.7092, 11.9745],
    ]
)
SAMPLES_PER_SECOND = 10
CALM, AGITATED, MOST_AGITATED = "#34d399", "#fbbf24", "#f43f5e"


def agitation_per_second(rotation_norm: np.ndarray, smoothing_s: int = 5) -> np.ndarray:
    """Mean rotation norm around each second of the ride (window of `smoothing_s` seconds)."""
    half = smoothing_s * SAMPLES_PER_SECOND // 2
    starts = range(0, len(rotation_norm), SAMPLES_PER_SECOND)
    return np.array([rotation_norm[max(0, i - half) : i + half].mean() for i in starts])


def colours(levels: np.ndarray) -> list[str]:
    """Top 10 % of the seconds in red, the next 25 % in orange, the rest in green.

    Strict comparisons: a perfectly steady ride stays green instead of turning red.
    """
    red, orange = np.quantile(levels, 0.9), np.quantile(levels, 0.65)
    return [MOST_AGITATED if v > red else AGITATED if v > orange else CALM for v in levels]


def point_along_route(fraction: float) -> list[float]:
    """Position at `fraction` (0 = start, 1 = end) of the route length."""
    steps = np.diff(ROUTE, axis=0) * [1.0, np.cos(np.radians(ROUTE[0, 0]))]
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(steps[:, 0], steps[:, 1]))])
    target = fraction * cumulative[-1]
    k = int(np.clip(np.searchsorted(cumulative, target), 1, len(ROUTE) - 1))
    t = (target - cumulative[k - 1]) / (cumulative[k] - cumulative[k - 1])
    return (ROUTE[k - 1] + t * (ROUTE[k] - ROUTE[k - 1])).tolist()


def simulated_route(rotation_norm: np.ndarray) -> dict:
    """Coloured segments (one per second) and the most agitated second of the ride."""
    levels = agitation_per_second(rotation_norm)
    n = len(levels)
    segments = [
        {"path": [point_along_route(i / n), point_along_route((i + 1) / n)], "colour": colour}
        for i, colour in enumerate(colours(levels))
    ]
    peak = int(np.argmax(levels))
    return {
        "segments": segments,
        "peak": {"second": peak, "point": point_along_route((peak + 0.5) / n)},
        "start": ROUTE[0].tolist(),
        "end": ROUTE[-1].tolist(),
    }


def route_map_html(route: dict, height: int = 300) -> str:
    """Leaflet map (OpenStreetMap tiles, darkened) drawing the simulated route."""
    return MAP_TEMPLATE.replace("__ROUTE__", json.dumps(route)).replace("__HEIGHT__", str(height))


MAP_TEMPLATE = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { margin: 0; background: #0b1020; }
  #map { height: __HEIGHT__px; border-radius: 14px; border: 1px solid #1f2a44; }
  .leaflet-container { background: #0b1020; font-family: system-ui, sans-serif; }
  .dark-tiles { filter: invert(100%) hue-rotate(180deg) brightness(85%) contrast(90%)
    saturate(60%); }
  .peak-label { background: #f43f5e; color: #fff; border: 0; font-weight: 600;
    box-shadow: 0 0 12px #f43f5e; }
  .peak-label::before { display: none; }
</style>
<div id="map"></div>
<script>
  const route = __ROUTE__;
  const map = L.map('map', { zoomSnap: 0.25, zoomControl: false, scrollWheelZoom: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 19, className: 'dark-tiles',
  }).addTo(map);
  const bounds = [];
  for (const segment of route.segments) {
    bounds.push(...segment.path);
    L.polyline(segment.path, { color: segment.colour, weight: 6, opacity: 0.95,
      lineCap: 'round' }).addTo(map);
  }
  L.circleMarker(route.start, { radius: 5, color: '#22d3ee', fillColor: '#22d3ee',
    fillOpacity: 1 }).addTo(map).bindTooltip('start', { direction: 'left' });
  L.circleMarker(route.end, { radius: 5, color: '#a855f7', fillColor: '#a855f7',
    fillOpacity: 1 }).addTo(map).bindTooltip('end', { direction: 'right' });
  L.circleMarker(route.peak.point, { radius: 8, color: '#fff', weight: 2,
    fillColor: '#f43f5e', fillOpacity: 1 }).addTo(map)
    .bindTooltip(`peak at ${route.peak.second} s`, { permanent: true, direction: 'right',
      className: 'peak-label', offset: [10, 0] });
  map.fitBounds(bounds, { padding: [24, 24] });
</script>
"""
