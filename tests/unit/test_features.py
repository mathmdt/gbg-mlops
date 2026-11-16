"""Shared preprocessing: units, cleaning, windows, normalisation, aggregation."""

import numpy as np
import pytest

from gbg_mlops import features as F


def ride(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return np.arange(n) * F.SAMPLE_PERIOD_MS, rng.normal(size=(n, 6))


def test_deg_to_rad_converts_gyroscope_only():
    values = np.array([[1.0, 2.0, 3.0, 180.0, 90.0, -57.2958]])
    out = F.to_rad_s(values, "deg_s")
    np.testing.assert_allclose(out[0, :3], [1, 2, 3])
    np.testing.assert_allclose(out[0, 3:], [np.pi, np.pi / 2, -1.0], rtol=1e-5)
    np.testing.assert_array_equal(values[0, 3], 180.0)  # input untouched
    np.testing.assert_array_equal(F.to_rad_s(values, "rad_s"), values)
    with pytest.raises(ValueError):
        F.to_rad_s(values, "rpm")


def test_clean_ride_sorts_dedups_and_interpolates():
    time_index = np.array([200, 0, 100, 100, 300])
    values = np.zeros((5, 6))
    values[:, 0] = [2.0, 0.0, np.nan, 99.0, 3.0]  # duplicate 100 → first occurrence (NaN)
    t, v = F.clean_ride(time_index, values)
    np.testing.assert_array_equal(t, [0, 100, 200, 300])
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 2.0, 3.0])


def test_clean_ride_fills_edges_and_rejects_bad_input():
    values = np.ones((4, 6))
    values[0, 1] = values[-1, 1] = np.nan
    _, v = F.clean_ride(np.arange(4) * 100, values)
    assert not np.isnan(v).any()
    with pytest.raises(ValueError, match="increase by"):
        F.clean_ride(np.array([0, 100, 300]), np.ones((3, 6)))
    with pytest.raises(ValueError, match="no valid value"):
        bad = np.ones((3, 6))
        bad[:, 2] = np.nan
        F.clean_ride(np.arange(3) * 100, bad)
    with pytest.raises(ValueError, match="infinite"):
        F.clean_ride(np.arange(3) * 100, np.full((3, 6), np.inf))


@pytest.mark.parametrize(
    "n, expected",
    [(300, [0]), (450, [0, 150]), (500, [0, 150, 200]), (600, [0, 150, 300])],
)
def test_window_starts_cover_the_whole_ride(n, expected):
    assert F.window_starts(n).tolist() == expected


def test_window_starts_rejects_short_rides():
    with pytest.raises(ValueError, match="at least 300"):
        F.window_starts(299)


def test_make_windows_shape_order_and_dtype():
    _, values = ride(500)
    windows = F.make_windows(values)
    assert windows.shape == (3, 6, 300) and windows.dtype == np.float32
    np.testing.assert_allclose(windows[1, 3, :], values[150:450, 3], rtol=1e-6)


def test_window_labels_flag_windows_containing_a_topple():
    topple = np.zeros(600)
    topple[320] = 1
    assert F.window_labels(topple).tolist() == [0, 1, 1]


def test_normalization_fit_apply_and_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    windows = (rng.normal(size=(20, 6, 300)) * 3 + 5).astype(np.float32)
    windows[:, 5, :] = 7.0  # constant channel
    norm = F.Normalization.fit(windows)
    out = norm.apply(windows)
    np.testing.assert_allclose(out[:, :5].mean(axis=(0, 2)), 0, atol=1e-4)
    np.testing.assert_allclose(out[:, :5].std(axis=(0, 2)), 1, atol=1e-3)
    assert norm.std[5] == 1.0
    norm.save(tmp_path / "normalization.json")
    assert F.Normalization.load(tmp_path / "normalization.json") == norm


def test_normalization_rejects_contract_mismatch():
    data = F.Normalization.identity().to_dict()
    with pytest.raises(ValueError, match="contract"):
        F.Normalization.from_dict({**data, "gyro_units": "deg_s"})
    with pytest.raises(ValueError, match="positive"):
        F.Normalization.from_dict({**data, "std": [1, 1, 1, 1, 1, 0]})


def test_prepare_ride_applies_unit_conversion():
    time_index, values = ride(300)
    in_rad = F.prepare_ride(time_index, values, "rad_s", F.Normalization.identity())
    in_deg = F.prepare_ride(
        time_index,
        F.to_rad_s(values, "rad_s") * [1, 1, 1, *[F.DEG_PER_RAD] * 3],
        "deg_s",
        F.Normalization.identity(),
    )
    np.testing.assert_allclose(in_deg, in_rad, rtol=1e-5)
    assert in_rad.shape == (1, 6, 300)


def test_aggregate_ride_score_is_the_max():
    assert F.aggregate_ride_score(np.array([0.1, 0.8, 0.3])) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        F.aggregate_ride_score(np.array([]))
    with pytest.raises(ValueError):
        F.aggregate_ride_score(np.array([0.1, np.nan]))
