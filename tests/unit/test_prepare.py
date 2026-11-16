"""prepare.py turns synthetic raw files (same layout as SRV3490 / GBG500) into the contract."""

import csv
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gbg_mlops.features import CHANNELS, WINDOW, Normalization
from pipelines.prepare import prepare

N_GROUPS = 20
GBG_GYRO_DEG_S = 90.0  # constant gyroscope in GBG500 → π/2 rad/s after conversion


def column(values) -> pa.Array:
    values = np.asarray(values, dtype=float)
    return pa.array(values, mask=np.isnan(values))  # NaN → null, as in the raw files


def write_raw(raw, n_groups=N_GROUPS, seed=0):
    rng = np.random.default_rng(seed)
    srv = {k: [] for k in ("ride_id", "crash_ride_id", "city", "time_index", *CHANNELS, "topple")}
    for g in range(n_groups):
        crash = f"crash{g:02d}"
        for k in range(5):
            n = int(rng.integers(320, 900))
            values = rng.normal(size=(n, 6)) * [1, 1, 1, 0.1, 0.1, 0.1]
            values[n // 2 : n // 2 + 5] = np.nan  # run of missing rows
            topple = np.zeros(n)
            if k == 0:  # the crash ride; every 7th group has two falls
                topple[rng.choice(n, size=1 + (g % 7 == 0), replace=False)] = 1
            time_index = np.arange(n) * 100
            if k == 1:  # duplicated timestamp after the original row: must be dropped
                time_index = np.r_[time_index, 1000]
                values = np.vstack([values, np.full(6, 1000.0)])
                topple = np.r_[topple, 0]
            srv["ride_id"] += [crash if k == 0 else f"base{g:02d}_{k}"] * len(time_index)
            srv["crash_ride_id"] += [None if k == 0 else crash] * len(time_index)
            srv["city"] += [g % 3] * len(time_index)
            srv["time_index"] += time_index.tolist()
            for c, channel in enumerate(CHANNELS):
                srv[channel] += values[:, c].tolist()
            srv["topple"] += topple.tolist()
    table = {k: column(v) if k in CHANNELS else pa.array(v) for k, v in srv.items()}
    pq.write_table(pa.table(table), raw / "SRV3490.parquet")

    gbg = {k: [] for k in ("ride_id", "time_index", *CHANNELS)}
    for r in range(4):
        n = int(rng.integers(400, 700))
        values = np.c_[rng.normal(size=(n, 3)), np.full((n, 3), GBG_GYRO_DEG_S)]
        values[100:110] = np.nan
        gbg["ride_id"] += [f"gbg{r}"] * n
        gbg["time_index"] += (np.arange(n) * 100).tolist()
        for c, channel in enumerate(CHANNELS):
            gbg[channel] += values[:, c].tolist()
    table = {k: column(v) if k in CHANNELS else pa.array(v) for k, v in gbg.items()}
    pq.write_table(pa.table(table), raw / "GBG500.parquet")
    labels = [("gbg0", "Safe"), ("gbg1", "Reckless"), ("gbg2", "Tandem"), ("gbg3", "Safe")]
    with (raw / "GBG500_labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("RIDE_ID", "LABEL"), *labels])


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw")
    write_raw(path)
    return path


@pytest.fixture(scope="module")
def out(raw, tmp_path_factory):
    path = tmp_path_factory.mktemp("processed")
    prepare(raw, path)
    return path


def load(out, name):
    return dict(np.load(out / f"{name}.npz"))  # no allow_pickle: strings are fixed-width


def test_no_group_in_two_splits(out):
    split = json.loads((out / "split.json").read_text())
    sets = {name: set(ids) for name, ids in split.items()}
    assert sets["train"].isdisjoint(sets["val"] | sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert len(sets["train"] | sets["val"] | sets["test"]) == N_GROUPS
    assert [len(sets[n]) for n in ("train", "val", "test")] == [14, 3, 3]
    for name, ids in sets.items():
        assert set(load(out, name)["group_id"]) == ids


def test_files_follow_the_contract(out):
    for name in ("train", "val", "test", "gbg_eval"):
        data = load(out, name)
        n = len(data["X"])
        assert data["X"].shape == (n, len(CHANNELS), WINDOW) and data["X"].dtype == np.float32
        assert np.isfinite(data["X"]).all(), f"NaN in {name}"
        assert data["window_start"].shape == (n,) and data["window_start"].dtype == np.int64
        assert data["ride_id"].dtype == np.dtype("<U64") and data["group_id"].shape == (n,)
        if name == "gbg_eval":
            assert "y" not in data
            np.testing.assert_array_equal(data["group_id"], data["ride_id"])
        else:
            assert data["y"].shape == (n,) and data["y"].dtype == np.int64


def test_labels_come_from_topple(out):
    for name in ("train", "val", "test"):
        data = load(out, name)
        crash = np.char.startswith(data["ride_id"], "crash")
        assert data["y"][~crash].sum() == 0
        for ride in set(data["ride_id"][crash]):
            assert data["y"][data["ride_id"] == ride].sum() >= 1


def test_normalization_is_fitted_on_train(out):
    norm = Normalization.load(out / "normalization.json")
    train = load(out, "train")["X"]
    np.testing.assert_allclose(train.mean(axis=(0, 2)), 0, atol=1e-4)
    np.testing.assert_allclose(train.std(axis=(0, 2)), 1, atol=1e-3)
    raw = train * np.asarray(norm.std)[:, None] + np.asarray(norm.mean)[:, None]
    assert np.abs(raw).max() < 100  # the duplicated row (1000.0) was dropped


def test_gbg_gyroscope_is_converted_to_rad_s(out):
    norm = Normalization.load(out / "normalization.json")
    gyro = load(out, "gbg_eval")["X"][:, 3:]
    raw = gyro * np.asarray(norm.std[3:])[:, None] + np.asarray(norm.mean[3:])[:, None]
    np.testing.assert_allclose(raw, np.pi / 2, rtol=1e-4)


def test_gbg_labels_are_copied(out):
    with (out / "gbg_eval_labels.csv").open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["ride_id", "label"] and len(rows) == 5
    assert set(load(out, "gbg_eval")["ride_id"]) == {r[0] for r in rows[1:]}


def test_sample_keeps_only_n_groups(raw, tmp_path):
    prepare(raw, tmp_path, sample_groups=10)
    split = json.loads((tmp_path / "split.json").read_text())
    assert sum(len(ids) for ids in split.values()) == 10
    assert (tmp_path / "gbg_eval.npz").exists()
