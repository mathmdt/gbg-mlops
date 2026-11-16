"""export_ride reads one ride from a parquet file and writes CSV or Parquet."""

import numpy as np
import pandas as pd
import pytest

from scripts.export_ride import read_ride, write_ride


@pytest.fixture
def source(tmp_path):
    rows = [
        {"ride_id": rid, "time_index": t, **{c: 1.0 for c in ("ax", "ay", "az", "rx", "ry", "rz")}}
        for rid in ("a", "b")
        for t in (200, 0, 100)
    ]
    rows[0]["ax"] = np.nan
    path = tmp_path / "rides.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_reads_only_the_requested_ride_sorted(source):
    frame = read_ride(source, "a")
    assert frame["ride_id"].unique().tolist() == ["a"]
    assert frame["time_index"].tolist() == [0, 100, 200]


def test_unknown_ride_is_an_error(source):
    with pytest.raises(ValueError, match="not found"):
        read_ride(source, "zzz")


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_roundtrip(source, tmp_path, suffix):
    out = tmp_path / f"ride{suffix}"
    write_ride(read_ride(source, "b"), out)
    back = pd.read_csv(out) if suffix == ".csv" else pd.read_parquet(out)
    assert len(back) == 3 and list(back.columns)[:2] == ["ride_id", "time_index"]
    with pytest.raises(ValueError):
        write_ride(read_ride(source, "b"), tmp_path / "ride.json")
