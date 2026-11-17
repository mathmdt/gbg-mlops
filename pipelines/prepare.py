"""Raw parquet files → windows ready for training (CONTRACTS.md, section 1).

    uv run python -m pipelines.prepare                                    # make data
    uv run python -m pipelines.prepare --sample-groups 50 --out data/processed/sample

SRV3490 is read city by city with DuckDB (the whole file does not fit in pandas).
Each ride is cleaned, cut into windows and labelled with `topple`. The split is made
by group (a crash ride and its 4 baselines) so that no fall context leaks between
train, val and test. The normalisation is fitted on train only and applied to every
file, GBG500 included (after its gyroscope is converted from °/s to rad/s).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np

from gbg_mlops.features import (
    CHANNELS,
    SAMPLE_PERIOD_MS,
    Normalization,
    Units,
    clean_ride,
    make_windows,
    to_rad_s,
    window_labels,
    window_starts,
)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SEED = 42
TRAIN, VAL = 0.70, 0.15  # test = the rest
ID_DTYPE = "<U64"  # fixed-width strings: np.load works without allow_pickle
COLUMNS = ", ".join(CHANNELS)


def floats(column) -> np.ndarray:
    """DuckDB column → float64 with nulls as NaN (`fetchnumpy` returns masked arrays)."""
    return np.ma.filled(np.ma.asarray(column, dtype=np.float64), np.nan)


def split_groups(con, srv: Path, sample_groups: int | None = None) -> dict[str, list[str]]:
    """70/15/15 split of the groups, stratified on crash rides with several topple points."""
    rows = con.execute(
        "SELECT coalesce(crash_ride_id, ride_id), sum(coalesce(topple, 0)) > 1 "
        "FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
        [str(srv)],
    ).fetchall()
    order = np.random.default_rng(SEED).permutation(len(rows))[:sample_groups]
    groups = np.array([rows[i][0] for i in order])
    multi = np.array([rows[i][1] for i in order], dtype=bool)
    # Spread each stratum evenly along the list, then cut it: same share of
    # multi-fall groups in every split, and split sizes exactly 70/15/15.
    rank = np.empty(len(groups))
    for stratum in (False, True):
        members = np.flatnonzero(multi == stratum)
        rank[members] = (np.arange(len(members)) + 0.5) / len(members)
    groups = groups[np.argsort(rank, kind="stable")].tolist()
    n_train, n_val = round(len(groups) * TRAIN), round(len(groups) * VAL)
    split = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }
    if not all(split.values()):
        raise ValueError(f"{len(groups)} groups are too few for a train/val/test split")
    return {name: sorted(ids) for name, ids in split.items()}


def rides(con, query: str, params: list) -> Iterator[dict[str, np.ndarray]]:
    """Run a query ordered by `ride_id` and yield the columns of one ride at a time."""
    cols = con.execute(query, params).fetchnumpy()
    ids = np.asarray(cols["ride_id"])
    if len(ids) == 0:  # e.g. a city with no ride of this split
        return
    cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    for start, end in zip(np.r_[0, cuts], np.r_[cuts, len(ids)], strict=True):
        yield {name: col[start:end] for name, col in cols.items()}


def process_ride(cols: dict[str, np.ndarray], units: Units):
    """One raw ride → windows `(k, 6, 300)`, window starts `(k,)`, labels `(k,)`."""
    raw_time = np.asarray(cols["time_index"], dtype=np.int64)
    values = to_rad_s(np.column_stack([floats(cols[c]) for c in CHANNELS]), units)
    time_index, clean = clean_ride(raw_time, values)
    topple = np.zeros(len(time_index))
    if "topple" in cols:
        # A duplicated timestamp keeps the topple of any of its rows: no fall is lost.
        position = (raw_time - time_index[0]) // SAMPLE_PERIOD_MS
        np.maximum.at(topple, position, np.nan_to_num(floats(cols["topple"])))
    return make_windows(clean), window_starts(len(clean)), window_labels(topple)


def collect(ride_iter, units: Units, group_col: str = "ride_id") -> dict[str, np.ndarray]:
    """Stack the windows of several rides into the arrays of one `.npz` file."""
    parts = {"X": [], "y": [], "ride_id": [], "group_id": [], "window_start": []}
    for cols in ride_iter:
        ride_id, group_id = str(cols["ride_id"][0]), str(cols[group_col][0])
        if max(len(ride_id), len(group_id)) > int(ID_DTYPE[2:]):
            raise ValueError(f"id longer than {ID_DTYPE} characters: {ride_id!r}")
        windows, starts, labels = process_ride(cols, units)
        parts["X"].append(windows)
        parts["y"].append(labels)
        parts["ride_id"].append(np.full(len(windows), ride_id, dtype=ID_DTYPE))
        parts["group_id"].append(np.full(len(windows), group_id, dtype=ID_DTYPE))
        parts["window_start"].append(starts)
    if not parts["X"]:
        raise ValueError("no ride to process")
    return {key: np.concatenate(arrays) for key, arrays in parts.items()}


def srv_rides(con, srv: Path, groups: list[str]) -> Iterator[dict[str, np.ndarray]]:
    """Rides of SRV3490 whose group is in `groups`, fetched one city at a time."""
    cities = con.execute("SELECT DISTINCT city FROM read_parquet(?)", [str(srv)]).fetchall()
    query = (
        f"SELECT ride_id, coalesce(crash_ride_id, ride_id) AS group_id, time_index, {COLUMNS}, "
        "topple FROM read_parquet(?, file_row_number = true) "
        "WHERE city = ? AND coalesce(crash_ride_id, ride_id) IN (SELECT unnest(?)) "
        "ORDER BY ride_id, file_row_number"  # file order inside a ride: first duplicate wins
    )
    for (city,) in sorted(cities):
        yield from rides(con, query, [str(srv), city, groups])


def save(path: Path, data: dict, norm: Normalization) -> None:
    np.savez_compressed(path, **{**data, "X": norm.apply(data["X"])})
    positives = f", {data['y'].mean():.2%} positive" if "y" in data else ""
    print(f"{path}: {len(data['X'])} windows, {len(set(data['ride_id']))} rides{positives}")


def prepare(raw: Path = RAW_DIR, out: Path = OUT_DIR, sample_groups: int | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    srv, gbg = raw / "SRV3490.parquet", raw / "GBG500.parquet"
    con = duckdb.connect()

    split = split_groups(con, srv, sample_groups)
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    parts = {
        name: collect(srv_rides(con, srv, ids), "rad_s", group_col="group_id")
        for name, ids in split.items()
    }
    norm = Normalization.fit(parts["train"]["X"])
    norm.save(out / "normalization.json")
    for name, data in parts.items():
        save(out / f"{name}.npz", data, norm)
    del parts

    query = (
        f"SELECT ride_id, time_index, {COLUMNS} FROM read_parquet(?, file_row_number = true) "
        "ORDER BY ride_id, file_row_number"
    )
    gbg_data = collect(rides(con, query, [str(gbg)]), "deg_s")
    del gbg_data["y"]  # GBG labels are ride-level, in gbg_eval_labels.csv
    save(out / "gbg_eval.npz", gbg_data, norm)

    with (raw / "GBG500_labels.csv").open(encoding="utf-8-sig", newline="") as src:
        labels = [(row["RIDE_ID"].strip(), row["LABEL"].strip()) for row in csv.DictReader(src)]
    missing = set(gbg_data["ride_id"]) - {ride_id for ride_id, _ in labels}
    if missing:
        raise ValueError(f"{len(missing)} GBG500 rides have no label, e.g. {sorted(missing)[0]}")
    with (out / "gbg_eval_labels.csv").open("w", newline="") as dst:
        csv.writer(dst).writerows([("ride_id", "label"), *labels])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="folder of the raw files")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    parser.add_argument("--sample-groups", type=int, help="keep only N groups (quick sample)")
    args = parser.parse_args(argv)
    prepare(args.raw, args.out, args.sample_groups)


if __name__ == "__main__":
    main()
