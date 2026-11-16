"""Extract one ride from a raw parquet file, ready for `POST /predict/file`.

    uv run python scripts/export_ride.py --ride-id <id> --out ride.csv
    uv run python scripts/export_ride.py --list Reckless      # ride ids with this label

Reads only the matching rows (pyarrow filter), so it also works on SRV3490.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from gbg_mlops.features import CHANNELS

DEFAULT_INPUT = Path("data/raw/GBG500.parquet")
DEFAULT_LABELS = Path("data/raw/GBG500_labels.csv")
COLUMNS = ["ride_id", "time_index", *CHANNELS]


def read_ride(source: Path, ride_id: str) -> pd.DataFrame:
    """Rows of one ride, sorted by time; raises if the ride does not exist."""
    table = pq.read_table(source, columns=COLUMNS, filters=[("ride_id", "==", ride_id)])
    if table.num_rows == 0:
        raise ValueError(f"ride {ride_id!r} not found in {source}")
    return table.to_pandas().sort_values("time_index", kind="stable").reset_index(drop=True)


def write_ride(frame: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        frame.to_csv(out, index=False)
    elif out.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(out, index=False)
    else:
        raise ValueError("--out must end with .csv or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--ride-id", help="ride to export")
    action.add_argument("--list", metavar="LABEL", help="print ride ids having this label")
    parser.add_argument("--out", type=Path, help="output .csv or .parquet")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    try:
        if args.list:
            labels = pd.read_csv(args.labels)
            ids = labels.loc[labels["LABEL"].str.lower() == args.list.lower(), "RIDE_ID"]
            print("\n".join(ids))
            return
        if args.out is None:
            parser.error("--out is required with --ride-id")
        frame = read_ride(args.input, args.ride_id)
        write_ride(frame, args.out)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"export_ride: {error}")
    missing = int(frame[list(CHANNELS)].isna().all(axis=1).sum())
    print(f"{args.out}: {len(frame)} points, {missing} missing rows, ride {args.ride_id}")


if __name__ == "__main__":
    main()
