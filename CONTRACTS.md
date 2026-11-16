# Contracts

Interfaces between the components of the project. Each owner can work
independently as long as these contracts hold. **Any change goes through a PR
that edits this file**, reviewed by the producer and the consumer.

| # | Contract | Producer → consumer |
|---|---|---|
| 1 | [Processed data](#1-processed-data) | pipelines (Lucas) → training (Ariane) |
| 2 | [Model in MLflow](#2-model-in-mlflow) | training (Ariane) → API (Mathilde) |
| 3 | [HTTP API](#3-http-api) | API (Mathilde) → web app, e2e tests (Laura) |
| 4 | [Environment variables](#4-environment-variables) | everyone |
| 5 | [Docker images](#5-docker-images) | infra (Emma), CI (Laura) |
| 6 | [Open questions](#6-open-questions) | team + teachers |

The code version of contracts 1 to 3 lives in
[`src/gbg_mlops/features.py`](src/gbg_mlops/features.py) and
[`src/gbg_mlops/schemas.py`](src/gbg_mlops/schemas.py): import them, do not copy values.

## 0. Raw data, as observed

| | `SRV3490.parquet` (training) | `GBG500.parquet` (evaluation only) |
|---|---|---|
| Rides / rows | 3,490 / 16,892,002 | 500 / 1,529,547 |
| Columns | `ride_id, crash_ride_id, city, time_index, ax, ay, az, rx, ry, rz, topple` | `ride_id, time_index, ax, ay, az, rx, ry, rz` |
| Points per ride | 1,221 – 10,476 (median 4,743) | 1,807 – 4,767 (median 2,906) |
| Gyroscope units | rad/s (std of `rx` ≈ 0.10) | **°/s** (std of `rx` ≈ 5.7) |
| Missing values | 271,518 rows, whole rows (all 6 channels null), runs of 1–21 points, never at the start or end of a ride | 10,201 rows, same pattern (runs ≤ 19) |
| Duplicated `time_index` | 9 (8 with different values) | 1 |
| Labels | `topple` = 1 on 761 points in the 698 crash rides (663 rides have 1 point, 35 have 2–11 separate events) | `GBG500_labels.csv` (`RIDE_ID, LABEL`): Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6 |

`time_index` starts at 0 and increases by 100 ms (10 Hz) everywhere, apart from the duplicates.
SRV3490 groups: each crash ride (`crash_ride_id` null) has exactly 4 baseline rides whose
`crash_ride_id` points to it. Cities are coded `0`–`7`.

> ⚠️ **DuckDB tip.** `fetchnumpy()` returns `MaskedArray`s: `np.asarray(col, dtype=float)`
> silently turns nulls into numbers. Use `np.ma.filled(np.ma.asarray(col, dtype=float), np.nan)`
> or `.df()` so that missing values stay `NaN` and are handled by `clean_ride`.

## 1. Processed data

**Producer**: `pipelines/` · **Consumer**: `training/`, evaluation.

### Per-ride processing (`features.py`)

Applied in this order, by the pipeline **and** by the API (`prepare_ride`):

1. **Channels** in the order `CHANNELS = (ax, ay, az, rx, ry, rz)`.
2. **Units**: accelerations in m/s², gyroscope in **rad/s**. GBG500 is converted with
   `to_rad_s(values, "deg_s")` (÷ 57.2958). SRV3490 is already in rad/s.
3. **Cleaning** `clean_ride`: sort by `time_index`, keep the first of duplicated
   timestamps, require a 100 ms step, linearly interpolate missing values per channel.
4. **Windows** `make_windows`: `WINDOW = 300` points (30 s), `STRIDE = 150`; one last window
   aligned on the end of the ride is added so that every point is covered. Rides shorter
   than 300 points are rejected (none in the data).
5. **Normalisation** `Normalization`: per-channel mean and std computed **on the train
   split only** (`Normalization.fit(X_train)`), saved as `normalization.json`, then applied
   to every split and at inference.

These are defaults defined in `features.py`; they may change, but only with a PR that
also regenerates the processed data and retrains the model.

### Window labels (SRV3490)

`y = 1` if the window contains at least one point with `topple = 1`
(`window_labels`), else `0`. All windows of baseline rides are `0`.

### Split

- `group_id` = the crash ride id, shared by the crash ride **and** its 4 baselines
  (`crash_ride_id` for baselines, `ride_id` for the crash ride).
- Split **by group, never by window**: 70 % train / 15 % val / 15 % test of the 698 groups,
  seed `42`. The list of groups per split is saved in `data/processed/split.json`.

### Files

```text
data/processed/
├── train.npz  val.npz  test.npz   # SRV3490
├── gbg_eval.npz                    # GBG500, evaluation only
├── gbg_eval_labels.csv             # ride_id,label  (from GBG500_labels.csv)
├── normalization.json              # fitted on train.npz
└── split.json                      # {"train": [group_id, ...], "val": [...], "test": [...]}
```

| Key in `.npz` | dtype | Shape | Content |
|---|---|---|---|
| `X` | float32 | `(n, 6, 300)` | normalised windows, channels in `CHANNELS` order |
| `y` | int64 | `(n,)` | window label (SRV files only) |
| `ride_id` | unicode `<U64` | `(n,)` | ride of each window |
| `group_id` | unicode `<U64` | `(n,)` | group of each window (= `ride_id` in `gbg_eval.npz`) |
| `window_start` | int64 | `(n,)` | index of the first point of the window in the cleaned ride |

Save with `np.savez_compressed(path, X=..., y=..., ...)`; strings as fixed-width unicode
so that `np.load(path)` works without `allow_pickle`. GBG labels are **not** in the
`.npz`: join `gbg_eval_labels.csv` on `ride_id`.

`normalization.json` example:

```json
{
  "channels": ["ax", "ay", "az", "rx", "ry", "rz"],
  "mean": [-0.41, -9.12, -1.37, 0.0, 0.0, 0.0],
  "std": [1.25, 1.34, 1.31, 0.10, 0.37, 0.15],
  "window": 300,
  "stride": 150,
  "gyro_units": "rad_s"
}
```

*(values illustrative)*

## 2. Model in MLflow

**Producer**: `training/` · **Consumer**: API, evaluation.

| Item | Value |
|---|---|
| Experiment | `gbg-risk` |
| Registered model | `gbg-risk` |
| Served version | alias `champion` → `models:/gbg-risk@champion` |
| Flavor | `mlflow.pyfunc` (e.g. a `PythonModel` wrapping the PyTorch network) |
| Input | **normalised** windows, `np.float32` array of shape `(n_windows, 6, 300)` |
| Output | `np.ndarray` of shape `(n_windows,)`, float, probability of risk in `[0, 1]` per window |
| Signature | mandatory: input `TensorSpec(float32, (-1, 6, 300))`, output `TensorSpec(float32, (-1,))` |
| Artifacts of the run | `normalization.json` at the run root (`mlflow.log_dict(norm.to_dict(), "normalization.json")`) |
| Artifact store | `s3://mlflow` on MinIO |

```python
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np

signature = ModelSignature(
    inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 6, 300))]),
    outputs=Schema([TensorSpec(np.dtype(np.float32), (-1,))]),
)
```

- The model does **not** normalise: the caller applies `normalization.json` first.
- **Ride score** = maximum of its window scores (`aggregate_ride_score`), the same in
  evaluation and in the API.
- Promotion: a new version is registered without moving `champion`; moving the alias is
  an explicit, reviewed step (the previous champion stays available for rollback).
- Recommended tags on the version: `git_commit`, `data_split` (hash of `split.json`),
  and evaluation metrics on GBG500 as run metrics.

## 3. HTTP API

**Producer**: `api/` · **Consumers**: web app, e2e tests, CI smoke tests.
Swagger UI: `http://localhost:8000/docs`, OpenAPI: `/openapi.json`.

### `GET /health`

| Code | When | Body |
|---|---|---|
| 200 | model loaded | `{"status": "ready", "model_name": "gbg-risk", "model_version": "3", "model_alias": "champion", "detail": null}` |
| 503 | model not loaded (registry down, no `champion`, invalid artifacts) | `{"status": "not_ready", "model_name": "gbg-risk", "model_version": null, "model_alias": "champion", "detail": "..."}` |

### `POST /predict`

Request (`application/json`):

```json
{
  "ride_id": "demo-ride-001",
  "units": "deg_s",
  "samples": [
    {"time_index": 0,   "ax": -2.513, "ay": -9.385, "az": -1.053, "rx": -0.009, "ry": 0.009, "rz": -0.201},
    {"time_index": 100, "ax": -2.491, "ay": -9.385, "az": -1.079, "rx": -0.037, "ry": 0.027, "rz": -0.156}
  ]
}
```

- `ride_id`: optional string (≤ 128 chars), echoed back.
- `units`: **required**, `"deg_s"` (GBG500) or `"rad_s"` (SRV3490). No default, to avoid a
  silent 57× error on the gyroscope.
- `samples`: 300 to 20,000 items, sorted, `time_index` increasing by exactly 100 ms.
  Channel values are numbers or `null` (missing, interpolated); NaN/Infinity, booleans and
  unknown fields are rejected. A channel cannot be entirely `null`.

Response 200:

```json
{
  "prediction_id": "3f6c2a8e-1b7d-4c55-9d0e-2a4b8f9c1e77",
  "ride_id": "demo-ride-001",
  "risk_score": 0.05,
  "is_reckless": false,
  "threshold": 0.5,
  "predicted_class": null,
  "model_name": "gbg-risk",
  "model_version": "1",
  "n_windows": 19
}
```

- `risk_score` = max of window probabilities; `is_reckless` = `risk_score >= threshold`;
  `threshold` = `RISK_THRESHOLD`.
- `predicted_class` is always `null` for now (see open questions).

### `POST /predict/file`

`multipart/form-data` with `file` (`.csv` or `.parquet`, one ride, columns
`time_index, ax, ay, az, rx, ry, rz`; extra columns such as `ride_id` are ignored) and
form field `units` (`deg_s` | `rad_s`). Unlike the JSON endpoint, the file is cleaned
with `clean_ride` (sorted, duplicates dropped), which matches the raw parquet files.
Same response as `/predict`.

### Status codes

| Code | Meaning | Body |
|---|---|---|
| 200 | prediction done | `PredictResponse` |
| 422 | invalid input (schema, too short/long, irregular time, bad file) | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` — input values are not echoed |
| 503 | model not ready | `{"detail": "Model not ready"}` |

## 4. Environment variables

Read through `gbg_mlops.config.Settings` (`.env` in dev, see `.env.example`).

| Variable | Dev value (compose, from host) | Used by | Meaning |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` (`http://mlflow:5000` inside compose) | training, API, scripts | MLflow server |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` (`http://minio:9000` inside compose) | training, API | S3 endpoint for artifacts |
| `AWS_ACCESS_KEY_ID` | dev value in `.env.example` | training, API | MinIO credentials |
| `AWS_SECRET_ACCESS_KEY` | dev value in `.env.example` | training, API | MinIO credentials |
| `MODEL_NAME` | `gbg-risk` | API, scripts | registered model |
| `MODEL_ALIAS` | `champion` | API | alias to serve |
| `RISK_THRESHOLD` | `0.5` | API | decision threshold on `risk_score`, in `]0, 1[` |

Real credentials never go in Git: `.env` is ignored; CI and Kubernetes use secrets.

## 5. Docker images

| Image | Content | Tags |
|---|---|---|
| `<dockerhub_user>/gbg-api` | FastAPI service (`api/`, `src/gbg_mlops/`), port 8000 | commit SHA (`${GITHUB_SHA}`) + `latest` |
| `<dockerhub_user>/gbg-app` | test web app (`app/`) | commit SHA + `latest` |

Deployments reference the SHA tag (reproducible); `latest` is for local convenience only.
The API image does not contain the model: it loads `MODEL_NAME@MODEL_ALIAS` at startup.

## 6. Open questions

| Question | Options | Current choice |
|---|---|---|
| Target: binary *Reckless* detection or 5-class classification? | binary score (fits the "risk" framing and the 6 positives) / 5 classes (needs labelled training data we do not have) | binary score; `predicted_class` reserved and `null` — **to confirm with the teachers** |
| How does a model trained on falls (SRV3490) detect *Reckless* riding (GBG500)? | fall probability used as a risk proxy / anomaly score (e.g. autoencoder on normal rides) | to decide by Ariane with the GBG500 results |
| Decision threshold | fixed `0.5` / tuned on SRV val / tuned on GBG500 (only 6 positives → risk of overfitting) | `0.5` via `RISK_THRESHOLD`, revisit after evaluation |
| Window size and stride | `300/150` / shorter windows around falls | `300/150` defaults |
| Evaluation metric on GBG500 | ROC-AUC, PR-AUC, recall of Reckless at fixed false positive rate on Safe overall / Bad weather / Tandem | to agree with Ariane |
