# GBG MLOps — risk detection on IMU rides

End-to-end MLOps pipeline around a deep learning model that scores bike/scooter
rides recorded by IMU sensors (accelerometer + gyroscope, 10 Hz).

- **Training data**: `SRV3490` — 3,490 rides in 8 European cities, 698 confirmed falls.
- **Evaluation data**: `GBG500` — 500 rides in Gothenburg; the target class is *Reckless*.
- **Serving**: FastAPI loads the model `gbg-risk@champion` from the MLflow Model
  Registry (artifacts on S3/MinIO) and returns a risk score per ride.

Albert School MLOps project — deadline 25/11/2026, defense 26/11/2026.

## Team

| Member | Scope |
|---|---|
| Mathilde | Repository, contracts, dev Docker Compose, dummy model, API, Registry + S3 |
| Lucas | Ingestion, preprocessing, DAG |
| Ariane | 1D CNN (PyTorch), MLflow tracking, evaluation on GBG500 |
| Emma | Dockerfiles, Docker Hub, Kubernetes/Helm, monitoring |
| Laura | CI/CD (GitHub Actions), test web app, end-to-end tests, final README |

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop.

```sh
make install        # .venv with API + dev dependencies
make lint
make test
```

Model work also needs the training group: `uv sync --group training`
(PyTorch has no wheel for Intel Macs: train on Apple Silicon or in a Linux container).

Raw data stays out of Git: put `SRV3490.parquet`, `GBG500.parquet` and
`GBG500_labels.csv` in `data/raw/`.

### Dev stack (Docker Compose)

```sh
make up      # creates .env from .env.example if missing, builds, waits until healthy
make logs
make down    # stops; data is kept in the named volumes
```

| Service | URL | Credentials (dev only) |
|---|---|---|
| MLflow UI | http://localhost:5001 | — |
| MinIO console | http://localhost:9001 | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from `.env` |
| API (Swagger) | http://localhost:8000/docs | — |

Register the placeholder model `gbg-risk@champion` (needs the stack up; rerun = new version):

```sh
make dummy-model
```

MLflow is on 5001 because macOS uses port 5000 for AirPlay Receiver. Every port can be
changed in `.env`. Buckets `mlflow` (MLflow artifacts) and `raw` are created automatically.

## Repository layout

| Path | Content | Owner |
|---|---|---|
| `src/gbg_mlops/` | Shared package: settings, preprocessing, API schemas, model loader | Mathilde (features: Lucas) |
| `api/` | FastAPI application | Mathilde |
| `pipelines/` | Ingestion and DAG | Lucas |
| `training/` | Model training and evaluation | Ariane |
| `infra/k8s/` | Kubernetes manifests / Helm | Emma |
| `app/` | Test web app | Laura |
| `scripts/` | Utility scripts (dummy model registration) | Mathilde |
| `tests/` | `unit/`, `integration/`, `e2e/` | everyone |
| `docs/` | Documentation | everyone |

## Contributing

- Never push to `main`: one branch per feature, small explicit commits, PR with one review.
- Interfaces between components are fixed in [`CONTRACTS.md`](CONTRACTS.md); change
  them only through a PR that the whole team sees.
- See [`docs/`](docs/) for the documentation.
