# GBG MLOps — risk detection on IMU rides

End-to-end MLOps pipeline around a deep learning model that scores bike/scooter
rides recorded by IMU sensors (accelerometer + gyroscope, 10 Hz).

- **Training data**: `SRV3490` — 3,490 rides in 8 European cities, 698 confirmed falls.
- **Evaluation data**: `GBG500` — 500 rides in Gothenburg; the target class is *Reckless*.
- **Serving**: FastAPI loads the model `gbg-risk@champion` from the MLflow Model
  Registry (artifacts on S3/MinIO) and returns a risk score per ride.

Albert School MLOps project — deadline 25/11/2026, defense 26/11/2026.

```mermaid
flowchart LR
    RAW[(data/raw<br>SRV3490, GBG500)] --> PIPE[pipelines/<br>features.py]
    PIPE --> NPZ[(data/processed<br>*.npz + normalization.json)]
    NPZ --> TRAIN[training/<br>1D CNN]
    TRAIN --> MLF[MLflow<br>tracking + registry]
    MLF --> S3[(MinIO<br>s3://mlflow)]
    MLF -->|gbg-risk@champion| API[api/<br>FastAPI]
    API --> APP[app/<br>web app]
```

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
Raw data stays out of Git: put `SRV3490.parquet`, `GBG500.parquet` and
`GBG500_labels.csv` in `data/raw/`.

```sh
make install        # .venv with API + dev dependencies
make lint           # ruff
make test           # unit + integration tests (no server needed)
make up             # dev stack: Postgres, MinIO, MLflow, API (creates .env if missing)
make dummy-model    # registers gbg-risk@champion (placeholder) and reloads the API
curl localhost:8000/health
make down           # stops the stack; data is kept in named volumes
```

| Service | URL | Credentials (dev only) |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5001 | — |
| MinIO console | http://localhost:9001 | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from `.env` |

MLflow is on 5001 because macOS uses port 5000 for AirPlay Receiver. Every host port can
be changed in `.env` (e.g. `API_PORT=8002` if 8000 is busy). Buckets `mlflow` and `raw` are
created automatically. Model work also needs `uv sync --group training` (PyTorch has no
wheel for Intel Macs: train on Apple Silicon or in a Linux container).

### Score a ride

```sh
# JSON: 300 to 20,000 samples every 100 ms; null = missing value
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @ride.json

# File: one ride, columns time_index, ax, ay, az, rx, ry, rz (CSV or Parquet)
uv run python scripts/export_ride.py --list Reckless           # ride ids of a GBG500 label
uv run python scripts/export_ride.py --ride-id <id> --out data/demo/ride.csv
curl -X POST localhost:8000/predict/file -F file=@data/demo/ride.csv -F units=deg_s
```

```json
{"prediction_id": "05edde9c-…", "ride_id": "bc3a3597…", "risk_score": 0.05,
 "is_reckless": false, "threshold": 0.5, "predicted_class": null,
 "model_name": "gbg-risk", "model_version": "2", "n_windows": 24}
```

`units` is mandatory: `deg_s` for GBG500, `rad_s` for SRV3490. Full contract:
[`CONTRACTS.md`](CONTRACTS.md).

## Pour démarrer

Chacun peut avancer dès maintenant, sans attendre les autres : les interfaces sont
fixées dans [`CONTRACTS.md`](CONTRACTS.md) et un modèle factice est déjà servi.

| Qui | Dossier | Ce que tu peux faire dès maintenant | S'appuyer sur |
|---|---|---|---|
| **Lucas** | `pipelines/` | Lire les parquet avec DuckDB par lots, produire `data/processed/{train,val,test,gbg_eval}.npz`, `normalization.json` et `split.json` (découpage par `group_id`), puis le DAG (Airflow ou Prefect). | `features.py` : `clean_ride`, `to_rad_s`, `make_windows`, `window_labels`, `Normalization.fit`. Attention au piège `fetchnumpy()` (contrat, section 0). |
| **Ariane** | `training/` | CNN 1D PyTorch sur `(n, 6, 300)`, tracking dans l'expérience `gbg-risk`, enregistrement dans `gbg-risk` avec `MODEL_SIGNATURE` et `normalization.json`, évaluation sur GBG500 (score du trajet = max des fenêtres). En attendant les `.npz`, générer des tableaux factices de la bonne forme. | `model_loader.py` (constantes), `scripts/register_dummy_model.py` (exemple complet d'enregistrement), contrat section 2. Tag `model_type=cnn1d` pour que le modèle factice ne le remplace jamais. |
| **Emma** | `infra/`, `api/Dockerfile` | Optimiser l'image de l'API (multi-stage, taille), build multi-architecture `amd64` + `arm64` vers Docker Hub, manifestes Kubernetes / Helm pour Docker Desktop (Deployment, Service, Secrets, sondes sur `/health`). | `docker-compose.yml` (variables et dépendances), contrat sections 4 et 5. Ton Mac est Intel : les images doivent aussi tourner en `arm64` pour les autres. |
| **Laura** | `.github/workflows/`, `app/`, `tests/e2e/` | CI sur les PR (`make lint`, `make test`), build et push des images (SHA + `latest`) via le runner auto-hébergé ; web app qui appelle `/health` et `/predict/file` ; tests e2e marqués `@pytest.mark.e2e` contre `make up` + `make dummy-model`. | Swagger sur `/docs`, contrat section 3. Les tests e2e sont exclus de `make test` (`-m "not e2e"`). |
| **Mathilde** | `api/`, `src/gbg_mlops/`, `scripts/` | Relectures des PR, évolutions du contrat, intégration du vrai modèle dès qu'Ariane l'enregistre. | — |

Raccourcis utiles : `make help` liste toutes les commandes ; `make reload-api` fait
recharger le `champion` actuel par l'API après une promotion.

## Repository layout

| Path | Content | Owner |
|---|---|---|
| `src/gbg_mlops/` | Shared package: settings, preprocessing, API schemas, model loading | Mathilde (features: Lucas) |
| `api/` | FastAPI application and its Dockerfile | Mathilde (Dockerfile: Emma) |
| `pipelines/` | Ingestion and DAG | Lucas |
| `training/` | Model training and evaluation | Ariane |
| `infra/k8s/` | Kubernetes manifests / Helm | Emma |
| `app/` | Test web app | Laura |
| `scripts/` | Utility scripts (dummy model registration) | Mathilde |
| `docker/` | Dev-only images (MLflow server) | Mathilde |
| `tests/` | `unit/`, `integration/`, `e2e/` | everyone |
| `docs/` | Documentation | everyone |

## Contributing

- Never push to `main`: one branch per feature, small explicit commits, PR with one review.
- Interfaces between components are fixed in [`CONTRACTS.md`](CONTRACTS.md); change
  them only through a PR that the whole team sees.
- No data, model, `.env` or secret in Git.
