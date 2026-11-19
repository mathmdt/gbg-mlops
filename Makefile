# Common developer commands. Run `make help` for the list.
.DEFAULT_GOAL := help
UV ?= uv
PREFECT_PORT ?= $(or $(shell grep -E '^PREFECT_PORT=' .env 2>/dev/null | cut -d= -f2),4200)

.PHONY: help install lint format test e2e env up down logs dummy-model promote rollback reload-api api \
	ingest data data-sample dag train compare-runs

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create .venv with API + dev dependencies (add --group training for model work)
	$(UV) sync

lint: ## Check style and common errors
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-fix lint issues and format the code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test: ## Run unit and integration tests (e2e excluded)
	$(UV) run pytest -m "not e2e"

e2e: ## Run end-to-end tests against a running API (API_URL, default http://localhost:8000)
	$(UV) run pytest -m e2e tests/e2e

env: ## Create .env from .env.example if it does not exist
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

up: env ## Start the dev stack (Postgres, MinIO, MLflow, API) and wait until healthy
	docker compose up -d --build --wait
	@echo "MLflow         http://localhost:$${MLFLOW_PORT:-$$(grep -E '^MLFLOW_PORT=' .env | cut -d= -f2)}"
	@echo "MinIO console  http://localhost:$${MINIO_CONSOLE_PORT:-$$(grep -E '^MINIO_CONSOLE_PORT=' .env | cut -d= -f2)}"
	@echo "API docs       http://localhost:$${API_PORT:-$$(grep -E '^API_PORT=' .env | cut -d= -f2)}/docs"
	@echo "Prefect UI     http://localhost:$(PREFECT_PORT)"

down: ## Stop the dev stack (volumes are kept)
	docker compose down

logs: ## Follow the logs of the dev stack
	docker compose logs -f --tail=100

dummy-model: ## Register the placeholder model as gbg-risk@champion, then reload the API
	$(UV) run python scripts/register_dummy_model.py
	@$(MAKE) --no-print-directory reload-api

promote: ## Promote VERSION=N to champion if its GBG500 metric is better (FORCE=1 to skip), reload the API
	@test -n "$(VERSION)" || (echo "usage: make promote VERSION=N [FORCE=1]" && exit 1)
	$(UV) run python scripts/promote_model.py --candidate-version $(VERSION) $(if $(FORCE),--force)
	@$(MAKE) --no-print-directory reload-api

rollback: ## Move champion back to the previous version, reload the API
	$(UV) run python scripts/promote_model.py --rollback
	@$(MAKE) --no-print-directory reload-api

reload-api: ## Restart the compose API so it loads the current champion
	@if docker compose ps --status running --services 2>/dev/null | grep -qx api; then \
		docker compose restart api && docker compose up -d --wait api; \
	else echo "API container not running: skipped"; fi

api: ## Run the API locally with auto-reload on http://localhost:8000
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ingest: ## Upload data/raw/* to the MinIO bucket raw with their SHA-256 (needs `make up`)
	$(UV) run python -m pipelines.ingest

data: ## Rebuild data/processed/ (windows, split, normalisation) from data/raw/
	$(UV) run python -m pipelines.prepare

data-sample: ## Same on 50 groups only, in data/processed/sample/
	$(UV) run python -m pipelines.prepare --sample-groups 50 --out data/processed/sample

train: ## Train the 1D CNN model using training/train.py
	$(UV) run --group training python -m training.train

compare-runs: ## Run and compare 3 model variations in MLflow (step 8)
	$(UV) run --group training python scripts/compare_runs.py

dag: ## Run the DAG ingest → prepare → train → evaluate → promote (Prefect UI)
	PREFECT_API_URL=http://localhost:$(PREFECT_PORT)/api \
		$(UV) run --group pipelines --group training python -m pipelines.dag
