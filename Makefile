# Common developer commands. Run `make help` for the list.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format test env up down logs dummy-model api

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

env: ## Create .env from .env.example if it does not exist
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

up: env ## Start the dev stack (Postgres, MinIO, MLflow, API) and wait until healthy
	docker compose up -d --build --wait
	@echo "MLflow         http://localhost:$${MLFLOW_PORT:-$$(grep -E '^MLFLOW_PORT=' .env | cut -d= -f2)}"
	@echo "MinIO console  http://localhost:$${MINIO_CONSOLE_PORT:-$$(grep -E '^MINIO_CONSOLE_PORT=' .env | cut -d= -f2)}"
	@echo "API docs       http://localhost:$${API_PORT:-$$(grep -E '^API_PORT=' .env | cut -d= -f2)}/docs"

down: ## Stop the dev stack (volumes are kept)
	docker compose down

logs: ## Follow the logs of the dev stack
	docker compose logs -f --tail=100

dummy-model: ## Register the placeholder model as gbg-risk@champion
	$(UV) run python scripts/register_dummy_model.py

api: ## Run the API locally with auto-reload on http://localhost:8000
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
