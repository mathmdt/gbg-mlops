# Common developer commands. Run `make help` for the list.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format test up down dummy-model api

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

up: ## Start the dev stack (Postgres, MinIO, MLflow, API)
	docker compose up -d --build

down: ## Stop the dev stack (volumes are kept)
	docker compose down

dummy-model: ## Register the placeholder model as gbg-risk@champion
	$(UV) run python scripts/register_dummy_model.py

api: ## Run the API locally with auto-reload on http://localhost:8000
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
