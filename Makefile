# Common developer commands. Run `make help` for the list.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format test env up down logs dummy-model promote rollback reload-api api

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
