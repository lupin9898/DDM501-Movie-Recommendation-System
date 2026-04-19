.PHONY: help setup gen-secrets data train train-all serve test test-quick \
        lint lint-check typecheck docker-up docker-down docker-logs docker-ps \
        health train-and-reload mlflow-ui clean

COMPOSE := docker compose -f docker-compose.prod.yml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local dev setup ──────────────────────────────────────────────────────────
setup:  ## Install Python dependencies for dev + tests
	pip install -r requirements.txt -r requirements-test.txt

gen-secrets:  ## Generate .env with random passwords (fails if .env exists)
	./scripts/gen-secrets.sh

# ── Data & training ──────────────────────────────────────────────────────────
data:  ## Preprocess raw MovieLens data (ratings.csv + movies.csv must exist in data/raw/)
	python -m src.data.preprocessing

train:  ## Train LightFM hybrid model (logs to MLflow)
	python -m src.training.train --model lightfm

train-all:  ## Train all model types (random, popular, lightfm)
	python -m src.training.train --model random
	python -m src.training.train --model popular
	python -m src.training.train --model lightfm

# ── API ──────────────────────────────────────────────────────────────────────
serve:  ## Run API locally with hot reload (uvicorn)
	uvicorn src.serving.app:app --reload --port 8000

# ── Quality gates ────────────────────────────────────────────────────────────
test:  ## Run tests with coverage
	pytest tests/ -v --cov=src

test-quick:  ## Run tests, stop on first failure
	pytest tests/ -x -q

lint:  ## Lint + auto-format
	ruff check src/ tests/ --fix
	ruff format src/ tests/

lint-check:  ## Lint check without fixing (CI)
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:  ## Run mypy (warning-only for now)
	mypy src/ || true

# ── Docker stack ─────────────────────────────────────────────────────────────
docker-up:  ## Start full production stack (docker-compose.prod.yml)
	$(COMPOSE) up -d --build

docker-down:  ## Stop and remove all services
	$(COMPOSE) down

docker-logs:  ## Tail logs for a service. Usage: make docker-logs SERVICE=api
	$(COMPOSE) logs -f --tail=100 $(SERVICE)

docker-ps:  ## List running services with status
	$(COMPOSE) ps

# ── Operator shortcuts ───────────────────────────────────────────────────────
health:  ## Curl /health on every HTTP service
	@printf "API        "; curl -fsS http://localhost:8000/health      | head -c 200 || echo "  DOWN"
	@printf "\nMLflow     "; curl -fsS http://localhost:5000/health      | head -c 80  || echo "  DOWN"
	@printf "\nGrafana    "; curl -fsS http://localhost:3000/api/health  | head -c 80  || echo "  DOWN"
	@printf "\nPrometheus "; curl -fsS http://localhost:9090/-/healthy   | head -c 80  || echo "  DOWN"
	@printf "\nEvidently  "; curl -fsS http://localhost:8001/health      | head -c 80  || echo "  DOWN"
	@echo

train-and-reload:  ## Train locally then restart API to pick up new model
	python -m src.training.train --model lightfm
	$(COMPOSE) restart api
	@echo "API restarted — check /health for new model_version"

mlflow-ui:  ## Open MLflow UI in browser
	@echo "Opening http://localhost:5000"
	@open http://localhost:5000 2>/dev/null || xdg-open http://localhost:5000 2>/dev/null || true

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:  ## Remove caches and coverage artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
