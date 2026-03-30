.PHONY: setup data train serve test lint docker-up docker-down mlflow-ui clean

setup:
	pip install -r requirements.txt -r requirements-test.txt

data:
	python -m src.data.ingestion
	python -m src.data.preprocessing

train:
	python -m src.training.train --model als

train-all:
	python -m src.training.train --model random
	python -m src.training.train --model popular
	python -m src.training.train --model als

serve:
	uvicorn src.serving.app:app --reload --port 8000

test:
	pytest tests/ -v --cov=src

test-quick:
	pytest tests/ -x -q

lint:
	ruff check src/ tests/
	ruff format src/ tests/

lint-check:
	ruff check src/ tests/
	ruff format --check src/ tests/

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

mlflow-ui:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
