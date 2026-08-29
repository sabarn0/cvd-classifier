.PHONY: install data train test mlflow-ui build up down logs smoke-test clean

VENV_PY=python

install:
	$(VENV_PY) -m pip install -r requirements.txt

## Download + preprocess the dataset (kagglehub)
data:
	$(VENV_PY) data/prepare_data.py

## Fast local smoke test with a tiny subset of data (no need for full download)
data-smoke:
	$(VENV_PY) data/prepare_data.py --max-per-class 60

## Train the model, tracked with MLflow
train:
	$(VENV_PY) -m src.train --epochs 5 --batch-size 32

train-smoke:
	$(VENV_PY) -m src.train --epochs 1 --batch-size 8 --max-train-samples 40 --max-val-samples 10 --max-test-samples 10

## Unit tests
test:
	$(VENV_PY) -m pytest -q

## Launch the MLflow UI (http://localhost:5000) against local ./mlruns
mlflow-ui:
	$(VENV_PY) -m mlflow ui --backend-store-uri mlruns --port 5000

## Build & start the full local stack: API + MLflow + Prometheus + Grafana
build:
	docker compose build

up:
	docker compose up -d
	@echo "API:        http://localhost:8000/docs"
	@echo "MLflow:     http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"

down:
	docker compose down

logs:
	docker compose logs -f api

## Post-deploy smoke test: health + one prediction call
smoke-test:
	bash scripts/smoke_test.sh

clean:
	rm -rf __pycache__ .pytest_cache artifacts
	find . -name "__pycache__" -type d -exec rm -rf {} +
