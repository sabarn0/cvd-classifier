.PHONY: install data train test mlflow-ui build up down logs smoke-test clean \
        v2-install v2-tune v2-promote v2-mlflow-ui v2-build v2-up v2-down v2-logs v2-smoke-test

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
	@echo "API:        http://localhost:9000/docs"
	@echo "MLflow:     http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:9030 (admin/admin)"

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

# =============================================================================
# v2 targets (main-v2 branch) — all resources on 80XX ports
# =============================================================================

## Install v2 dependencies (CUDA PyTorch + Optuna)
v2-install:
	$(VENV_PY) -m pip install torch==2.3.1 torchvision==0.18.1 \
		--index-url https://download.pytorch.org/whl/cu121
	$(VENV_PY) -m pip install optuna==3.6.1 "optuna-integration[mlflow]==3.6.0"

## Run Optuna HPO (20 trials x 3 epochs, ~50 min on GPU)
v2-tune:
	$(VENV_PY) -m src.tune

## Quick smoke HPO (3 trials x 1 epoch, tiny subset)
v2-tune-smoke:
	$(VENV_PY) -m src.tune --n-trials 3 --epochs-per-trial 1 \
		--max-train-samples 200 --max-val-samples 50

## Retrain winner + promote to models/model.pt
v2-promote:
	$(VENV_PY) -m src.select_best

## Quick smoke promote
v2-promote-smoke:
	$(VENV_PY) -m src.select_best --epochs 2 \
		--max-train-samples 200 --max-val-samples 50 --max-test-samples 50

## Launch MLflow UI on port 8080
v2-mlflow-ui:
	$(VENV_PY) -m mlflow ui --backend-store-uri mlruns --port 8080

## Build & start the v2 stack (API:8000 MLflow:8080 Prometheus:8090 Grafana:8030)
v2-build:
	docker compose -f docker-compose.v2.yml build

v2-up:
	docker compose -f docker-compose.v2.yml up -d
	@echo "API:        http://localhost:8000/docs"
	@echo "Prometheus: http://localhost:8090"
	@echo "Grafana:    http://localhost:8030 (admin/admin)"
	@echo "(MLflow: run 'make v2-mlflow-ui' locally on port 8080)"

v2-down:
	docker compose -f docker-compose.v2.yml down

v2-logs:
	docker compose -f docker-compose.v2.yml logs -f api

## Post-deploy smoke test against the v2 API
v2-smoke-test:
	bash scripts/smoke_test.sh http://localhost:8000

