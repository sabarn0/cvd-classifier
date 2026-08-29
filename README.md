# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification (cats vs dogs) for a pet-adoption platform, built as
a local-first MLOps pipeline: data prep → training + MLflow tracking → FastAPI
inference service → Docker → Prometheus/Grafana monitoring. GitHub Actions
CI/CD is scaffolded and ready to switch on when you push this to GitHub.

## 1. Project layout

```
data/prepare_data.py     Downloads (kagglehub) + preprocesses + splits the dataset
src/dataset.py            PyTorch Dataset + train/eval transforms (augmentation)
src/model.py               SimpleCNN architecture
src/train.py                Training loop + MLflow logging (params/metrics/artifacts/model)
src/inference.py            preprocess_image() + predict() — shared by API and tests
api/main.py                 FastAPI app: /health, /predict, /metrics
tests/                        pytest unit tests for preprocessing + inference
monitoring/                 prometheus.yml + Grafana datasource/dashboard provisioning
Dockerfile                  Container for the inference service
docker-compose.yml           Full local stack: api + mlflow + prometheus + grafana
Makefile                      One-word commands for every step below
scripts/smoke_test.sh        Post-deploy health + prediction check
.github/workflows/ci-cd.yml  CI/CD skeleton for the GitHub phase (currently manual-trigger only)
```

## 2. Prerequisites

- Python 3.11
- Docker + Docker Compose
- A Kaggle account with the Kaggle API configured for `kagglehub`
  (`~/.kaggle/kaggle.json`, or run `kagglehub` login flow) — needed once, for `make data`.

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
make install
```

## 4. M1 — Data & Model Development

```bash
# Downloads via kagglehub, resizes to 224x224, splits 80/10/10 into
# data/processed/{train,val,test}/{cat,dog}/
make data

# Or, for a fast local dry run without the full dataset:
make data-smoke

# Train the baseline CNN, tracked in MLflow (params, per-epoch metrics,
# confusion matrix, loss/accuracy curves, and the model artifact)
make train
# quick smoke version:
make train-smoke

# View experiments
make mlflow-ui       # http://localhost:5000
```

This writes:
- `models/model.pt` — state_dict used directly by the FastAPI service
- `models/model_config.json` — class mapping + test metrics
- an MLflow run under the `cats-vs-dogs` experiment with the logged model, params,
  metrics, confusion matrix and loss curves as artifacts

**Data & code versioning (Git + DVC):**
```bash
git init
git add .
git commit -m "Initial MLOps pipeline"

pip install dvc
dvc init
dvc add data/processed
git add data/processed.dvc .gitignore
dvc remote add -d localremote /path/to/some/local/or/cloud/storage
dvc push
```

## 5. M2 — Packaging & Containerization

Run the API directly (no Docker) to iterate quickly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 9000 --reload
```

Test it:
```bash
curl http://localhost:9000/health

curl -X POST http://localhost:9000/predict \
  -F "file=@/path/to/some/cat_or_dog.jpg;type=image/jpeg"
```

Build & run in Docker:
```bash
docker build -t cats-dogs-api .
docker run -p 9000:9000 -v $(pwd)/models:/app/models:ro cats-dogs-api
```

## 6. M3 — CI Pipeline (tests, build)

```bash
make test        # pytest — preprocessing + inference unit tests
```

`.github/workflows/ci-cd.yml` defines the checkout → install →
`pytest` → `docker build` → push to Docker Hub steps.

## 7. M4 — "CD" & Deployment (local-first & GCP VM)

For local stack:

```bash
make build
make up
```

This starts:
| Service    | URL                          |
|------------|-------------------------------|
| API        | http://localhost:9000/docs   |
| MLflow     | http://localhost:5000        |
| Prometheus | http://localhost:9090        |
| Grafana    | http://localhost:9030 (admin/admin) |

Post-deploy smoke test (health + one prediction call, fails the pipeline on error):
```bash
make smoke-test
```

Stop everything:
```bash
make down
```

**Next phase (GitHub Actions):** the commented-out `deploy` job in
`ci-cd.yml` shows where to plug in `kubectl apply` (kind/minikube/microk8s)
or an SSH/`docker compose up` step against a VM, followed by
`scripts/smoke_test.sh` against the deployed host — same script used here.

## 8. M5 — Monitoring & Logs

- **Request/response logging**: `api/main.py` logs filename, size, predicted
  label and latency (never raw image bytes) to `logs/api.log` (rotating) and stdout.
- **Metrics**: `prometheus-fastapi-instrumentator` exposes default HTTP metrics
  (`http_requests_total`, `http_request_duration_seconds_bucket`, ...) at `/metrics`,
  plus custom metrics defined in `api/main.py`:
  - `prediction_requests_total{predicted_class}`
  - `prediction_latency_seconds`
  - `prediction_errors_total`
- **Grafana dashboard** "Cats vs Dogs Inference API" is auto-provisioned with
  panels for request rate, HTTP p95 latency, predictions by class, inference
  latency (p50/p95), and error rate.
- **Model performance tracking (post-deployment)**: send a batch of labeled
  requests through `/predict` and compare returned labels against your ground
  truth (e.g. a small script reusing `scripts/smoke_test.sh`'s pattern, looping
  over a held-out folder and logging accuracy) — a good place to log this
  as a new MLflow run for drift comparisons.

## 9. Deliverables checklist

- [x] Git repo (init locally, push to GitHub for submission)
- [x] DVC config for `data/processed` (see §4)
- [x] Source code: data prep, training, API, tests
- [x] `requirements.txt` (pinned)
- [x] `Dockerfile`, `docker-compose.yml`
- [x] CI/CD workflow file (`.github/workflows/ci-cd.yml`)
- [x] Monitoring config (`monitoring/`)
- [x] Trained model artifacts (`models/model.pt`, `models/model_config.json`) — generate via `make train`
- [ ] Screen recording (<5 min): code change → CI run → image build → deploy → `/predict` call → Grafana dashboard
