# Cats vs Dogs — End-to-End MLOps Pipeline (v2) [https://github.com/sabarn0/cvd-classifier]

Binary image classification (cats vs. dogs) for a pet-adoption platform, engineered as an automated, production-grade MLOps pipeline:
**Data Prep & DVC Versioning** $\rightarrow$ **Experiment Tracking & Model Selection (MLflow)** $\rightarrow$ **FastAPI Microservice** $\rightarrow$ **Containerization (Docker)** $\rightarrow$ **Automated CI/CD (GitHub Actions)** $\rightarrow$ **GCP VM Deployment** $\rightarrow$ **Full-Stack Monitoring (Prometheus & Grafana)**.

---

## 📹 Video Demo
🎥 **[Watch the Complete Pipeline Demo Recording on Google Drive](https://drive.google.com/drive/folders/1yVHcCq1ILCeC7rIjcJw4Drp9Q3CnG90b?usp=sharing)**
*(Demonstrates code push $\rightarrow$ GitHub Actions CI/CD $\rightarrow$ GCP VM auto-deployment $\rightarrow$ live inference $\rightarrow$ real-time Grafana dashboard monitoring).*

---

## 1. Architecture & Port Allocation

All services in the v2 stack run on the **`70XX`** port range:

| Service | Port | Endpoint / Description |
| :--- | :--- | :--- |
| **FastAPI Inference Service** | `7000` | `http://localhost:7000/docs` (Swagger UI, `/predict`, `/health`, `/model-info`, `/metrics`) |
| **Grafana Dashboard** | `7030` | `http://localhost:7030` (`admin`/`admin` — pre-provisioned *Cats vs Dogs* dashboard) |
| **Prometheus Monitoring** | `7090` | `http://localhost:7090` (Scrapes `v2-api:7000/metrics`) |
| **MLflow Tracking UI** | `7080` | `http://localhost:7080` (Interactive dashboard for `cvd-classifier` experiment) |

---

## 2. Project Layout

```text
├── Dockerfile.v2                                     # Multi-stage Docker build with pre-cached weights
├── docker-compose.v2.yml                             # Local & VM stack: API + Prometheus + Grafana (70XX ports)
├── requirements.txt                                  # Pinned production dependencies (PyTorch 2.3.1, Optuna, MLflow)
├── Makefile                                          # One-word CLI automation commands
├── .github/workflows/
│   └── ci-cd-v2.yml                                 # Automated CI/CD pipeline (Test -> Build -> DockerHub -> GCP VM Deploy)
├── api/
│   └── main.py                                       # FastAPI service: /health, /model-info, /predict, /metrics
├── src/
│   ├── dataset.py                                    # PyTorch Dataset + ImageNet / [-1,1] transforms
│   ├── model.py                                      # Architectures: SimpleCNN (baseline) & MobileNetV3Small (champion)
│   ├── train.py                                      # MLflow-instrumented training loop
│   ├── tune.py                                       # Optuna Bayesian hyperparameter optimization
│   ├── select_best.py                                # Automated model comparison & champion promotion
│   └── inference.py                                  # Preprocessing, forward pass, and state_dict loader
├── monitoring/
│   ├── prometheus.v2.yml                             # Prometheus scrape config (targets: v2-api:7000)
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/api-dashboard.json         # Zero-scroll, single-window dashboard with stat headline cards
│           └── datasources/datasource.yml            # Auto-provisioned Prometheus datasource
├── scripts/
│   ├── smoke_test.sh                                 # Zero-dependency post-deployment health & prediction validation
│   └── test_batch_endpoint.py                        # Automated 25-image batch prediction & ground-truth validation tool
├── tests/
│   ├── test_inference.py                             # PyTest suite for model predictions
│   ├── test_preprocessing.py                         # PyTest suite for tensor transforms
│   ├── test_live_endpoint.py                         # PyTest suite for batch endpoint validation (>= 80% accuracy)
│   └── fixtures/sample.jpg                           # Sample test fixture image
├── models/
│   ├── model.pt                                      # CPU-compatible champion model weights (MobileNetV3Small)
│   └── model_config.json                             # Promoted model metadata and official test metrics
├── data/
│   └── prepare_data.py                               # Kagglehub download + 80/10/10 split (cat/dog)
└── misc/util/v1/                                     # Archived legacy v1 artifacts (gitignored)
```

---

## 3. Quickstart & Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Kaggle API token configured (`~/.kaggle/kaggle.json` or `kagglehub` auth)

```bash
# 1. Clone & create virtual environment
git clone <repo-url>
cd mlops-assignment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download & preprocess dataset (80/10/10 train/val/test split)
python data/prepare_data.py
```

---

## 4. M1 — Model Training & MLflow Experiment Tracking

The project evaluates both a custom **SimpleCNN** baseline and transfer-learning with **MobileNetV3-Small** under the **`cvd-classifier`** MLflow experiment:

```bash
# Train Model 1 (SimpleCNN Baseline)
python -m src.train --model-name SimpleCNN --run-name model-1-ep0 --epochs 5 --batch-size 64

# Train Model 2 (MobileNetV3Small Champion)
python -m src.train --model-name MobileNetV3Small --run-name model-2-ep0 --epochs 5 --batch-size 64

# (Optional) Run Optuna HPO study
python -m src.tune --n-trials 20 --epochs-per-trial 3

# View interactive MLflow dashboard on port 7080
make v2-mlflow-ui
```

### **Model Comparison Results (`cvd-classifier`)**

| Run Name | Architecture | `test_acc` | `test_f1` | `test_precision` | `test_recall` | `val_acc` | `val_loss` | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`model-2-ep0`** | **MobileNetV3Small** | **96.00%** | **0.9592** | **0.9792** | **0.9400** | **95.76%** | **0.1015** | 🏆 **Promoted Champion** |
| **`model-1-ep0`** | SimpleCNN | 86.05% | 0.8524 | 0.9048 | 0.8058 | 83.87% | 0.3736 | Baseline |

- **Champion weights**: Saved at `models/model.pt` (CPU state_dict for edge/container inference).
- **Artifacts**: Confusion matrix (`artifacts/confusion_matrix.png`) & Loss/Accuracy curves (`artifacts/loss_accuracy_curves.png`).

---

## 5. M2 & M3 — Testing, Packaging & Containerization

### Running Tests
```bash
# Run full PyTest suite (7 unit & integration tests)
pytest -v

# Run automated batch endpoint validation (25 random unseen images)
python scripts/test_batch_endpoint.py --host http://localhost:7000 --n-samples 25
```

### Local Docker Stack
```bash
# Build & start full v2 stack (API:7000, Prometheus:7090, Grafana:7030)
make v2-build
make v2-up

# Test service health
curl http://localhost:7000/health

# Run post-deployment smoke test
make v2-smoke-test

# Stop containers
make v2-down
```

---

## 6. M4 — CI/CD Pipeline (GitHub Actions $\rightarrow$ GCP VM)

The automated pipeline in [`.github/workflows/ci-cd-v2.yml`](.github/workflows/ci-cd-v2.yml) triggers on every push to **`v2-dev`**:

1. **Continuous Integration (`test-and-build`)**:
   - Sets up Python 3.11 with pip dependency caching.
   - Runs the `pytest` test suite.
   - Builds `Dockerfile.v2` (pre-caching PyTorch weights).
   - Pushes multi-tag image `cats-dogs-api-v2:latest` to Docker Hub.
2. **Continuous Deployment (`deploy`)**:
   - Securely copies deployment configs to `~/app-v2` on the GCP VM via SCP.
   - Automatically runs `docker system prune` to prevent disk space exhaustion.
   - Pulls the latest container and spins up the stack with Docker Compose.
   - Runs on-host zero-dependency smoke test (`scripts/smoke_test.sh http://localhost:7000`).

---

## 7. M5 — Monitoring & Observability

- **API Metrics Exporter**: Custom Prometheus counters & latency histograms instrumented in `api/main.py` at `/metrics`:
  - `prediction_requests_total{predicted_class="cat|dog"}`
  - `prediction_latency_seconds_bucket`
  - `prediction_errors_total`
- **Request Logging**: Structured JSON request/response metadata (latency, file size, prediction) logged to `logs/api.log` (excluding raw image bytes).
- **Single-Window Grafana Dashboard (`http://<HOST>:7030`)**:
  - **Top Row (Stat Cards)**: Total Predictions, Request Rate (req/s), p95 Inference Latency (ms), and Prediction Error Rate.
  - **Middle Row**: HTTP Request Rate & HTTP p95 Latency by endpoint (with noise like `/openapi.json` filtered out).
  - **Bottom Row**: Class distribution breakdown (`cat` vs `dog`) and median vs. 95th percentile inference latency (`p50` vs `p95`).

---

## 8. Deliverables Checklist

- [x] **Git Repository & DVC Data Management**: Versioned dataset under `data/processed/` with `.gitignore` policies.
- [x] **Model Training & Comparison**: Experiment tracking in MLflow with 96.00% accuracy champion model.
- [x] **Containerized Inference Service**: FastAPI app with CPU-optimized PyTorch execution on port `7000`.
- [x] **Comprehensive Testing Suite**: Unit, preprocessing, and live batch validation tests in `tests/`.
- [x] **Automated CI/CD Pipeline**: GitHub Actions workflow deploying to GCP Compute Engine VM.
- [x] **Production Monitoring Stack**: Prometheus scraping and single-window Grafana dashboard on port `7030`.
- [x] **Video Demo Recording**: [Google Drive Demo Recording](https://drive.google.com/drive/folders/1yVHcCq1ILCeC7rIjcJw4Drp9Q3CnG90b?usp=sharing).
