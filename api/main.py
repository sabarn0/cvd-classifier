"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> multipart image upload -> predicted label + probabilities
    GET  /metrics  -> Prometheus metrics (scraped by Prometheus)

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference import load_model, predict, preprocess_image

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pt")
LOG_DIR = os.environ.get("LOG_DIR", "logs")

# --------------------------------------------------------------------------
# Logging (request/response logging, excluding raw image bytes)
# --------------------------------------------------------------------------
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("inference_api")
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(f"{LOG_DIR}/api.log", maxBytes=2_000_000, backupCount=3)
_handler.setFormatter(logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}'
))
logger.addHandler(_handler)
logger.addHandler(logging.StreamHandler())

# --------------------------------------------------------------------------
# Custom Prometheus metrics (in addition to the default HTTP metrics that
# prometheus-fastapi-instrumentator exposes automatically: request count,
# latency histograms, etc.)
# --------------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "prediction_requests_total", "Total number of prediction requests", ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent running model inference"
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Total number of failed prediction requests"
)

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")

# Wires up /metrics with default HTTP request count/latency metrics.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_model = None


@app.on_event("startup")
def _startup():
    global _model
    if Path(MODEL_PATH).exists():
        _model = load_model(MODEL_PATH)
        logger.info(f'"Model loaded from {MODEL_PATH}"')
    else:
        logger.info(f'"Model file not found at {MODEL_PATH}; /predict will 503 until it exists"')


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()

    start = time.time()
    try:
        input_tensor = preprocess_image(image_bytes)
        with PREDICTION_LATENCY.time():
            label, probabilities = predict(_model, input_tensor)
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.inc()
        logger.info(f'"Prediction failed for {file.filename}: {exc}"')
        raise HTTPException(status_code=400, detail="Could not process image.") from exc

    latency_ms = round((time.time() - start) * 1000, 2)
    PREDICTION_COUNTER.labels(predicted_class=label).inc()

    # Log request/response metadata only -- never the raw image bytes.
    logger.info(
        f'"filename={file.filename} size_bytes={len(image_bytes)} '
        f'label={label} latency_ms={latency_ms}"'
    )

    return {
        "label": label,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
    }
