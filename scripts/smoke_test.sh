#!/usr/bin/env bash
# Post-deploy smoke test: calls /health and /predict, fails (non-zero exit)
# if either check does not succeed. Intended for CI/CD (M4) as well as
# manual local verification.
set -euo pipefail

HOST="${1:-http://localhost:9000}"
TEST_IMAGE="${2:-tests/fixtures/sample.jpg}"

echo "== Smoke test against $HOST =="

echo "-- Health check --"
HEALTH_RESPONSE=$(curl -s -o /tmp/health.json -w "%{http_code}" "$HOST/health")
cat /tmp/health.json
if [ "$HEALTH_RESPONSE" != "200" ]; then
  echo "Health check FAILED (HTTP $HEALTH_RESPONSE)"
  exit 1
fi
MODEL_LOADED=$(python3 -c "import json;print(json.load(open('/tmp/health.json'))['model_loaded'])")
if [ "$MODEL_LOADED" != "True" ]; then
  echo "Health check FAILED: model_loaded=$MODEL_LOADED"
  exit 1
fi
echo "Health check OK"

echo "-- Prediction check --"
if [ ! -f "$TEST_IMAGE" ]; then
  echo "Test image $TEST_IMAGE not found, generating a throwaway one..."
  python3 -c "from PIL import Image; Image.new('RGB', (224,224), (120,80,40)).save('/tmp/smoke.jpg')"
  TEST_IMAGE="/tmp/smoke.jpg"
fi

PREDICT_RESPONSE=$(curl -s -o /tmp/predict.json -w "%{http_code}" -X POST \
  -F "file=@${TEST_IMAGE};type=image/jpeg" "$HOST/predict")
cat /tmp/predict.json
if [ "$PREDICT_RESPONSE" != "200" ]; then
  echo "Prediction check FAILED (HTTP $PREDICT_RESPONSE)"
  exit 1
fi
echo "Prediction check OK"

echo "== Smoke test PASSED =="
