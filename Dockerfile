FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
# Trained model artifact -- normally produced by `python -m src.train`.
# When using docker-compose, ./models is also volume-mounted so you can
# retrain without rebuilding the image.
COPY models/ ./models/

ENV MODEL_PATH=models/model.pt
ENV LOG_DIR=logs
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
