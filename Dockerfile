# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install only the runtime dependencies (keeps the image lean for fast Cloud Run
# cold-starts). Dev/test/experiment tooling (mlflow, xgboost, ruff, ...) is not
# installed here.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application package + the trained model artifact (challenge/model.joblib).
# The notebook and the training script are excluded via .dockerignore.
COPY challenge/ ./challenge/

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run injects $PORT; bind uvicorn to it. Shell form so $PORT is expanded.
CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT}
