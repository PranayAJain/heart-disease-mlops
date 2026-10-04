# Heart-disease risk API - production image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) dependencies first (cached layer)
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) application code + packaged model
COPY src/ src/
COPY models/heart_model.joblib models/model_metadata.json models/
ENV PYTHONPATH=/app/src

# 3) run as non-root
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "heart.api:app", "--host", "0.0.0.0", "--port", "8000"]
