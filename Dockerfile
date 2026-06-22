# ── Stage 1: dependencies ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="CineMatch API"
LABEL org.opencontainers.image.description="Movie Recommendation System — FastAPI backend"
LABEL org.opencontainers.image.authors="Subin Thapa"

WORKDIR /app
ARG CACHEBUST=20260622095456

COPY --from=builder /install /usr/local

COPY src/        ./src/
COPY app/        ./app/
COPY frontend/   ./frontend/
COPY models/     ./models/
# data/ is mounted as a volume in docker-compose.
# Copy here only as fallback for standalone docker run.

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
