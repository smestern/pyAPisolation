# Multi-stage Dockerfile for the gigaseal analysis web app.
#
# Build:   docker build -t gigaseal-web .
# Public:  docker run -p 8000:8000 -e GIGASEAL_WEB_PROFILE=public gigaseal-web
# Lab:     docker run -p 8000:8000 -e GIGASEAL_WEB_PROFILE=lab \
#              -e GIGASEAL_API_TOKEN=changeme \
#              -v /data:/data:ro -e GIGASEAL_SERVER_PATH_ROOT=/data \
#              gigaseal-web

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIGASEAL_WEB_PROFILE=lab \
    GIGASEAL_SESSION_DIR=/var/lib/gigaseal/sessions

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential libhdf5-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY gigaseal/ ./gigaseal/
COPY data/ ./data/

RUN pip install --upgrade pip \
 && pip install ".[server]" "flask-limiter>=3.0" "pyarrow>=12.0"

RUN mkdir -p /var/lib/gigaseal/sessions \
 && useradd -m -u 1000 gigaseal \
 && chown -R gigaseal:gigaseal /var/lib/gigaseal

USER gigaseal

EXPOSE 8000

# 2 workers keeps free-tier RAM happy; --max-requests recycles them
# defensively. The job queue inside the app has its own concurrency cap.
CMD ["gunicorn", \
     "-w", "2", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--max-requests", "50", \
     "--max-requests-jitter", "10", \
     "gigaseal.webViz.analysis_web.wsgi:app"]
