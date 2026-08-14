# Dev/test image for local reproducibility and CI parity.
# Superseded by a slim inference-only image in Phase 04 (serving).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --extra dev --extra training --extra serving --no-install-project

COPY . .
RUN uv sync --extra dev --extra training --extra serving

CMD ["uv", "run", "pytest"]
