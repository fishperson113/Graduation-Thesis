FROM python:3.10-slim

WORKDIR /app

# System deps for building numpy/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv inside the container
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install deps first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Source code mounted at runtime via docker-compose volume
COPY . .

# Re-install in editable mode with source present
RUN pip install --no-cache-dir -e ".[dev]"
