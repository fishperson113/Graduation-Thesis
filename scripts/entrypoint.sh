#!/usr/bin/env bash
set -e

# Activate venv
source /app/.venv/bin/activate

# Re-install in editable mode if pyproject.toml changed
pip install --no-cache-dir -q -e ".[dev]"

exec "$@"
