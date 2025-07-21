# syntax=docker/dockerfile:1
# DO NOT CHANGE THIS FILE.
FROM downloads.unstructured.io/unstructured-io/unstructured:latest

ENV UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Create a project venv and sync exactly to uv.lock (default behaviour) 
RUN python -m venv --system-site-packages .venv \
    && . .venv/bin/activate \
    && uv sync --locked --inexact

# Expose venv on PATH for *all* subsequent RUN/CMD layers
ENV PATH="/app/.venv/bin:${PATH}"

COPY src/  ./src/
COPY api/  ./api/
COPY tests ./tests/
RUN mkdir -p Data logs artifacts

CMD ["uvicorn", "api.parse_doc_api:app", "--host", "0.0.0.0", "--port", "3000"]
