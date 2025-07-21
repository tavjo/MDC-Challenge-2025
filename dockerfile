# Use the official Unstructured Docker image as base
FROM downloads.unstructured.io/unstructured-io/unstructured:latest

# # Set environment variables for uv
ENV UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONPATH=/app

# # Install uv (if not already available)
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock first (for better caching)
COPY pyproject.toml uv.lock ./

# Install additional Python dependencies using uv
# Note: unstructured[all] is already installed in the base image
RUN uv sync --frozen --no-dev
# RUN uv lock


# Copy the rest of the application
COPY src/ ./src/
COPY tests/ ./tests/
COPY api/ ./api/

# Create necessary directories
RUN mkdir -p Data logs tests artifacts

# Set the default command to run the citation entity extractor
CMD ["uvicorn", "api.parse_doc_api:app", "--host", "0.0.0.0", "--port", "3000"]