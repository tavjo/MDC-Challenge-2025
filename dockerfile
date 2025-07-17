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

# Create necessary directories
RUN mkdir -p Data logs artifacts configs tests

# Set the default command to run the citation entity extractor
CMD ["python", "src/get_document_objects.py"]