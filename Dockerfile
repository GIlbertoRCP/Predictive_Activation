FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so code changes don't bust the layer cache.
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

COPY models ./models
COPY data ./data
COPY benchmarks ./benchmarks

ENV PA_ROOT=/app
EXPOSE 8000 8501

# Start the dashboard by default (docker-compose overrides per service)
CMD ["streamlit", "run", "src/predictive_activation/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
