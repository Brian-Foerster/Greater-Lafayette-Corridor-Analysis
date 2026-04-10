# =============================================================================
# UrbanSim APM Corridor Evaluation — Docker image
# =============================================================================
# Build:   docker build -t urbansim-apm .
# Run:     docker run --rm -v ./data/processed:/app/data/processed urbansim-apm \
#              python scripts/run_feedback_loop.py --scenario current_zoning
# Test:    docker run --rm urbansim-apm python -m pytest tests/ -x -q -m "not slow"
# Serve:   docker run --rm -p 8080:8080 urbansim-apm \
#              python scripts/run_feedback_loop.py --all-scenarios --serve
# =============================================================================

FROM python:3.11-slim AS base

# System dependencies for geospatial stack (GEOS, PROJ, GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
        libproj-dev \
        libgdal-dev \
        gdal-bin \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pandana

# Copy project source
COPY src/ src/
COPY scripts/ scripts/
COPY urbansim/ urbansim/
COPY conftest.py scenarios_config.json ./
COPY tests/ tests/

# Copy tracked raw data
COPY data/raw/ data/raw/

# Create output directories
RUN mkdir -p data/processed cache

# Ensure src is importable
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run the fast test suite
CMD ["python", "-m", "pytest", "tests/", "-x", "-q", "-m", "not slow"]
