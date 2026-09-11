# DepthWizard - single-view height estimation and 3D flythrough (ISRO PS 26175)
#
# Runs the full pipeline and viewer with nothing installed on the host but Docker.
#
#   docker build -t depthwizard .
#   docker run --rm -p 8000:8000 depthwizard
#   -> http://localhost:8000
#
# Notes on the choices here:
#   * Python 3.12, not 3.13 - rasterio and friends ship manylinux wheels with GDAL
#     bundled for 3.12, so no system GDAL install is needed and the image stays small.
#   * CPU build of PyTorch. The CUDA wheels add ~2.5 GB and GPU passthrough needs
#     host-specific setup; inference here is a few seconds on CPU.
#   * Model weights are baked in at build time. Downloading them on first request
#     would make the first upload slow and the container dependent on network
#     access - neither is acceptable when this is being demonstrated live.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    DW_TILED=off \
    DW_PREWARM=true \
    DW_MESH_DIM=1024

WORKDIR /app

# curl is only needed for the container healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Torch first, from the CPU index, so the dependency resolver never reaches for
# the CUDA build when transformers pulls torch in.
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the Depth Anything V2 weights (~100 MB) into the image so the container
# starts fast and works with no network.
RUN python -c "from transformers import pipeline; \
pipeline(task='depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=-1)"

# Application code. tests/ is included because the startup path falls back to its
# synthetic terrain generator when no sample image is present.
COPY api/ ./api/
COPY src/ ./src/
COPY web/ ./web/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY data/input/ ./data/input/
COPY docs/ ./docs/
COPY README.md HANDOFF.md ./

# Ship the measured benchmark so the validation panel is populated on first load.
COPY data/benchmark.json ./data/benchmark.json

RUN mkdir -p data/web_jobs data/output data/srtm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/benchmark >/dev/null || exit 1

# 0.0.0.0, not 127.0.0.1 - binding to loopback inside the container makes the
# published port unreachable from the host.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
