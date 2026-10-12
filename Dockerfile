# AstraMed image-only API (research prototype; not for clinical use).
# The model checkpoint is NOT baked into the image: mount it read-only at run time.
# See docs/DEPLOYMENT.md for the exact commands.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (cached layer). CPU-only PyTorch keeps the image much smaller than the
# default Linux wheels, which bundle CUDA. Versions must match requirements-api.txt.
COPY requirements-api.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1 \
    && pip install -r requirements-api.txt

# Application code only (see .dockerignore: no data, outputs, checkpoints, notebooks or .env).
COPY src/ ./src/

# Unprivileged user; the application never writes to disk.
RUN useradd --system --uid 10001 --no-create-home astramed
USER astramed

ENV ASTRAMED_API_ENV=production \
    ASTRAMED_HOST=0.0.0.0 \
    ASTRAMED_PORT=8000 \
    ASTRAMED_LOG_LEVEL=INFO

EXPOSE 8000

# Liveness only (does not need the checkpoint). Use /api/v1/ready for readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status == 200 else 1)"]

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
