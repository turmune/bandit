# Single image, two roles (api / worker) selected by the compose command.
# One image means one build and one layer cache instead of two nearly identical ones.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg gives torchaudio/soundfile broad container support (mp4, mkv, mp3, m4a).
# libsndfile1 backs soundfile's incremental writer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU wheel index matters: the default PyPI torch pulls ~2.5 GB of CUDA
# libraries that cannot be used on a CPU-only host.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
 # Both indexes carry torch 2.5.1, so resolution to the CUDA build is possible.
 # That silently adds ~5 GB of libraries this host can never use, so assert the
 # CPU build was chosen rather than discovering it from the image size.
 && python -c "import torch, sys; \
sys.exit(0) if torch.version.cuda is None else sys.exit( \
    'ERROR: resolved the CUDA build of torch (%s); expected +cpu' % torch.__version__)"

COPY bandit_api/ ./bandit_api/
COPY scripts/ ./scripts/

# Weights are NOT baked in: 447 MB per variant would bloat every rebuild and
# Zenodo is slow. They live on the shared volume, fetched once by the worker's
# entrypoint. See docker-compose.yaml.
ENV BANDIT_DATA_DIR=/data \
    BANDIT_CKPT_PATH=/data/models/checkpoint-multi-inference.pt

RUN useradd --create-home --uid 10001 bandit && mkdir -p /data && chown bandit /data
USER bandit

EXPOSE 8000
CMD ["uvicorn", "bandit_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
