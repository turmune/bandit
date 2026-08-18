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

# Which torch build to install. "cpu" is the default so the CPU-only deployment
# keeps building exactly as before; a GPU worker host passes a CUDA tag instead:
#   docker build --build-arg TORCH_VARIANT=cu124 .
# No CUDA base image is needed -- the cu124 wheels bundle the CUDA runtime
# libraries themselves, so the host supplies only the driver.
ARG TORCH_VARIANT=cpu

# The wheel index matters in both directions: PyPI's default torch pulls ~2.5 GB
# of CUDA libraries a CPU-only host cannot use, and the CPU index yields a build
# that ignores an accelerator entirely.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    -r requirements.txt \
 # Every index carries torch 2.5.1, so resolving the wrong build is possible and
 # silent. Two distinct failures to catch: a CPU image carrying ~5 GB of
 # libraries it can never use, or a GPU image that quietly runs the model on the
 # CPU at a fraction of the speed. Assert whichever was actually asked for,
 # rather than discovering it from the image size or a benchmark.
 && TORCH_VARIANT="${TORCH_VARIANT}" python -c "import os, sys, torch; \
want_cuda = os.environ['TORCH_VARIANT'] != 'cpu'; \
sys.exit(0) if bool(torch.version.cuda) == want_cuda else sys.exit( \
    'ERROR: asked for the %s build but resolved torch %s (cuda=%s)' \
    % (os.environ['TORCH_VARIANT'], torch.__version__, torch.version.cuda))"

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
