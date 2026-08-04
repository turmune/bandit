#!/bin/sh
# Worker entrypoint: ensure weights are present, then hand off to the worker.
#
# This lives in a file rather than a compose `command:` because YAML folded
# scalars (`>`) preserve newlines on lines indented deeper than the first,
# which silently turned `&& python -m bandit_api.worker` into its own line and
# broke the worker with `sh: Syntax error: "&&" unexpected`.
set -e

MODELS_DIR="$(dirname "${BANDIT_CKPT_PATH:-/data/models/checkpoint-multi-inference.pt}")"
VARIANT="${BANDIT_VARIANT:-multi}"

python scripts/fetch_weights.py --variant "$VARIANT" --dest "$MODELS_DIR" --convert

# exec so the worker becomes PID 1 and receives SIGTERM directly on shutdown,
# rather than sh swallowing it and forcing a 10s kill timeout.
exec python -m bandit_api.worker
