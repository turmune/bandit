#!/bin/sh
# Delete job artifacts past their TTL. Three stereo stems per job run to ~100 MB;
# without this the volume fills within days.
set -e

DATA_DIR="${BANDIT_DATA_DIR:-/data}"
TTL_MINUTES="${BANDIT_REAP_AFTER_MINUTES:-1440}"
INTERVAL_SECONDS="${BANDIT_REAP_INTERVAL_SECONDS:-3600}"

while true; do
    # Liveness marker for the healthcheck; touched first so a slow sweep does
    # not look like a dead loop.
    touch /tmp/reaper-alive

    find "$DATA_DIR/outputs" -mindepth 1 -maxdepth 1 -type d \
        -mmin "+$TTL_MINUTES" -exec rm -rf {} + 2>/dev/null || true
    find "$DATA_DIR/inbox" -type f \
        -mmin "+$TTL_MINUTES" -delete 2>/dev/null || true

    sleep "$INTERVAL_SECONDS"
done
