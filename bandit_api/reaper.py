"""Periodic janitor: expire artifacts, and settle jobs nobody is polling.

Was a shell script, which restated the retention window as a literal in minutes
and so drifted the moment ``save_job`` started giving unsettled jobs a longer
TTL. Reading ``settings`` keeps one definition of the policy.

It also sweeps stale jobs. ``settle_if_stale`` only runs when a client polls, so
a job submitted with a ``callback_url`` and never polled would sit at "running"
forever after its worker died -- the caller waiting on a webhook that no code
path would ever fire. Sweeping here makes recovery independent of who is
watching.

Run with:  python -m bandit_api.reaper
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

from .config import settings
from .jobs import Job, JobStatus, _key, get_redis, settle_if_stale

log = logging.getLogger("bandit.reaper")

LIVENESS_FILE = Path("/tmp/reaper-alive")


def sweep_stale_jobs() -> int:
    """Settle RUNNING jobs whose worker is gone, regardless of who is polling."""
    redis = get_redis()
    settled = 0
    for key in redis.scan_iter(match=_key("*"), count=100):
        raw = redis.get(key)
        if not raw:
            continue
        try:
            job = Job.from_json(raw)
        except Exception:
            continue
        if job.status is not JobStatus.RUNNING:
            continue
        if settle_if_stale(job).status is JobStatus.FAILED:
            settled += 1
    return settled


def sweep_artifacts() -> int:
    """Delete outputs and orphaned uploads past the retention window."""
    cutoff = time.time() - settings.result_ttl_seconds
    removed = 0

    for path in (settings.outputs_dir, settings.inbox_dir):
        if not path.exists():
            continue
        for entry in path.iterdir():
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                removed += 1
            except OSError as exc:
                log.warning("could not remove %s: %s", entry, exc)
    return removed


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    interval = max(60, settings.result_ttl_seconds // 24)
    log.info("sweeping every %ds, retention %ds", interval, settings.result_ttl_seconds)

    while True:
        # Touched first so a slow sweep does not look like a dead loop to the
        # container healthcheck.
        LIVENESS_FILE.touch()
        try:
            stale = sweep_stale_jobs()
            removed = sweep_artifacts()
            if stale or removed:
                log.info("settled %d stale job(s), removed %d artifact(s)",
                         stale, removed)
        except Exception as exc:
            log.warning("sweep failed, will retry: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
