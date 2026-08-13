"""Worker entrypoint.

Uses ``SimpleWorker`` rather than the default RQ worker on purpose. The default
forks a work horse per job, which would reload ~150 MB of weights and pay the
model-build cost on every single request. SimpleWorker runs jobs in-process, so
the module-level separator in ``jobs.py`` stays warm across jobs.

The trade-off is that a hard crash takes the worker down with it. That is what
the container restart policy is for.

Run with:  python -m bandit_api.worker
"""

from __future__ import annotations

import logging
import sys

from rq import Queue, SimpleWorker

from .config import settings
from .jobs import (
    QUEUE_NAME,
    get_redis,
    get_separator,
    reconcile_orphaned_jobs,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("bandit.worker")

    if not settings.ckpt_path.exists():
        log.error(
            "checkpoint missing at %s -- run scripts/fetch_weights.py",
            settings.ckpt_path,
        )
        return 1

    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    # Anything still flagged "running" belongs to a previous incarnation of
    # this worker and is never coming back. Settle those before accepting new
    # work, so clients polling them get an answer instead of hanging.
    orphaned = reconcile_orphaned_jobs()
    if orphaned:
        log.warning("failed %d job(s) orphaned by a previous worker restart",
                    orphaned)

    # Load before accepting work so the first request is not penalised and so
    # a broken checkpoint fails the container at boot rather than mid-job.
    get_separator()

    log.info("listening on queue %r", QUEUE_NAME)
    worker = SimpleWorker([Queue(QUEUE_NAME, connection=get_redis())],
                          connection=get_redis())
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
