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
import os
import socket
import sys

from rq import SimpleWorker, Worker

from .config import settings
from .jobs import (
    QUEUE_NAME,
    get_queue,
    get_redis,
    get_separator,
    reconcile_orphaned_jobs,
    start_liveness_heartbeat,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("bandit.worker")

    # Bury registrations from previous incarnations FIRST. Only one worker runs
    # at a time, so anything registered that is not us is dead. This has to
    # precede the checkpoint check and the model load: a worker crash-looping on
    # a bad checkpoint or an OOM would otherwise never reach it, and /readyz
    # would report phantom capacity for the 12h life of the stale registration.
    queue = get_queue()
    worker = SimpleWorker([queue], connection=get_redis(),
                          name=os.environ.setdefault("RQ_WORKER_NAME",
                                                     socket.gethostname()))
    for other in Worker.all(queue=queue):
        if other.name != worker.name:
            log.warning("burying registration from dead worker %s", other.name)
            other.register_death()

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

    # Assert liveness for the rest of this process's life. Started after the
    # model loads so /readyz stays honest during the first-boot weight download.
    start_liveness_heartbeat(worker.name)

    log.info("listening on queue %r", QUEUE_NAME)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
