"""Container health probes.

Two reasons these exist beyond silencing Coolify's "no health check
configured" warning:

* A worker that died, or wedged partway through loading the model, would
  otherwise sit there holding the queue with nothing to show for it. A failing
  probe lets the restart policy do its job.
* Coolify and Docker report a container with no healthcheck as health
  "unknown", which is indistinguishable from "broken" at a glance.

Usage:  python -m bandit_api.healthcheck
"""

from __future__ import annotations

import os
import socket
import sys


def check_worker() -> int:
    """Pass only if *this container's* worker is asserting liveness.

    Matching on the worker's own name matters: a shared "any worker alive" check
    would let one container's probe pass because a different worker is healthy.
    """
    try:
        from .jobs import worker_is_live

        name = os.environ.get("RQ_WORKER_NAME") or socket.gethostname()
        alive = worker_is_live(name)
    except Exception as exc:
        print(f"cannot reach redis: {exc}")
        return 1

    if not alive:
        # Expected while the model loads on first boot, which is why the compose
        # healthcheck gives this a long start_period.
        print(f"no heartbeat from worker {name}")
        return 1

    print(f"worker {name} alive")
    return 0


def main(argv: list[str]) -> int:
    del argv  # single probe; the api container is probed with curl
    return check_worker()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
