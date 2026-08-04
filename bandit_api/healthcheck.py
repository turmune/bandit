"""Container health probes.

Two reasons these exist beyond silencing Coolify's "no health check
configured" warning:

* A worker that died, or wedged partway through loading the model, would
  otherwise sit there holding the queue with nothing to show for it. A failing
  probe lets the restart policy do its job.
* Coolify and Docker report a container with no healthcheck as health
  "unknown", which is indistinguishable from "broken" at a glance.

Usage:  python -m bandit_api.healthcheck {worker|api}
"""

from __future__ import annotations

import socket
import sys


def check_worker() -> int:
    """Pass only if an RQ worker for *this container* is alive in Redis.

    RQ workers refresh a heartbeat key with a TTL, and ``Worker.all()`` only
    returns those whose key has not expired -- so a crashed or hung process
    drops out on its own. Matching on hostname keeps one container's probe from
    passing because a *different* worker is healthy.
    """
    try:
        from rq import Worker

        from .jobs import get_redis

        redis = get_redis()
        redis.ping()
    except Exception as exc:
        print(f"redis unreachable: {exc}")
        return 1

    me = socket.gethostname()
    try:
        mine = [w for w in Worker.all(connection=redis) if w.hostname == me]
    except Exception as exc:
        print(f"could not enumerate workers: {exc}")
        return 1

    if not mine:
        # Expected while the model loads on first boot, which is why the
        # compose healthcheck gives this a long start_period.
        print(f"no live rq worker registered for host {me}")
        return 1

    print(f"worker {mine[0].name} alive (state={mine[0].get_state()})")
    return 0


def check_api() -> int:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:8000/healthz", timeout=5) as r:
            if r.status == 200:
                print("api ok")
                return 0
            print(f"api returned {r.status}")
            return 1
    except Exception as exc:
        print(f"api unreachable: {exc}")
        return 1


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "api"
    if target == "worker":
        return check_worker()
    if target == "api":
        return check_api()
    print(f"unknown probe target {target!r}; expected 'worker' or 'api'")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
