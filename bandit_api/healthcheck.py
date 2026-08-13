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

import sys


def check_worker() -> int:
    """Pass only if a worker process is asserting liveness.

    Shares ``worker_is_live()`` with /readyz rather than re-deriving liveness
    from RQ's registry. The previous hostname-match-on-Worker.all() approach
    rested on the premise that crashed workers "drop out on their own", which is
    false: RQ pins a busy registration to job_timeout, so a worker killed
    mid-job stayed registered for 12 hours and this probe kept passing.
    """
    try:
        from .jobs import get_redis, worker_is_live

        get_redis().ping()
    except Exception as exc:
        print(f"redis unreachable: {exc}")
        return 1

    if not worker_is_live():
        # Expected while the model loads on first boot, which is why the compose
        # healthcheck gives this a long start_period.
        print("no worker heartbeat")
        return 1

    print("worker alive")
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
