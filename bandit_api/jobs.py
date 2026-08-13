"""Job state and the worker-side separation task.

Job records live in Redis as JSON with a TTL. The queue is RQ. The API process
never imports torch -- only this module's ``run_separation`` does, and only
inside the worker.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import batched
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import redis
from rq import Queue

from .config import settings

log = logging.getLogger(__name__)

QUEUE_NAME = "separation"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    quality: str = "balanced"
    stems: list[str] = field(default_factory=list)
    output_format: str = "wav"
    source_name: str | None = None
    duration_seconds: float | None = None
    progress: float = 0.0
    error: str | None = None
    callback_url: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # Set on every save_job(); the basis for how long a job has been silent.
    updated_at: float = field(default_factory=time.time)

    @property
    def silent_for(self) -> float:
        """Seconds since anything last wrote this record."""
        return time.time() - self.updated_at

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Job":
        d = json.loads(raw)
        d["status"] = JobStatus(d["status"])
        return cls(**d)


_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url)
    return _redis


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def _key(job_id: str) -> str:
    return f"bandit:job:{job_id}"


def save_job(job: Job) -> None:
    """Persist a job. Terminal records get the result TTL; unsettled ones get
    that plus the job timeout, so a job cannot expire while still queued or
    running behind a backlog.
    """
    job.updated_at = time.time()
    ttl = settings.result_ttl_seconds
    if not job.is_terminal:
        ttl += settings.job_timeout_seconds
    get_redis().set(_key(job.id), job.to_json(), ex=ttl)


def load_job(job_id: str) -> Job | None:
    raw = get_redis().get(_key(job_id))
    return Job.from_json(raw) if raw else None


def job_dir(job_id: str) -> Path:
    return settings.outputs_dir / job_id


def _cancel_key(job_id: str) -> str:
    return f"bandit:cancel:{job_id}"


def _alive_key(worker: str) -> str:
    return f"bandit:worker:{worker}:alive"


def live_workers() -> list[str]:
    """Names of worker processes currently asserting liveness.

    Each worker refreshes its own short-TTL key, so a key outlives its process
    by at most ``worker_alive_ttl_seconds``. Deliberately not derived from RQ's
    registry, which pins a busy worker's entry to the job timeout and so reports
    a killed worker as alive for hours.
    """
    keys = get_redis().keys(_alive_key("*"))
    prefix, suffix = len(b"bandit:worker:"), len(b":alive")
    return [k[prefix:-suffix].decode() for k in keys]


def worker_is_live(name: str | None = None) -> bool:
    """Is a worker alive -- this specific one, or any at all?"""
    if name is not None:
        return bool(get_redis().exists(_alive_key(name)))
    return bool(live_workers())


def start_liveness_heartbeat(name: str) -> None:
    """Assert this process's liveness until it dies.

    A daemon thread rather than a call inside the work loop, so the beat
    continues through model loading, a blocking ffmpeg decode and idle waits
    alike -- there is no phase in which a healthy worker looks dead.
    """
    ttl = settings.worker_alive_ttl_seconds
    key = _alive_key(name)

    def beat() -> None:
        while True:
            try:
                get_redis().set(key, "1", ex=ttl)
            except Exception as exc:
                log.warning("liveness heartbeat failed: %s", exc)
            time.sleep(max(5, ttl // 3))

    threading.Thread(target=beat, name="liveness", daemon=True).start()


def _cancel_rq_job(job_id: str, *, delete: bool = False) -> None:
    """Drop a job from RQ. Missing/never-registered jobs are not an error."""
    try:
        from rq.job import Job as RQJob

        rq_job = RQJob.fetch(job_id, connection=get_redis())
        rq_job.cancel()
        if delete:
            rq_job.delete()
    except Exception as exc:
        log.debug("no RQ job to cancel for %s (%s)", job_id, exc)


def request_cancel(job_id: str) -> None:
    """Tombstone a job so a worker already running it stops and stays deleted.

    SimpleWorker runs jobs in-process, so there is no work horse to signal. The
    running job polls this from its progress callback instead. Without it, DELETE
    returned 204, the worker kept going, and the record came back as "succeeded"
    with every stem 404ing because the files had been removed underneath it.
    """
    get_redis().set(_cancel_key(job_id), "1", ex=settings.job_timeout_seconds)


def is_cancelled(job_id: str) -> bool:
    return bool(get_redis().exists(_cancel_key(job_id)))


class JobCancelled(Exception):
    """Raised inside a running job when its record has been deleted."""


def delete_job(job_id: str) -> bool:
    request_cancel(job_id)
    _cancel_rq_job(job_id)
    existed = bool(get_redis().delete(_key(job_id)))
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    for leftover in settings.inbox_dir.glob(f"{job_id}.*"):
        leftover.unlink(missing_ok=True)
    return existed


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

# Module-level so the model survives between jobs. This only works because the
# worker runs as rq SimpleWorker: the default rq worker forks a work horse per
# job, which would reload 150 MB of weights every time.
_separator: Any = None


def get_separator():
    global _separator
    if _separator is None:
        from .separator import Separator

        log.info("loading model (first job on this worker)")
        t0 = time.perf_counter()
        _separator = Separator(
            settings.ckpt_path,
            device="cpu",
            n_threads=settings.threads,
        )
        log.info("model ready in %.1fs", time.perf_counter() - t0)
    return _separator


def _notify(job: Job) -> None:
    """POST the job to its callback_url, off the calling thread.

    Nobody waits on the result, and the call is against a client-controlled
    host: inline it would park a request thread (and DNS resolution is not
    covered by the socket timeout). Non-daemon so a webhook is not lost if the
    process exits right after settling -- bounded by the timeout.
    """
    if not job.callback_url:
        return

    url, payload = job.callback_url, job.to_json().encode()

    def post() -> None:
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(
                req, timeout=settings.callback_timeout_seconds
            ).close()
        except Exception as exc:  # a broken webhook must not fail the job
            log.warning("callback to %s failed: %s", url, exc)

    threading.Thread(target=post, name="callback", daemon=False).start()


def settle(job: Job) -> Job:
    """Finish and announce a job, whatever its outcome.

    Success and failure differ only in the status they set; stamping the end
    time, persisting under the terminal retention rule, and firing the webhook
    are common to both -- and skipped entirely if the client deleted the record
    while the job was in flight.
    """
    job.finished_at = time.time()
    if not is_cancelled(job.id):
        save_job(job)
        _notify(job)
    return job


def fail_job(job: Job, reason: str) -> Job:
    """Fail a job from inside its own execution.

    RQ still owns the lifecycle here -- the exception propagates and RQ marks
    its own entry -- so this must not touch the queue. Use ``abandon_job`` when
    settling a job from outside.
    """
    job.status = JobStatus.FAILED
    job.error = reason
    return settle(job)


def abandon_job(job: Job, reason: str) -> Job:
    """Fail a job whose executor is gone, from outside that execution.

    Always repudiates the RQ entry: nothing is running this job, so leaving the
    entry intact lets RQ re-dispatch it and race the settlement, which surfaces
    to a client as "failed" followed minutes later by "succeeded".
    """
    _cancel_rq_job(job.id, delete=True)
    return fail_job(job, reason)


def settle_if_stale(job: Job, after: float | None = None) -> Job:
    """Fail a job that is still RUNNING but has stopped being written.

    Covers the case a worker heartbeat cannot: the worker is alive but the job
    behind it is wedged. Worker death is detected far faster by
    ``live_workers()``; this is the slower backstop for a job going quiet.

    The threshold is a parameter rather than a property of the record so that a
    caller can vary it without reaching through global config.
    """
    if job.status is not JobStatus.RUNNING:
        return job

    limit = settings.stale_job_seconds if after is None else after
    if job.silent_for < limit:
        return job

    silent_min = job.silent_for / 60
    log.warning("job %s silent for %.0f min; abandoning", job.id, silent_min)
    return abandon_job(
        job,
        f"no progress for {silent_min:.0f} minutes; the worker handling this "
        f"job is gone. Retry the job.",
    )


def _abandon_running(redis: redis.Redis, keys: Sequence[bytes]) -> int:
    """Abandon every RUNNING record among ``keys``. Returns how many."""
    failed = 0
    for raw in redis.mget(keys):
        if not raw:
            continue
        try:
            job = Job.from_json(raw)
        except Exception:
            log.warning("skipping unreadable job record")
            continue
        if job.status is not JobStatus.RUNNING:
            continue
        abandon_job(
            job,
            "worker restarted while this job was running; it was not resumed. "
            "Retry the job.",
        )
        log.warning("marked orphaned job %s as failed", job.id)
        failed += 1
    return failed


def reconcile_orphaned_jobs() -> int:
    """Fail any job still marked ``running`` at worker startup.

    Exactly one worker runs at a time, so if this process is only now booting,
    nothing can legitimately be mid-flight. A worker killed mid-job -- OOM, a
    redeploy, a host reboot -- never reaches ``run_separation``'s finally block,
    so its record would sit at "running" until the 24h TTL expired. A client
    polling it would wait forever for a status that can never change.

    Failing rather than requeueing is deliberate: whatever killed the worker
    (most often memory) would very likely kill it again on the same input, and
    a crash-loop is worse than an honest error.
    """
    redis = get_redis()
    orphaned = 0

    # MGET each scan batch rather than a round-trip per key.
    for chunk in batched(redis.scan_iter(match=_key("*"), count=100), 100):
        orphaned += _abandon_running(redis, chunk)

    return orphaned


def _materialize_input(job: Job, source: str) -> Path:
    """Return a local path for ``source``, downloading it if it is a URL.

    Fetching happens in the worker rather than the API so that a slow origin
    delays one queued job instead of occupying a web worker.
    """
    if not source.startswith(("http://", "https://")):
        return Path(source)

    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.inbox_dir / f"{job.id}.download"
    log.info("fetching %s", source)
    last_stamp = time.time()
    with urllib.request.urlopen(source, timeout=60) as resp, dest.open("wb") as out:
        downloaded = 0
        while block := resp.read(1 << 20):
            downloaded += len(block)
            if downloaded > settings.max_upload_bytes:
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"source exceeds max_upload_bytes ({settings.max_upload_bytes} B)"
                )
            out.write(block)
            # A slow fetch writes nothing for minutes; without a heartbeat here
            # settle_if_stale() would declare this live job dead.
            if time.time() - last_stamp > 60:
                last_stamp = time.time()
                save_job(job)
    return dest


def run_separation(job_id: str, input_path: str) -> dict:
    """RQ entrypoint. Runs in the worker process with the model already warm."""
    from .separator import SeparationConfig

    job = load_job(job_id)
    if job is None:
        raise RuntimeError(f"job {job_id} vanished from Redis before it ran")

    job.status = JobStatus.RUNNING
    job.started_at = time.time()
    save_job(job)

    local_input = input_path
    try:
        local_input = str(_materialize_input(job, input_path))
        separator = get_separator()
        cfg = SeparationConfig(
            quality=job.quality,
            inference_batch_size=settings.batch_size,
            segment_seconds=settings.segment_seconds,
            output_format=job.output_format,
        )

        last_written = 0.0

        def on_progress(fraction: float) -> None:
            # Throttled: the callback fires per inference batch, and both the
            # cancel check and the write are round-trips.
            nonlocal last_written
            now = time.time()
            if now - last_written < 2.0 and fraction < 1.0:
                return
            last_written = now
            if is_cancelled(job_id):
                raise JobCancelled(job_id)
            job.progress = round(fraction, 4)
            save_job(job)

        paths = separator.separate_file(
            local_input,
            job_dir(job_id),
            cfg=cfg,
            stems=job.stems or None,
            progress_cb=on_progress,
        )

        # The reaper sweeps by directory mtime, which is set when the stem
        # files are created -- at job start, since they are all opened up front.
        # Restamp so artifacts are not swept before the record they belong to.
        os.utime(job_dir(job_id), None)

        job.status = JobStatus.SUCCEEDED
        job.progress = 1.0
        job.stems = sorted(paths)
        settle(job)
        return {"job_id": job_id, "stems": job.stems}

    except JobCancelled:
        log.info("job %s cancelled by DELETE", job_id)
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        return {"job_id": job_id, "cancelled": True}

    except Exception as exc:
        log.exception("job %s failed", job_id)
        fail_job(job, f"{type(exc).__name__}: {exc}")
        raise

    finally:
        # The source audio is dead weight either way.
        Path(local_input).unlink(missing_ok=True)
