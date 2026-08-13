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
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
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
    # Set on every save_job(). Staleness was previously inferred from the key's
    # remaining TTL, which silently breaks as soon as anything refreshes the TTL
    # without rewriting the record -- which is exactly what keep_alive() does.
    updated_at: float = field(default_factory=time.time)

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
    job.updated_at = time.time()
    get_redis().set(_key(job.id), job.to_json(), ex=settings.result_ttl_seconds)


def keep_alive(job_id: str) -> None:
    """Extend a record's TTL without counting as progress.

    A queued job is written once, at enqueue. With one worker and jobs running
    25-95 minutes, a modest backlog pushes later jobs past the 24h TTL: the
    record evaporates while the RQ entry survives, so the client gets 404 for a
    job that is still genuinely pending and will later run with nowhere to
    report its result.
    """
    get_redis().expire(_key(job_id), settings.result_ttl_seconds)


def load_job(job_id: str) -> Job | None:
    raw = get_redis().get(_key(job_id))
    return Job.from_json(raw) if raw else None


def job_dir(job_id: str) -> Path:
    return settings.outputs_dir / job_id


def _cancel_key(job_id: str) -> str:
    return f"bandit:cancel:{job_id}"


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
    try:
        from rq.job import Job as RQJob

        RQJob.fetch(job_id, connection=get_redis()).cancel()
    except Exception:
        pass  # not queued, or already gone
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
    if not job.callback_url:
        return
    try:
        req = urllib.request.Request(
            job.callback_url,
            data=job.to_json().encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15).close()
    except Exception as exc:  # a broken webhook must not fail the job
        log.warning("callback to %s failed: %s", job.callback_url, exc)


# A running job rewrites its record every couple of seconds via the progress
# callback. Half an hour of silence means nothing is behind it. Generous on
# purpose: the pre-inference phases (fetching a source URL, ffmpeg-decoding a
# long video) write nothing, and failing a live job would be worse than being
# slow to notice a dead one.
STALE_RUNNING_SECONDS = 30 * 60


def settle_if_stale(job: Job) -> Job:
    """Fail a job whose record has stopped being written.

    ``reconcile_orphaned_jobs`` only runs when a worker *starts*. If the worker
    dies and never comes back -- the host is down, the container is stopped, the
    restart policy gave up -- nothing settles the job and a client polls a
    permanently frozen "running" until the 24h record TTL expires.

    ``save_job`` re-sets that TTL on every write, so the *remaining* TTL is a
    free proxy for time-since-last-write: no extra field, no clock skew between
    processes. Checking it on read makes the API self-healing regardless of
    whether a worker ever returns.
    """
    if job.status is not JobStatus.RUNNING:
        return job

    silent_for = time.time() - (job.updated_at or job.created_at)
    if silent_for < STALE_RUNNING_SECONDS:
        return job

    log.warning(
        "job %s has not been written in %.0f min; settling as failed",
        job.id, silent_for / 60,
    )
    job.status = JobStatus.FAILED
    job.error = (
        f"no progress for {silent_for / 60:.0f} minutes; the worker handling "
        f"this job is gone. Retry the job."
    )
    job.finished_at = time.time()
    save_job(job)
    _notify(job)
    return job


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

    for key in redis.scan_iter(match="bandit:job:*", count=100):
        raw = redis.get(key)
        if not raw:
            continue
        try:
            job = Job.from_json(raw)
        except Exception:
            log.warning("skipping unreadable job record %r", key)
            continue

        if job.status is not JobStatus.RUNNING:
            continue

        # Drop it from RQ first. RQ may otherwise re-dispatch the interrupted
        # job, which would race this reconciliation: the client would see
        # "failed" and then, minutes later, "succeeded" -- or two webhooks
        # contradicting each other. Cancelling makes the outcome deterministic.
        try:
            from rq.job import Job as RQJob

            rq_job = RQJob.fetch(job.id, connection=redis)
            rq_job.cancel()
            rq_job.delete()
        except Exception as exc:  # already gone, or never registered
            log.debug("no RQ job to cancel for %s (%s)", job.id, exc)

        job.status = JobStatus.FAILED
        job.error = (
            "worker restarted while this job was running; it was not resumed. "
            "Retry the job, or use a lower quality preset if the input is long."
        )
        job.finished_at = time.time()
        save_job(job)
        _notify(job)
        orphaned += 1
        log.warning("marked orphaned job %s as failed", job.id)

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
        # Fetching a source URL writes nothing; stamp the record so a slow
        # download is not mistaken for a dead worker by settle_if_stale().
        save_job(job)
        separator = get_separator()
        cfg = SeparationConfig(
            quality=job.quality,
            inference_batch_size=settings.batch_size,
            segment_seconds=settings.segment_seconds,
            output_format=job.output_format,
        )

        last_written = 0.0

        def on_progress(fraction: float) -> None:
            if is_cancelled(job_id):
                raise JobCancelled(job_id)
            # Redis write per segment, not per chunk; throttle anyway so a long
            # file does not hammer it.
            nonlocal last_written
            now = time.time()
            if now - last_written < 2.0 and fraction < 1.0:
                return
            last_written = now
            job.progress = round(fraction, 4)
            save_job(job)

        paths = separator.separate_file(
            local_input,
            job_dir(job_id),
            cfg=cfg,
            stems=job.stems or None,
            progress_cb=on_progress,
        )

        # The reaper sweeps output dirs by mtime, and a directory's mtime is
        # set when its files are *created* -- i.e. at job start, since all stem
        # files are opened up front. Left alone, a 90-minute job's artifacts are
        # deleted 90 minutes before its record expires, leaving a "succeeded"
        # job whose every stem download 404s. Restamp it at completion.
        os.utime(job_dir(job_id), None)

        job.status = JobStatus.SUCCEEDED
        job.progress = 1.0
        job.stems = sorted(paths)
    except JobCancelled:
        log.info("job %s cancelled by DELETE", job_id)
        Path(local_input).unlink(missing_ok=True)
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        return {"job_id": job_id, "cancelled": True}
    except Exception as exc:
        log.exception("job %s failed", job_id)
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        job.finished_at = time.time()
        # The source audio is dead weight once separated.
        Path(local_input).unlink(missing_ok=True)
        # Neither resurrect nor announce a record the client deleted mid-flight.
        if not is_cancelled(job_id):
            save_job(job)
            _notify(job)

    return {"job_id": job_id, "stems": job.stems}
