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

    @property
    def silent_for(self) -> float:
        """Seconds since anything last wrote this record."""
        return time.time() - self.updated_at

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)

    @property
    def is_stale(self) -> bool:
        """Running, but nothing has touched it for long enough to be dead."""
        return (
            self.status is JobStatus.RUNNING
            and self.silent_for >= settings.stale_job_seconds
        )

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
    """Persist a job, expiring it on a schedule that matches its status.

    ``result_ttl_seconds`` is a *result* retention policy, so applying it to a
    job that has no result yet is simply wrong: a queued job is written once, at
    enqueue, and with one worker and 25-95 minute jobs a backlog pushes later
    jobs past 24h. The record would evaporate while the RQ entry survived,
    404ing a job that was still genuinely pending.

    Making the TTL depend on status fixes that where the state is written, so it
    also holds for a job nobody ever polls -- the documented callback_url flow.
    """
    job.updated_at = time.time()
    ttl = (
        settings.result_ttl_seconds
        if job.is_terminal
        # Long enough to outlive the queue plus the job itself; the clock that
        # actually matters restarts when the job settles.
        else settings.result_ttl_seconds + settings.job_timeout_seconds
    )
    get_redis().set(_key(job.id), job.to_json(), ex=ttl)


def load_job(job_id: str) -> Job | None:
    raw = get_redis().get(_key(job_id))
    return Job.from_json(raw) if raw else None


def job_dir(job_id: str) -> Path:
    return settings.outputs_dir / job_id


def _cancel_key(job_id: str) -> str:
    return f"bandit:cancel:{job_id}"


WORKER_ALIVE_KEY = "bandit:worker:alive"


def worker_is_live() -> bool:
    """Is a worker process actually running?

    Deliberately not derived from RQ's registry or from job records. RQ pins a
    busy worker's registration to job_timeout (12h here) and SimpleWorker never
    refreshes its heartbeat mid-job, so a killed worker looks alive for half a
    day; inferring liveness from a job's progress writes instead couples
    readiness to the Job schema and forces one timeout to serve two opposing
    policies. The worker asserts its own existence on a short TTL, so the key is
    gone within ``worker_alive_ttl_seconds`` of the process dying.

    "Process exists" and "this job is progressing" are separate questions:
    ``Job.is_stale`` answers the second.
    """
    return bool(get_redis().exists(WORKER_ALIVE_KEY))


def start_liveness_heartbeat(name: str) -> None:
    """Refresh the liveness key from a daemon thread for the process lifetime.

    A thread rather than a call inside the work loop: it keeps beating through
    model loading, a blocking ffmpeg decode, and idle waits alike, so there is
    no phase where a healthy worker looks dead. It dies with the process, which
    is exactly the semantics wanted.
    """
    import threading

    ttl = settings.worker_alive_ttl_seconds

    def beat() -> None:
        while True:
            try:
                get_redis().set(WORKER_ALIVE_KEY, name, ex=ttl)
            except Exception as exc:
                log.warning("liveness heartbeat failed: %s", exc)
            time.sleep(max(5, ttl // 3))

    get_redis().set(WORKER_ALIVE_KEY, name, ex=ttl)
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
    if not job.callback_url:
        return
    try:
        req = urllib.request.Request(
            job.callback_url,
            data=job.to_json().encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(
            req, timeout=settings.callback_timeout_seconds
        ).close()
    except Exception as exc:  # a broken webhook must not fail the job
        log.warning("callback to %s failed: %s", job.callback_url, exc)


def fail_job(job: Job, reason: str, *, drop_rq: bool = False) -> Job:
    """Settle a job as failed. The single owner of that transition.

    Previously open-coded in three places, which had already drifted: only one
    dropped the RQ job, only one honoured the cancel tombstone. Anything added
    to what failing means -- a reason code, a retry counter, artifact cleanup --
    belongs here and nowhere else.
    """
    if drop_rq:
        _cancel_rq_job(job.id, delete=True)

    job.status = JobStatus.FAILED
    job.error = reason
    job.finished_at = time.time()

    # Never resurrect or announce a record the client deleted mid-flight.
    if not is_cancelled(job.id):
        save_job(job)
        _notify(job)
    return job


# A running job rewrites its record every couple of seconds via the progress
# callback. Half an hour of silence means nothing is behind it. Generous on
# purpose: the pre-inference phases (fetching a source URL, ffmpeg-decoding a
# long video) write nothing, and failing a live job would be worse than being
# slow to notice a dead one.
STALE_RUNNING_SECONDS = 30 * 60


def settle_if_stale(job: Job) -> Job:
    """Fail a job whose record has stopped being written.

    ``reconcile_orphaned_jobs`` only runs when a worker *starts*. If the worker
    dies and never comes back -- host down, container stopped, restart policy
    exhausted -- nothing settles the job and a client polls a frozen "running"
    until the record expires. Checking on read makes recovery independent of
    whether a worker ever returns.

    Staleness itself lives on ``Job.is_stale`` so that this, reconciliation and
    the readiness probe cannot drift apart on what "dead" means.
    """
    if not job.is_stale:
        return job

    silent_min = job.silent_for / 60
    log.warning("job %s silent for %.0f min; settling as failed", job.id, silent_min)
    return fail_job(
        job,
        f"no progress for {silent_min:.0f} minutes; the worker handling this "
        f"job is gone. Retry the job.",
    )


def _fail_running(redis: "redis.Redis", keys: list) -> int:
    """Fail every RUNNING record among ``keys``. Returns how many."""
    if not keys:
        return 0
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
        # drop_rq: otherwise RQ re-dispatches the interrupted job and races this
        # reconciliation, giving the client "failed" then "succeeded".
        fail_job(
            job,
            "worker restarted while this job was running; it was not resumed. "
            "Retry the job.",
            drop_rq=True,
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
    batch: list[bytes] = []
    for key in redis.scan_iter(match=_key("*"), count=100):
        batch.append(key)
        if len(batch) < 100:
            continue
        orphaned += _fail_running(redis, batch)
        batch.clear()
    orphaned += _fail_running(redis, batch)

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

        # The reaper sweeps output dirs by mtime, and a directory's mtime is
        # set when its files are *created* -- i.e. at job start, since all stem
        # files are opened up front. Left alone, a 90-minute job's artifacts are
        # deleted 90 minutes before its record expires, leaving a "succeeded"
        # job whose every stem download 404s. Restamp it at completion.
        os.utime(job_dir(job_id), None)

        job.status = JobStatus.SUCCEEDED
        job.progress = 1.0
        job.stems = sorted(paths)
        job.finished_at = time.time()
        if not is_cancelled(job_id):
            save_job(job)
            _notify(job)
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
