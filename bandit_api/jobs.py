"""Job state and the worker-side separation task.

Job records live in Redis as JSON with a TTL. The queue is RQ. The API process
never imports torch -- only this module's ``run_separation`` does, and only
inside the worker.
"""

from __future__ import annotations

import json
import logging
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
    get_redis().set(_key(job.id), job.to_json(), ex=settings.result_ttl_seconds)


def load_job(job_id: str) -> Job | None:
    raw = get_redis().get(_key(job_id))
    return Job.from_json(raw) if raw else None


def job_dir(job_id: str) -> Path:
    return settings.outputs_dir / job_id


def delete_job(job_id: str) -> bool:
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


def _materialize_input(job_id: str, source: str) -> Path:
    """Return a local path for ``source``, downloading it if it is a URL.

    Fetching happens in the worker rather than the API so that a slow origin
    delays one queued job instead of occupying a web worker.
    """
    if not source.startswith(("http://", "https://")):
        return Path(source)

    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.inbox_dir / f"{job_id}.download"
    log.info("fetching %s", source)
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
        local_input = str(_materialize_input(job_id, input_path))
        separator = get_separator()
        cfg = SeparationConfig(
            quality=job.quality,
            inference_batch_size=settings.batch_size,
            segment_seconds=settings.segment_seconds,
            output_format=job.output_format,
        )

        last_written = 0.0

        def on_progress(fraction: float) -> None:
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

        job.status = JobStatus.SUCCEEDED
        job.progress = 1.0
        job.stems = sorted(paths)
    except Exception as exc:
        log.exception("job %s failed", job_id)
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        job.finished_at = time.time()
        save_job(job)
        # The source audio is dead weight once separated.
        Path(local_input).unlink(missing_ok=True)
        _notify(job)

    return {"job_id": job_id, "stems": job.stems}
