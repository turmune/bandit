"""FastAPI surface for BandIt v2 separation.

Deliberately thin: this process never imports torch and never separates
anything. It validates, persists the upload, enqueues, and serves results.
That keeps it restartable in milliseconds and keeps a 12-hour job from
occupying a web worker.
"""

from __future__ import annotations

import logging
import re
import secrets
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from .config import settings
from .jobs import (
    Job,
    JobStatus,
    QUEUE_NAME,
    delete_job,
    get_queue,
    get_redis,
    job_dir,
    load_job,
    save_job,
    settle_if_stale,
    worker_is_live,
)
from .model import STEMS

log = logging.getLogger(__name__)

app = FastAPI(
    title="BandIt v2 Separation API",
    version="1.0.0",
    description=(
        "Cinematic audio source separation into speech / music / sfx stems.\n\n"
        "Separation takes minutes to hours on CPU, so submission is "
        "asynchronous: POST returns a job id, then poll or supply a callback_url."
    ),
)

VALID_QUALITY = ("fast", "balanced", "best")
VALID_FORMATS = ("wav", "flac")


def require_api_key(authorization: str = Header(default="")) -> None:
    """Bearer-token gate on /v1. No-op when BANDIT_API_KEY is unset.

    Health endpoints stay open so the proxy and Coolify can probe them.
    """
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    # Constant-time compare: a naive == leaks key length and prefix by timing.
    if not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "missing or invalid bearer token")


protected = [Depends(require_api_key)]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    quality: str
    stems: list[str] = []
    output_format: str
    source_name: str | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int | None = None

    @classmethod
    def of(cls, job: Job, queue_position: int | None = None) -> "JobResponse":
        return cls(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            quality=job.quality,
            stems=job.stems,
            output_format=job.output_format,
            source_name=job.source_name,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            queue_position=queue_position,
        )


class UrlJobRequest(BaseModel):
    source_url: HttpUrl
    quality: str = settings.default_quality
    stems: list[str] | None = None
    output_format: str = settings.default_output_format
    callback_url: HttpUrl | None = None


def _validate(quality: str, output_format: str, stems: list[str] | None) -> list[str]:
    if quality not in VALID_QUALITY:
        raise HTTPException(422, f"quality must be one of {list(VALID_QUALITY)}")
    if output_format not in VALID_FORMATS:
        raise HTTPException(422, f"output_format must be one of {list(VALID_FORMATS)}")
    chosen = stems or list(STEMS)
    unknown = sorted(set(chosen) - set(STEMS))
    if unknown:
        raise HTTPException(422, f"unknown stems {unknown}; available: {STEMS}")
    return chosen


def _enqueue(job: Job, source: str) -> None:
    save_job(job)
    get_queue().enqueue(
        "bandit_api.jobs.run_separation",
        job.id,
        source,
        job_id=job.id,
        job_timeout=settings.job_timeout_seconds,
        result_ttl=settings.result_ttl_seconds,
    )


@app.post("/v1/jobs", response_model=JobResponse, status_code=202,
          dependencies=protected)
async def create_job(
    file: UploadFile = File(..., description="Audio or video file to separate"),
    quality: str = Form(settings.default_quality),
    stems: str | None = Form(None, description="Comma-separated subset of stems"),
    output_format: str = Form(settings.default_output_format),
    callback_url: str | None = Form(None),
) -> JobResponse:
    """Submit an uploaded file. Returns immediately with a job id."""
    requested = [s.strip() for s in stems.split(",")] if stems else None
    chosen = _validate(quality, output_format, requested)

    job_id = uuid.uuid4().hex
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)

    # Keep the extension so ffmpeg can sniff the container, but take it from a
    # whitelist rather than trusting a client-supplied filename.
    raw_suffix = Path(file.filename or "").suffix.lower()
    suffix = raw_suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", raw_suffix) else ".bin"
    dest = settings.inbox_dir / f"{job_id}{suffix}"

    # Stream to disk in chunks. Reading a multi-hundred-MB upload into memory
    # would compete with the worker for the same 16 GB.
    size = 0
    with dest.open("wb") as out:
        while block := await file.read(1 << 20):
            size += len(block)
            if size > settings.max_upload_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"upload exceeds {settings.max_upload_bytes} bytes"
                )
            out.write(block)

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "uploaded file is empty")

    job = Job(
        id=job_id,
        quality=quality,
        stems=chosen,
        output_format=output_format,
        source_name=file.filename,
        callback_url=callback_url,
    )
    _enqueue(job, str(dest))
    log.info("queued job %s (%s, %.1f MB)", job_id, file.filename, size / 1e6)
    return JobResponse.of(job, queue_position=len(get_queue()))


@app.post("/v1/jobs/from-url", response_model=JobResponse, status_code=202,
          dependencies=protected)
async def create_job_from_url(req: UrlJobRequest) -> JobResponse:
    """Submit by URL. The worker fetches it, so large sources do not tie up the API."""
    chosen = _validate(req.quality, req.output_format, req.stems)
    job = Job(
        id=uuid.uuid4().hex,
        quality=req.quality,
        stems=chosen,
        output_format=req.output_format,
        source_name=Path(str(req.source_url)).name,
        callback_url=str(req.callback_url) if req.callback_url else None,
    )
    _enqueue(job, str(req.source_url))
    log.info("queued job %s from url", job.id)
    return JobResponse.of(job, queue_position=len(get_queue()))


@app.get("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=protected)
def get_job(job_id: str) -> JobResponse:
    job = load_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found or expired")

    # Self-heal a job whose worker died and never came back: reconciliation at
    # worker startup cannot help if no worker ever starts.
    job = settle_if_stale(job)
    position = None
    if job.status is JobStatus.QUEUED:
        ids = get_queue().get_job_ids()
        position = ids.index(job_id) + 1 if job_id in ids else None
    return JobResponse.of(job, queue_position=position)


@app.get("/v1/jobs/{job_id}/stems/{stem}", dependencies=protected)
async def get_stem(job_id: str, stem: str) -> FileResponse:
    job = load_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found or expired")
    if job.status is not JobStatus.SUCCEEDED:
        raise HTTPException(409, f"job is {job.status.value}, not ready for download")

    path = job_dir(job_id) / f"{stem}.{job.output_format}"
    if not path.exists():
        raise HTTPException(404, f"stem {stem!r} not produced for this job")

    return FileResponse(
        path,
        media_type="audio/wav" if job.output_format == "wav" else "audio/flac",
        filename=f"{Path(job.source_name or job_id).stem}-{stem}.{job.output_format}",
    )


@app.delete("/v1/jobs/{job_id}", status_code=204, dependencies=protected)
async def remove_job(job_id: str) -> None:
    if not delete_job(job_id):
        raise HTTPException(404, "job not found or expired")


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness: the process is up. Deliberately does not touch Redis."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness: Redis reachable and a worker process actually alive.

    Sync `def` on purpose -- FastAPI runs it in the threadpool, so the blocking
    redis client cannot stall the event loop for every other request.
    """
    try:
        get_redis().ping()
    except Exception as exc:
        return JSONResponse({"status": "unavailable", "redis": str(exc)}, 503)

    live = worker_is_live()
    body = {
        "status": "ok" if live else "degraded",
        "queue": QUEUE_NAME,
        "queued_jobs": len(get_queue()),
        "workers": 1 if live else 0,
    }
    return JSONResponse(body, 200 if live else 503)
