"""Runtime configuration, all overridable by environment variable."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BANDIT_", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"

    data_dir: Path = Path("/data")
    ckpt_path: Path = Path("/data/models/checkpoint-multi-inference.pt")

    # Leave headroom for uvicorn, Redis and the OS. Oversubscribing torch
    # threads slows RNN inference down rather than speeding it up.
    threads: int = 6
    batch_size: int = 4
    segment_seconds: float = 60.0

    default_quality: str = "balanced"
    default_output_format: str = "wav"

    # Artifacts are large (tens of MB per stem). Without expiry a small disk
    # fills within days.
    result_ttl_seconds: int = 24 * 3600
    job_timeout_seconds: int = 12 * 3600

    # A running job rewrites its record every few seconds via the progress
    # callback. This much silence means nothing is behind it. Generous on
    # purpose: the pre-inference phases (fetching a source URL, ffmpeg-decoding
    # a long video) write nothing, and failing a live job is worse than being
    # slow to notice a dead one.
    stale_job_seconds: int = 30 * 60

    # The worker refreshes a liveness key from a daemon thread, independently of
    # what it is doing. Missing key => no worker process.
    worker_alive_ttl_seconds: int = 90

    # Webhooks are fire-and-forget; nobody waits on the result, and a slow
    # callback host must not hold up a request or worker startup.
    callback_timeout_seconds: int = 5

    max_upload_bytes: int = 2 * 1024**3  # 2 GiB

    # When set, every /v1 route requires `Authorization: Bearer <key>`.
    # Leave empty only if the API is not reachable from the internet: an open
    # endpoint here means anyone can queue multi-hour CPU jobs and 2 GiB uploads.
    api_key: str | None = None

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"


settings = Settings()
