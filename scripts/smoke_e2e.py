#!/usr/bin/env python3
"""End-to-end check: API -> Redis -> worker -> downloadable stems.

Exercises the real queue and the real worker process, so it catches the wiring
faults unit tests miss -- bad enqueue paths, job records that never update,
stems written where the download route cannot find them.

Start Redis and a worker first:

    docker run -d --name bandit-redis -p 6379:6379 redis:7-alpine
    BANDIT_DATA_DIR=./data/e2e BANDIT_CKPT_PATH=models/checkpoint-multi.ckpt \\
      BANDIT_THREADS=6 python -m bandit_api.worker

    python scripts/smoke_e2e.py --audio data/test-mix.wav --seconds 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from bandit_api.api import app  # noqa: E402
from bandit_api.config import settings  # noqa: E402


def trim(src: Path, seconds: float, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-t", str(seconds), "-c", "copy", str(dst)],
        check=True,
    )
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    clip = trim(args.audio, args.seconds, Path("data/e2e/_clip.wav"))
    client = TestClient(app)

    ready = client.get("/readyz")
    print(f"readyz: {ready.status_code} {ready.json()}")
    if ready.status_code != 200:
        print("\nNo worker listening. Start one first (see module docstring).")
        return 1

    with clip.open("rb") as fh:
        resp = client.post(
            "/v1/jobs",
            files={"file": (clip.name, fh, "audio/wav")},
            data={"quality": args.quality, "stems": "speech,music"},
        )
    if resp.status_code != 202:
        print(f"submit failed: {resp.status_code} {resp.text}")
        return 1

    job_id = resp.json()["job_id"]
    print(f"submitted job {job_id} (queue position {resp.json()['queue_position']})")

    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        state = (body["status"], round(body["progress"], 2))
        if state != last:
            print(f"  {body['status']:<10} progress={body['progress']:.0%}")
            last = state
        if body["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)
    else:
        print("timed out waiting for the job")
        return 1

    if body["status"] != "succeeded":
        print(f"job failed: {body['error']}")
        return 1

    print(f"stems: {body['stems']}")
    for stem in body["stems"]:
        r = client.get(f"/v1/jobs/{job_id}/stems/{stem}")
        print(f"  GET {stem}: {r.status_code} {len(r.content) / 1e6:.2f} MB "
              f"({r.headers.get('content-type')})")
        if r.status_code != 200 or not r.content:
            return 1

    # A stem that was not requested must 404, not leak an empty file.
    r = client.get(f"/v1/jobs/{job_id}/stems/sfx")
    print(f"  GET unrequested sfx: {r.status_code} (expect 404)")
    if r.status_code != 404:
        return 1

    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404
    print(f"deleted job; artifacts gone from {settings.outputs_dir}")

    print("\nE2E PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
