# BandIt Separation API — integration brief

Self-contained spec for calling this API from another codebase. Everything an
implementer needs is here; no other files required.

**Base URL:** `https://<host>.<tailnet>.ts.net` — ask whoever runs the API, or
read it from `tailscale serve status` on that machine.
**Auth:** `Authorization: Bearer <BANDIT_API_KEY>` on every `/v1/*` route.
Health routes need no auth.

**Network:** the API is on a private tailnet, not the public internet. Your
service must be a member of that tailnet to reach it at all — install Tailscale
and `tailscale up`. Nothing needs to be opened inbound on your side: `from-url`
fetches and `callback_url` webhooks are both outbound *from* the worker, so the
only requirement is that any URL you hand the API is reachable from the machine
running it.

## What it does

Takes an audio or video file and splits the audio into three stems:

| stem | contents |
|---|---|
| `speech` | dialogue / vocals |
| `music` | score, songs |
| `sfx` | effects, ambience, everything else |

Video input is accepted (mp4, mov, mkv, webm) — the audio track is extracted and
the video discarded. Output is always audio (WAV or FLAC), never video.

## The one thing that shapes your integration

**This is asynchronous.** You submit a job, get an ID back immediately, and
collect results later. There is no synchronous "send audio, get stems" call, and
faster hardware does not change that — build for it.

Separation runs on a GPU (RTX 3060). The CPU figures this brief used to quote —
16× realtime at `balanced`, ~47 minutes for a 3-minute file — no longer apply.
Measured replacements, from `scripts/benchmark.py` on a 30 s stereo excerpt at
48 kHz, batch 4, on the deployed host:

| `quality` | ~xRT | 3-min file | 60-min film |
|---|---:|---:|---:|
| `fast` | 0.26× | ~0.8 min | ~16 min |
| `balanced` (default) | 0.44× | ~1.3 min | ~26 min |
| `best` | 0.94× | ~2.8 min | ~57 min |

xRT is wall time per second of audio, so below 1.0 is faster than realtime —
`best` is roughly break-even, `fast` roughly 4× quicker than the audio plays.
Re-measure with `python scripts/benchmark.py --audio <file> --seconds 30
--device cuda --batch 4` if the hardware changes; batch 4 is what the host runs
and what these numbers assume.

Mono halves the work. Short clips are disproportionately slow (fixed padding
overhead). One worker processes one job at a time; everything else queues.

**Results cross a residential uplink.** The GPU host is on a home connection, so
the download is often slower than the separation. Two things cut it sharply, and
both are free: ask for `"output_format": "flac"` (about half the bytes of WAV,
losslessly), and request only the stems you actually consume — `"stems":
["speech"]` moves a third as much data as all three.

## Endpoints

### `POST /v1/jobs/from-url` — preferred

Submit by URL. The **worker** fetches it, so nothing large passes through your
app. The URL must be publicly reachable (use a signed URL if the file is
private).

```http
POST /v1/jobs/from-url
Authorization: Bearer <key>
Content-Type: application/json

{
  "source_url": "https://example.com/clip.mp4",
  "quality": "balanced",            // fast | balanced | best   (optional)
  "stems": ["speech"],              // subset (optional, default all three)
  "output_format": "wav",           // wav | flac              (optional)
  "callback_url": "https://you.example.com/api/hook"  // optional
}
```

→ `202` with a job object (below).

### `POST /v1/jobs` — direct upload

`multipart/form-data`. Use only when the file isn't reachable by URL.

| field | |
|---|---|
| `file` | the audio/video file (required) |
| `quality` | `fast` \| `balanced` \| `best` |
| `stems` | comma-separated string, e.g. `speech,music` |
| `output_format` | `wav` \| `flac` |
| `callback_url` | absolute URL |

Max 2 GiB. → `202` with a job object.

### `GET /v1/jobs/{job_id}` — poll status

```json
{
  "job_id": "7163f8d3...",
  "status": "running",          // queued | running | succeeded | failed
  "progress": 0.47,             // 0.0 - 1.0
  "quality": "balanced",
  "stems": ["speech", "music", "sfx"],
  "output_format": "wav",
  "source_name": "clip.mp4",
  "error": null,                // string when status is "failed"
  "created_at": 1754287200.0,   // unix seconds
  "started_at": 1754287205.0,
  "finished_at": null,
  "queue_position": null        // set while status is "queued"
}
```

### `GET /v1/jobs/{job_id}/stems/{stem}` — download

`stem` is `speech`, `music`, or `sfx`. Returns raw audio bytes.

- `409` if the job hasn't succeeded yet
- `404` if that stem wasn't requested for this job

### `DELETE /v1/jobs/{job_id}` — remove job and files. → `204`

### `GET /healthz` · `GET /readyz` — no auth

`/readyz` returns `503` when no worker is available:

```json
{"status": "ok", "queue": "separation", "queued_jobs": 0, "workers": 1}
```

## Webhook (recommended over polling)

Polling for 47 minutes is wasteful. Pass `callback_url` and the API POSTs the
job to it once, on success or failure.

> **The callback payload uses `id`, not `job_id`.** Every other response uses
> `job_id`. Key off `id` in your handler.

```json
{
  "id": "7163f8d3...",
  "status": "succeeded",
  "quality": "balanced",
  "stems": ["speech", "music", "sfx"],
  "output_format": "wav",
  "source_name": "clip.mp4",
  "progress": 1.0,
  "error": null,
  "created_at": 1754287200.0,
  "started_at": 1754287205.0,
  "finished_at": 1754290021.0
}
```

The callback is **unsigned** — anyone who learns the URL can POST to it. Treat
it as a hint to re-fetch `GET /v1/jobs/{id}` rather than as trusted data, or
put an unguessable token in the URL path.

A failing webhook never fails the job; it's logged and dropped. Always keep the
job ID so you can poll as a fallback.

## Constraints

- **Results expire after 24 hours** — job records and stem files are deleted
  together. Download and store anything you need to keep.
- Output is always 48 kHz. Channel count matches the input.
- Requesting fewer stems barely speeds anything up (~2%); the expensive shared
  trunk runs regardless. It does reduce memory and download size.

## Errors

| code | meaning |
|---|---|
| `401` | missing/invalid bearer token |
| `404` | job not found, expired, or stem not produced |
| `409` | download attempted before the job succeeded |
| `413` | upload over 2 GiB |
| `422` | bad `quality`, `stems`, `output_format`, or empty file |
| `503` | (`/readyz`) no worker registered |

## Two things not to do

1. **Do not call this API from browser JavaScript.** It would expose your API
   key to anyone viewing source, and there is no CORS middleware, so the
   request would be blocked anyway. Call it from your server, and have your
   frontend talk to your own endpoints.
2. **Do not hold an HTTP request open waiting for a job.** Jobs outlive any
   reasonable timeout. Submit, store the ID, return immediately.

## Suggested shape

1. Your app submits via `from-url` and stores `job_id` against your own record.
2. The API calls your `callback_url` on completion.
3. Your handler re-fetches `GET /v1/jobs/{id}` to confirm, then downloads the
   stems it needs and stores them somewhere durable (results vanish in 24h).
4. Your frontend reads your own DB — it never talks to this API.

## curl smoke test

```bash
KEY=<your key>
BASE=https://<host>.<tailnet>.ts.net    # from `tailscale serve status`

curl $BASE/readyz

curl -X POST $BASE/v1/jobs \
  -H "Authorization: Bearer $KEY" \
  -F file=@clip.wav -F quality=fast -F stems=speech

curl $BASE/v1/jobs/<job_id> -H "Authorization: Bearer $KEY"

curl -O -J $BASE/v1/jobs/<job_id>/stems/speech \
  -H "Authorization: Bearer $KEY"
```
