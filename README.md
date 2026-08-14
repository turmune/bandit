# BandIt v2 Separation API

CPU-only HTTP API around [BandIt v2](https://github.com/kwatcharasupat/bandit-v2),
which separates cinematic audio into **speech**, **music** and **sfx** stems.
Packaged for Coolify on an 8 vCPU / 16 GB host.

Separation on CPU takes minutes to hours, so the API is asynchronous: you submit
a job, then poll or receive a webhook.

```bash
curl -X POST https://your-host/v1/jobs \
  -F file=@movie-audio.wav -F quality=balanced
# {"job_id":"a1b2...","status":"queued","queue_position":1, ...}

curl https://your-host/v1/jobs/a1b2...
# {"status":"running","progress":0.42, ...}

curl -O -J https://your-host/v1/jobs/a1b2.../stems/speech
```

## Performance

Measured, not estimated — `scripts/benchmark.py`, 30 s stereo excerpt at 48 kHz,
6 torch threads, batch 4, on an **AMD Ryzen 7 3700X** (8 cores, AVX2, no
AVX-512/AMX):

| quality | hop | xRT | 3-min track | 60-min film | peak RSS |
|---|---|---:|---:|---:|---:|
| `fast` | 4.0 s | **6.9×** | ~21 min | ~7 h | 5.2 GB |
| `balanced` | 2.0 s | **16.4×** | ~49 min | ~16 h | 5.6 GB |
| `best` | 1.0 s | ~32× (extrapolated) | ~1.6 h | ~32 h | ~6 GB |

xRT = wall-clock seconds per second of audio. **Reproduce on your own hardware
before sizing anything** — pin `--threads` to the target core count, not the dev
box's:

```bash
python scripts/benchmark.py --audio your.wav --seconds 30 --threads 6
```

Read this honestly: on 8 CPU cores this is a batch system, not an interactive
one. A feature-length film at `balanced` runs overnight.

Caveat: a cloud "8 vCPU" is often 4 physical cores with SMT, which would be
meaningfully slower than the numbers above.

### Making it faster

In measured order of payoff:

1. **Use `fast`** — 2.4× over `balanced`, and the only lever that costs nothing
   but overlap-add smoothing.
2. **Send mono** — halves the work outright. The model is `in_channels=1` and
   the handler folds channels into the batch, so stereo is literally two passes.
   If the consumer is ASR on the speech stem, mono is free.
3. **Move the worker to a GPU host** — the job-queue boundary means this is a
   one-container swap, not a rewrite.

**int8 dynamic quantization: measured and rejected.** The obvious idea, since
~90% of the compute is bidirectional GRU + Linear. On this AVX2 CPU it returned
only **1.18×** (`scripts/experiment_quantize.py`), while measurably perturbing
the output — not a trade worth making. The likely reason is that fbgemm's int8
kernels need AVX-512 VNNI to pull ahead, which AMD Zen 2 does not have. **If
your server is an Intel Xeon with VNNI, re-run that script there before
dismissing it** — the same experiment can look very different on that hardware.
It prints both speedup and per-stem SNR against fp32, so the quality cost is a
number, not a guess.

## Why not just run upstream `inference.py`?

Upstream is research code driven by Hydra, Ray and PyTorch Lightning. Four things
block deploying it as-is:

| Blocker | Resolution |
|---|---|
| `requirements.in` depends on `nflx-manta`, `nflx_metaflow`, `jasper` — Netflix-internal, not on PyPI | vendored the inference subset; wrote a real dependency list |
| `inference.py` hardcodes `.to("cuda")` | device is a parameter |
| Handler accumulates every chunk for every stem in RAM before folding — memory grows with duration | outer segmentation with discarded context margins; peak RAM is now constant |
| `torchaudio.io.StreamReader` import fails on torchaudio ≥ 2.2 | streaming handlers dropped (unused for tensor inference) |

`bandit_api/vendor/` holds the vendored model code. Every change to it is listed
in that package's docstring.

## Layout

```
bandit_api/
  api.py         FastAPI surface — never imports torch
  worker.py      RQ SimpleWorker entrypoint, model warm across jobs
  jobs.py        job records in Redis + the separation task
  separator.py   CPU driver: segmentation, resampling, incremental writes
  model.py       architecture kwargs + strict checkpoint loading
  vendor/        vendored inference subset of bandit-v2 (Apache-2.0)
scripts/
  fetch_weights.py       download + md5-verify + slim the checkpoint
  benchmark.py           throughput per quality preset
  experiment_quantize.py int8 dynamic quantization: speed vs quality
  smoke_e2e.py           API -> queue -> worker -> download
```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python scripts/fetch_weights.py --variant multi --convert
docker run -d --name bandit-redis -p 6379:6379 redis:7-alpine

# terminal 1
BANDIT_DATA_DIR=./data BANDIT_CKPT_PATH=models/checkpoint-multi-inference.pt \
  BANDIT_REDIS_URL=redis://localhost:6379/0 python -m bandit_api.worker

# terminal 2
BANDIT_DATA_DIR=./data BANDIT_REDIS_URL=redis://localhost:6379/0 \
  uvicorn bandit_api.api:app --reload
```

Tests:

```bash
pytest tests/ -q            # fast: config + loading
pytest tests/ -q -m slow    # real inference; minutes
```

## Deploying on Coolify

**0. DNS first.** Point an `A` record at the server's IP and let it propagate.
Coolify issues the Let's Encrypt certificate during deploy, and that fails if
the name does not already resolve.

```
bandit.efobay.com.   A   <server-ip>
```

**1. Push this repo** somewhere Coolify can read (GitHub/Gitea, public or via a
deploy key). Coolify's Compose build pack builds from source, so it needs the
repository, not just the compose file.

**2. New Resource → Docker Compose**, select the repo.
- Branch: `main`
- Base Directory: `/`
- Compose file: `docker-compose.yaml`

**3. Environment Variables** (Coolify UI). `BANDIT_API_KEY` is required — the
compose file refuses to start without it, deliberately:

```
BANDIT_API_KEY=<paste: openssl rand -hex 32>
```

**4. Domain — on the `api` service only.** In that service's Domains field:

```
https://bandit.efobay.com:8000
```

The `:8000` tells Coolify which container port to route to; it is not part of
the public URL. Leave `worker`, `redis` and `reaper` with no domain.

**5. Deploy.** First boot the worker pulls ~447 MB from Zenodo and converts it,
so it is not ready immediately. Watch the `worker` logs; `/readyz` returns `503`
with `"status":"degraded"` until a worker registers.

**6. Verify.**

```bash
curl https://bandit.efobay.com/healthz
curl https://bandit.efobay.com/readyz          # {"status":"ok","workers":1}

curl -X POST https://bandit.efobay.com/v1/jobs \
  -H "Authorization: Bearer $BANDIT_API_KEY" \
  -F file=@clip.wav -F quality=fast
```

### Notes

**Uploads.** Traefik imposes no request-body limit by default, so large files
should pass through; the app's own 2 GiB cap (`BANDIT_MAX_UPLOAD_BYTES`) is what
you will hit first. If a proxy in front of Coolify does cap bodies, use
`POST /v1/jobs/from-url` and skip the upload path entirely.

**Never redeploy while a job is running.** A Docker build unpacks several GB
of torch while the worker is holding ~5 GB. On a 16 GB host that has been
observed exceeding total memory, at which point the kernel kills whichever
process is largest -- the worker mid-job, and it could as easily be Coolify.
Check `/readyz` for `queued_jobs: 0` and no running job first, or accept that
the in-flight job will be abandoned (it settles cleanly with a reason).

**Weights are not baked into the image** — 447 MB on every rebuild, and Coolify
rebuilds on every push. They live on the `bandit-data` volume, fetched once and
md5-verified.

**Three services build the same image.** `api`, `worker` and `reaper` all use
`build: .`; Docker's layer cache makes the second and third near-instant.

**Every service has a healthcheck**, so nothing reports health `unknown`:

| service | probe | start period |
|---|---|---|
| `api` | `GET /healthz` | 20 s |
| `worker` | an RQ worker for *this container* is beating in Redis | 1800 s |
| `redis` | `redis-cli ping` | — |
| `reaper` | sweep loop touched its liveness file within 70 min | 60 s |

The worker's long start period covers the first-boot Zenodo download; it shows
`starting` until the model is loaded and the worker registers. Failures during a
start period do not count toward retries and a first success ends it early, so
the generous value costs nothing. The probe matches on hostname deliberately —
otherwise one container's check could pass because a *different* worker is
healthy.

### Environment

| Variable | Default | Notes |
|---|---|---|
| `BANDIT_REDIS_URL` | `redis://localhost:6379/0` | |
| `BANDIT_DATA_DIR` | `/data` | holds `inbox/`, `outputs/`, `models/` |
| `BANDIT_CKPT_PATH` | `/data/models/checkpoint-multi-inference.pt` | |
| `BANDIT_THREADS` | `6` | torch intra-op threads; keep ≈ cores − 2 |
| `BANDIT_BATCH_SIZE` | `2` | chunks per forward pass; drives peak RAM |
| `BANDIT_SEGMENT_SECONDS` | `60` | outer segment size; drives peak RAM. **Must be an integer multiple of every hop you use** (4/2/1 s) — 120 satisfies all three; 90 would fail `fast` jobs at runtime |
| `BANDIT_DEFAULT_QUALITY` | `balanced` | |
| `BANDIT_RESULT_TTL_SECONDS` | `86400` | job records and artifacts expire together |
| `BANDIT_MAX_UPLOAD_BYTES` | `2147483648` | |

## Client (`./bandit`)

A standard-library-only CLI — no virtualenv, no torch, runs from any checkout.

```bash
cp .env.example .env      # then paste your BANDIT_API_KEY

cp ~/Music/song.wav data/in/
./bandit                  # separates everything in data/in -> data/out/<name>/
```

```
$ ./bandit
https://bandit.efobay.com  ready  1 worker(s), 0 queued
  quality=balanced  1 file(s)

song.wav  38.2MB
  job a1b2c3d4 queued
  separating 47% · ~12m left · 10.6m elapsed
  done in 22.4m -> data/out/song/
    music   34.1MB
    sfx     34.1MB
    speech  34.1MB
```

| command | |
|---|---|
| `./bandit` | separate everything in `data/in/` |
| `./bandit a.wav b.mp4` | separate specific files |
| `./bandit -q fast a.wav` | `fast` \| `balanced` \| `best` |
| `./bandit -s speech a.wav` | only the stems you want |
| `./bandit -f flac a.wav` | FLAC instead of WAV |
| `./bandit --status` | server health and queue depth |
| `./bandit --jobs` | jobs submitted from this machine |

Ctrl-C is safe — the job keeps running server-side, and `./bandit --jobs` picks
it back up.

Config resolves from `--url`/`--key`, then the environment, then `.env`. If the
URL carries a `:8000` suffix (Coolify's container-port hint, easy to copy by
mistake) the client strips it and says so, rather than hanging on a closed port.

**Video is handled locally.** Drop in an mp4/mov/mkv and the client copies the
audio stream out with ffmpeg before uploading — no re-encode, so it is near
instant and lossless. A 3-minute 1080p clip goes from ~200 MB to ~3 MB on the
wire. Rejecting video server-side would not help: by the time the API could
refuse it, the slow upload has already happened. Falls back to uploading the
original if ffmpeg is not installed.

**Short clips look slow.** A 6 s file spends most of its time on the analysis
padding around it, so it can measure 20×+ realtime where a 3-minute track
measures ~7× at `fast`. Benchmark with something at least a minute long.

## API

### `POST /v1/jobs`

Multipart upload. Form fields: `file` (required), `quality`, `stems`
(comma-separated), `output_format` (`wav`\|`flac`), `callback_url`.
Returns `202` with a job id.

### `POST /v1/jobs/from-url`

JSON: `{"source_url": "...", "quality": "balanced", "stems": ["speech"],
"callback_url": "..."}`. The **worker** fetches the URL, so a slow origin
queues rather than occupying a web worker. Preferred for large files — it
bypasses the proxy upload limit entirely.

### `GET /v1/jobs/{id}`

`status` is one of `queued`, `running`, `succeeded`, `failed`. Includes
`progress` (0–1) and, while queued, `queue_position`.

### `GET /v1/jobs/{id}/stems/{stem}`

Downloads a stem. `409` if the job has not succeeded, `404` if that stem was not
requested.

### `DELETE /v1/jobs/{id}`

Removes the record and its artifacts.

### `GET /healthz` · `GET /readyz`

Liveness (process up, no dependencies touched) and readiness (Redis reachable
**and** at least one worker registered — `503` otherwise, so Coolify will not
route to a stack that cannot do work).

### Quality presets

`quality` sets the overlap-add hop. Upstream's default re-processes every second
of audio 8 times; that is tuned for GPU-side quality, and it is the dominant cost
on CPU.

| preset | hop | overlap | relative cost |
|---|---|---|---|
| `fast` | 4.0 s | 2× | 1× |
| `balanced` | 2.0 s | 4× | ~2× |
| `best` | 1.0 s | 8× | ~4× (upstream default) |

## Design notes

**The API never imports torch.** It validates, writes the upload to disk,
enqueues, and serves files. It restarts in milliseconds and a 12-hour job cannot
occupy a web worker.

**`SimpleWorker`, not the default RQ worker.** RQ normally forks a work horse per
job, which would rebuild the model and reload weights on every request. Ours runs
jobs in-process so the model stays warm. A hard crash takes the worker with it —
that is what `restart: unless-stopped` is for.

**One worker, not N.** Replicas on one box halve each other's cores and multiply
peak RAM. Scale by adding hosts.

**Thread pinning.** Left alone, torch grabs every core and fights uvicorn and
Redis. Oversubscription slows RNN inference rather than speeding it up.

**Strict checkpoint loading.** Upstream loads with `strict=False` and ignores the
result. A key mismatch there yields a model of random weights that still runs and
still emits audio — just noise. `model.py` raises instead.

**Segment offsets must land on the chunk grid.** `segment_seconds` and
`margin_seconds` are validated as integer multiples of the preset's hop. The
handler unfolds chunks at a fixed stride from each segment's start, so a segment
boundary off the grid shifts every chunk relative to where it would sit in an
unsegmented run. Because the masks are nonlinear in chunk content, that is not a
rounding difference: measured against the unsegmented output it was **8 dB SNR**,
i.e. audibly wrong. `tests/test_separator.py::test_segmentation_is_transparent`
pins this.

**Stereo costs 2×.** The architecture is `in_channels=1`; the handler folds
channels into the batch dimension, so a stereo file is two mono passes.

**Requesting fewer stems saves memory, not much time.** Upstream always decodes
all three; we added an `active_stems` filter. The shared GRU trunk is ~98% of
the compute and runs regardless, so expect only a couple of percent off the
clock — but the inference handler then buffers one stem instead of three, and
that buffer is what sets peak RAM. Ask for just `speech` if that is all you
need.

## Attribution

Model and vendored inference code: [kwatcharasupat/bandit-v2](https://github.com/kwatcharasupat/bandit-v2),
Apache-2.0 (see `LICENSE-bandit-v2`). Weights: [Zenodo 12701995](https://zenodo.org/records/12701995).
Paper: [arXiv:2407.07275](https://arxiv.org/abs/2407.07275).

Upstream asks that commercial users consider contributing to music-related
non-profits.
