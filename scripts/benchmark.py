#!/usr/bin/env python3
"""Measure BandIt v2 CPU throughput so deployment sizing rests on numbers.

Reports wall time as a multiple of realtime (xRT) and peak RSS per quality
preset. xRT is what matters: at 20xRT a 3-minute track takes an hour.

Pin --threads to the *target* core count, not this machine's. Results from a
16-core dev box do not transfer to an 8-core server.

    python scripts/benchmark.py --audio sample.wav --threads 6
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torchaudio as ta  # noqa: E402

from bandit_api.model import SAMPLE_RATE  # noqa: E402
from bandit_api.separator import (  # noqa: E402
    QUALITY_PRESETS,
    SeparationConfig,
    Separator,
)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def make_excerpt(src: Path, seconds: float, dst: Path) -> tuple[Path, float]:
    """Trim to the first ``seconds`` so each preset benchmarks the same audio."""
    audio, fs = ta.load(str(src))
    if fs != SAMPLE_RATE:
        audio = ta.functional.resample(audio, fs, SAMPLE_RATE)
        fs = SAMPLE_RATE
    n = int(seconds * fs)
    if audio.shape[-1] > n:
        audio = audio[:, :n]
    ta.save(str(dst), audio, fs)
    return dst, audio.shape[-1] / fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--ckpt", default=Path("models/checkpoint-multi.ckpt"), type=Path)
    ap.add_argument("--seconds", type=float, default=20.0, help="excerpt length")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--quality", nargs="*", default=list(QUALITY_PRESETS))
    ap.add_argument("--outdir", type=Path, default=Path("data/bench"))
    ap.add_argument("--json", type=Path, help="write results as JSON")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    excerpt, duration = make_excerpt(
        args.audio, args.seconds, args.outdir / "_excerpt.wav"
    )

    print(f"torch {torch.__version__} | threads={args.threads} | batch={args.batch}")
    print(f"excerpt: {duration:.1f}s @ {SAMPLE_RATE} Hz\n")

    t0 = time.perf_counter()
    sep = Separator(args.ckpt, n_threads=args.threads)
    load_s = time.perf_counter() - t0
    print(f"model load: {load_s:.1f}s (paid once per worker, not per job)\n")

    results = []
    print(f"{'quality':<10} {'hop':>5} {'wall':>8} {'xRT':>7} {'3min ETA':>10} {'peak RSS':>10}")
    print("-" * 56)

    for quality in args.quality:
        cfg = SeparationConfig(
            quality=quality,
            inference_batch_size=args.batch,
            segment_seconds=60.0,
        )
        t0 = time.perf_counter()
        sep.separate_file(excerpt, args.outdir / quality, cfg=cfg)
        wall = time.perf_counter() - t0

        xrt = wall / duration
        row = {
            "quality": quality,
            "hop_seconds": cfg.hop_size_seconds,
            "wall_seconds": round(wall, 2),
            "xrt": round(xrt, 2),
            "eta_3min_seconds": round(xrt * 180),
            "peak_rss_gb": round(peak_rss_gb(), 2),
            "threads": args.threads,
            "batch": args.batch,
        }
        results.append(row)
        eta = xrt * 180
        print(
            f"{quality:<10} {cfg.hop_size_seconds:>4.0f}s {wall:>7.1f}s {xrt:>6.1f}x "
            f"{eta / 60:>9.1f}m {peak_rss_gb():>9.2f}G"
        )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    print("\nxRT = wall seconds per second of audio. '3min ETA' extrapolates to a")
    print("3-minute stereo track. Stereo costs 2x mono: the model is in_channels=1")
    print("and the handler folds channels into the batch dimension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
