#!/usr/bin/env python3
"""Does int8 dynamic quantization pay off, and what does it cost in quality?

~90% of BandIt's compute is bidirectional GRU + Linear, which is exactly what
``quantize_dynamic`` targets. On a CPU without AVX-512/AMX this is the main
lever available; bf16 autocast has no hardware to exploit.

Quality is reported as SNR of the quantized stems against the fp32 stems, so
"how much did I break it" is a number rather than a vibe. >30 dB is inaudible
in practice, <20 dB warrants a listen before shipping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from bandit_api.separator import SeparationConfig, Separator  # noqa: E402

VARIANTS = {
    "fp32": None,
    "int8-gru": {nn.GRU},
    "int8-gru-linear": {nn.GRU, nn.Linear},
}


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    noise = ref - test
    d = np.mean(noise**2)
    if d == 0:
        return float("inf")
    return float(10 * np.log10(np.mean(ref**2) / d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--ckpt", default=Path("models/checkpoint-multi.ckpt"), type=Path)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--outdir", type=Path, default=Path("data/quant"))
    ap.add_argument("--json", type=Path, default=Path("data/quant/results.json"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = SeparationConfig(quality=args.quality, inference_batch_size=args.batch)

    dur = sf.info(str(args.audio)).duration
    print(f"audio: {dur:.1f}s | quality={args.quality} | threads={args.threads}\n")

    results, stems_by_variant = [], {}

    for name, qspec in VARIANTS.items():
        sep = Separator(args.ckpt, n_threads=args.threads)

        if qspec is not None:
            t0 = time.perf_counter()
            sep.model = torch.ao.quantization.quantize_dynamic(
                sep.model, qspec, dtype=torch.qint8
            )
            print(f"{name}: quantized in {time.perf_counter() - t0:.1f}s")

        outdir = args.outdir / name
        t0 = time.perf_counter()
        paths = sep.separate_file(args.audio, outdir, cfg=cfg)
        wall = time.perf_counter() - t0

        stems_by_variant[name] = {
            s: sf.read(str(p), dtype="float32")[0] for s, p in paths.items()
        }
        row = {"variant": name, "wall_seconds": round(wall, 1), "xrt": round(wall / dur, 2)}
        results.append(row)
        print(f"{name:<16} {wall:>7.1f}s  {wall / dur:>6.2f}xRT")
        del sep

    ref = stems_by_variant["fp32"]
    print(f"\n{'variant':<16} {'xRT':>7} {'speedup':>8}   quality vs fp32 (SNR dB)")
    print("-" * 68)
    base_xrt = results[0]["xrt"]
    for row in results:
        name = row["variant"]
        if name == "fp32":
            print(f"{name:<16} {row['xrt']:>6.2f}x {'1.00x':>8}   (reference)")
            continue
        snrs = {s: snr_db(ref[s], stems_by_variant[name][s]) for s in ref}
        row["snr_db"] = {k: round(v, 1) for k, v in snrs.items()}
        row["speedup"] = round(base_xrt / row["xrt"], 2)
        qual = "  ".join(f"{s}:{v:5.1f}" for s, v in snrs.items())
        print(f"{name:<16} {row['xrt']:>6.2f}x {row['speedup']:>7.2f}x   {qual}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
