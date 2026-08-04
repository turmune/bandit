#!/usr/bin/env python3
"""Download and verify BandIt v2 weights from Zenodo, then slim them down.

The published files are 447 MB PyTorch Lightning training checkpoints: optimizer
state, LR schedulers, callbacks and loop state alongside the weights. Only the
``model.*`` tensors matter at inference, so ``--convert`` rewrites them as a bare
state_dict -- about a third of the size, and no Lightning parsing on worker boot.

    python scripts/fetch_weights.py --variant multi --convert
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ZENODO_RECORD = "12701995"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# md5 checksums straight from the Zenodo record's file listing.
CHECKPOINTS = {
    "multi": "fea2868787551b0cff36cfcf7c3622a3",
    "eng": "9b74787e7f752709ce986ba1b1ac29a9",
    "cmn": "8e57ed0ad89217342482d74ddefa4f25",
    "deu": "899dba0f7d4ead63f2c11d1ba45b3387",
    "spa": "f724266793e1a4f6f76a8bb527e1c7c9",
    "fra": "60613898c0700f7c5aa64012cfa37bca",
    "fao": "66c1a2e595e1fab3962b6f28d9e8b2fa",
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(variant: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{variant}.ckpt"
    target = dest / name
    expected = CHECKPOINTS[variant]

    if target.exists() and md5sum(target) == expected:
        print(f"{name}: already present and verified")
        return target

    url = f"{BASE_URL}/{name}?download=1"
    tmp = target.with_suffix(".ckpt.part")
    print(f"downloading {url}")

    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
    print()

    actual = md5sum(tmp)
    if actual != expected:
        tmp.unlink()
        raise SystemExit(f"md5 mismatch: got {actual}, expected {expected}")

    shutil.move(str(tmp), str(target))
    print(f"{name}: verified (md5 {actual})")
    return target


def converted_path(dest: Path, variant: str) -> Path:
    return dest / f"checkpoint-{variant}-inference.pt"


def is_usable(path: Path) -> bool:
    """Cheap integrity check on an already-converted checkpoint.

    Guards against a half-written file from a container killed mid-convert,
    which would otherwise be trusted forever and boot the worker on garbage.
    """
    if not path.exists() or path.stat().st_size < 100 * 1024 * 1024:
        return False
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        return len(state) > 2000  # the real checkpoint carries 2946 tensors
    except Exception as exc:
        print(f"{path.name}: unreadable ({exc}); will re-fetch")
        return False


def convert(src: Path) -> Path:
    """Strip Lightning scaffolding down to a bare inference state_dict."""
    import torch

    from bandit_api.model import _strip_state_dict

    dst = src.with_name(src.stem + "-inference.pt")
    raw = torch.load(src, map_location="cpu", weights_only=False)
    state = _strip_state_dict(raw)
    torch.save(state, dst)

    before, after = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"converted -> {dst.name}  ({before:.0f} MB -> {after:.0f} MB)")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="multi", choices=sorted(CHECKPOINTS))
    ap.add_argument("--dest", type=Path, default=Path("models"))
    ap.add_argument("--convert", action="store_true",
                    help="also write a slimmed bare state_dict")
    ap.add_argument("--keep-original", action="store_true",
                    help="keep the Lightning .ckpt after converting")
    args = ap.parse_args()

    # Must be idempotent: this runs on every worker start, and --convert deletes
    # the source .ckpt. Checking only for the .ckpt would therefore re-download
    # 447 MB from Zenodo on every single restart.
    if args.convert:
        target = converted_path(args.dest, args.variant)
        if is_usable(target):
            print(f"{target.name}: already present and usable, skipping fetch")
            return 0

    ckpt = download(args.variant, args.dest)
    if args.convert:
        convert(ckpt)
        if not args.keep_original:
            ckpt.unlink()
            print(f"removed {ckpt.name} (pass --keep-original to retain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
