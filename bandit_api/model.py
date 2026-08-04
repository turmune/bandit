"""Model construction + checkpoint loading for BandIt v2, CPU-first.

Upstream builds the model through Hydra + ``build_system()``, which imports Ray,
PyTorch Lightning and the Netflix-internal training stack. None of that is needed
to run inference, so the architecture kwargs are inlined here verbatim from
``configs/models/bandit-mus64.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .vendor.bandit import Bandit

log = logging.getLogger(__name__)

# Verbatim from configs/models/bandit-mus64.yaml
BANDIT_MUS64 = dict(
    in_channels=1,
    band_type="musical",
    n_bands=64,
    normalize_channel_independently=False,
    treat_channel_as_feature=True,
    n_sqm_modules=8,
    emb_dim=128,
    rnn_dim=256,
    bidirectional=True,
    rnn_type="GRU",
    mlp_dim=512,
    hidden_activation="Tanh",
    hidden_activation_kwargs=None,
    complex_mask=True,
    use_freq_weights=True,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    window_fn="hann_window",
    wkwargs=None,
    power=None,
    center=True,
    normalized=True,
    pad_mode="reflect",
    onesided=True,
)

# From configs/data/dnr-v3-com-smad-multi-v2b.yaml (commons.datasets.stems).
# Order matters only for readability; lookup is by name.
STEMS = ["speech", "music", "sfx"]

# Every published checkpoint is 48 kHz (expt/*.yaml: fs: 48000).
SAMPLE_RATE = 48000


def build_bandit(stems: list[str] | None = None) -> Bandit:
    """Construct the bandit-mus64 architecture with no weights loaded."""
    return Bandit(stems=list(stems or STEMS), fs=SAMPLE_RATE, **BANDIT_MUS64)


def _strip_state_dict(raw: dict) -> dict:
    """Normalise a raw checkpoint into a bare ``Bandit`` state_dict.

    The published checkpoints are PyTorch Lightning checkpoints of the training
    ``System``, so they carry optimizer state, LR schedulers and callbacks, and
    every model tensor is prefixed ``model.``. The ``loss_handler.*`` entries
    belong to the training objective and have no counterpart at inference.
    """
    sd = raw.get("state_dict", raw)

    # Must be idempotent: this runs on the published Lightning checkpoint AND
    # on the slimmed output of scripts/fetch_weights.py --convert, which has
    # already had the prefix removed. Stripping unconditionally would filter a
    # converted checkpoint down to nothing and load a model of random weights.
    wrapped = any(k.startswith("model.") for k in sd)

    out = {}
    for key, value in sd.items():
        if wrapped:
            if not key.startswith("model."):
                continue  # drops loss_handler.* weights
            key = key[len("model.") :]
        tensor = value
        # freq_weights buffers ship as float64; float64 math is roughly half
        # speed on CPU and buys nothing here.
        if hasattr(tensor, "dtype") and tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        out[key] = tensor
    return out


def load_bandit(
    ckpt_path: str | Path,
    stems: list[str] | None = None,
    device: str = "cpu",
) -> Bandit:
    """Build the model and load weights from a Lightning or bare checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run scripts/fetch_weights.py first."
        )

    model = build_bandit(stems)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_state_dict(raw)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Upstream loads with strict=False and never checks the result. We do check:
    # a silent key mismatch here yields a model full of random weights that still
    # runs and still emits audio, just noise. Fail loudly instead.
    real_missing = [k for k in missing if not k.endswith("freq_weights")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the bandit-mus64 architecture.\n"
            f"  missing ({len(real_missing)}): {real_missing[:5]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:5]}"
        )

    model.to(device)
    model.eval()
    log.info(
        "loaded %s (%.1fM params) on %s",
        ckpt_path.name,
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
    )
    return model
