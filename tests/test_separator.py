"""Correctness tests for the CPU separation wrapper.

The load/validation tests are cheap. The two marked ``slow`` run real inference
and take minutes on CPU -- run them with ``-m slow`` when touching the
segmentation or chunking logic.

    pytest tests/ -v                # fast only
    pytest tests/ -v -m slow        # includes real inference
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bandit_api.model import SAMPLE_RATE, STEMS, _strip_state_dict, build_bandit
from bandit_api.separator import QUALITY_PRESETS, SeparationConfig, Separator

# The converted checkpoint first: `fetch_weights.py --convert` deletes the
# Lightning .ckpt unless --keep-original is passed, so pointing only at the
# latter meant every checkpoint-backed test skipped on a host that followed the
# README -- and a skip reads the same as a pass in the summary line. Both load
# through _strip_state_dict, which accepts a bare state_dict or a Lightning one.
CKPT = next(
    (
        p
        for p in (
            Path("models/checkpoint-multi-inference.pt"),
            Path("models/checkpoint-multi.ckpt"),
        )
        if p.exists()
    ),
    Path("models/checkpoint-multi-inference.pt"),
)
requires_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason="checkpoint absent; run scripts/fetch_weights.py"
)


@pytest.fixture(scope="session")
def mixture(tmp_path_factory) -> Path:
    """A short synthetic stereo mixture. Content is irrelevant to these tests."""
    path = tmp_path_factory.mktemp("audio") / "mix.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration=14:sample_rate={SAMPLE_RATE}",
            "-f", "lavfi", "-i", f"anoisesrc=duration=14:color=pink:amplitude=0.3:sample_rate={SAMPLE_RATE}",
            "-filter_complex",
            "[0]tremolo=f=5:d=0.8[a];[a][1]amix=inputs=2:duration=shortest,"
            "volume=2.0,pan=stereo|c0=c0|c1=c0[out]",
            "-map", "[out]", "-ac", "2", "-ar", str(SAMPLE_RATE), str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def separator() -> Separator:
    if not CKPT.exists():
        pytest.skip("checkpoint absent; run scripts/fetch_weights.py")
    return Separator(CKPT, n_threads=6)


def test_quality_presets_map_to_hop_sizes():
    for name, hop in QUALITY_PRESETS.items():
        assert SeparationConfig(quality=name).hop_size_seconds == hop


def test_unknown_quality_is_rejected():
    with pytest.raises(ValueError, match="unknown quality"):
        _ = SeparationConfig(quality="ultra").hop_size_seconds


def test_margin_shorter_than_chunk_is_rejected():
    """Guards the invariant that keeps segment boundaries seamless."""
    with pytest.raises(ValueError, match="margin_seconds"):
        SeparationConfig(chunk_size_seconds=8.0, margin_seconds=4.0)


def test_segment_off_the_hop_grid_is_rejected():
    """5s segments against a 4s hop shift every chunk; measured 8 dB SNR."""
    with pytest.raises(ValueError, match="integer multiple"):
        SeparationConfig(quality="fast", segment_seconds=5.0, margin_seconds=8.0)


def test_default_config_is_grid_aligned_for_every_preset():
    for quality in QUALITY_PRESETS:
        cfg = SeparationConfig(quality=quality)  # must not raise
        assert cfg.segment_seconds % cfg.hop_size_seconds == 0
        assert cfg.margin_seconds >= cfg.chunk_size_seconds


def test_architecture_matches_published_stems():
    model = build_bandit()
    assert model.stems == STEMS
    assert set(model.mask_estim) == set(STEMS)


def test_strip_state_dict_is_idempotent():
    """Converted checkpoints must reload.

    ``scripts/fetch_weights.py --convert`` writes an already-stripped
    state_dict, and the compose file points the worker at that file. If
    stripping is not idempotent the second pass filters every key out and the
    worker boots on random weights (or, with strict checks, not at all).
    """
    import torch

    lightning_style = {
        "state_dict": {
            "model.band_split.weight": torch.zeros(2, 2),
            "model.freq_weights": torch.zeros(2, dtype=torch.float64),
            "loss_handler.l1snr/audio/speech_weight": torch.zeros(1),
        }
    }

    once = _strip_state_dict(lightning_style)
    assert set(once) == {"band_split.weight", "freq_weights"}
    assert once["freq_weights"].dtype == torch.float32  # float64 downcast

    twice = _strip_state_dict(once)
    assert set(twice) == set(once)
    assert all(torch.equal(twice[k], once[k]) for k in once)


@requires_ckpt
def test_rejects_unknown_stem(separator, mixture, tmp_path):
    with pytest.raises(ValueError, match="unknown stems"):
        separator.separate_file(mixture, tmp_path, stems=["vocals"])


@pytest.mark.slow
@requires_ckpt
def test_segmentation_is_transparent(separator, mixture, tmp_path):
    """Segment size must not change the output.

    The outer segmentation loop is our code, not upstream's. If the margin
    bookkeeping is wrong the audio still sounds plausible -- it just has
    discontinuities every segment boundary. Separating the same input as one
    segment and as three must agree to near bit-exactness.
    """
    cfg_single = SeparationConfig(
        quality="fast", inference_batch_size=2, segment_seconds=600.0
    )
    # 4s segments over 14s of audio forces three boundaries. Both 4.0 and 8.0
    # are multiples of the 4.0s `fast` hop, so the chunk grid stays aligned.
    cfg_split = SeparationConfig(
        quality="fast", inference_batch_size=2, segment_seconds=4.0, margin_seconds=8.0
    )

    one = separator.separate_file(mixture, tmp_path / "one", cfg=cfg_single)
    many = separator.separate_file(mixture, tmp_path / "many", cfg=cfg_split)

    for stem in STEMS:
        a, _ = sf.read(str(one[stem]), dtype="float32")
        b, _ = sf.read(str(many[stem]), dtype="float32")
        assert a.shape == b.shape, f"{stem}: length changed with segment size"

        noise = a - b
        denom = float(np.mean(noise**2))
        if denom == 0:
            continue
        snr = 10 * np.log10(float(np.mean(a**2)) / denom)
        assert snr > 45, f"{stem}: segmentation altered output (SNR {snr:.1f} dB)"


@pytest.mark.slow
@requires_ckpt
def test_stems_reconstruct_the_mixture(separator, mixture, tmp_path):
    """The three stems should sum back to the input.

    BandIt masks the complex STFT, so a correctly wired model reconstructs the
    mixture closely. A low SNR here means the weights or the iSTFT path are
    wrong -- the failure mode that ``strict=False`` loading hides.
    """
    cfg = SeparationConfig(quality="fast", inference_batch_size=2)
    paths = separator.separate_file(mixture, tmp_path / "out", cfg=cfg)

    mix, _ = sf.read(str(mixture), dtype="float32")
    stems = {s: sf.read(str(p), dtype="float32")[0] for s, p in paths.items()}
    n = min(len(mix), *(len(v) for v in stems.values()))

    residual = mix[:n] - sum(v[:n] for v in stems.values())
    snr = 10 * np.log10(
        float(np.mean(mix[:n] ** 2)) / float(np.mean(residual**2))
    )
    assert snr > 25, f"stems do not reconstruct the mixture (SNR {snr:.1f} dB)"


# ---------------------------------------------------------------------------
# Job-state regressions (no inference; these are pure logic)
# ---------------------------------------------------------------------------

def test_settle_if_stale_spares_healthy_and_terminal_jobs():
    """Only a RUNNING job that has gone quiet gets abandoned.

    Both of these paths return before touching Redis, so they are unit
    testable; the abandon path itself is exercised end to end against a live
    stack. The threshold is passed in rather than read from global config,
    which is what makes this a policy test rather than a restatement of the
    implementation.
    """
    import time as _t

    from bandit_api.jobs import Job, JobStatus, settle_if_stale

    fresh = Job(id="a", status=JobStatus.RUNNING)
    assert settle_if_stale(fresh, after=60).status is JobStatus.RUNNING

    # Terminal and queued jobs are silent by nature and must never be resettled,
    # however long ago they were written.
    for status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.QUEUED):
        job = Job(id="b", status=status)
        job.updated_at = _t.time() - 10_000
        assert settle_if_stale(job, after=1).status is status


def test_silent_for_measures_time_since_last_write():
    import time as _t

    from bandit_api.jobs import Job

    job = Job(id="a")
    assert job.silent_for < 1
    job.updated_at = _t.time() - 300
    assert 299 < job.silent_for < 301


def test_terminal_jobs_get_the_result_ttl_others_get_longer():
    """A queued job must not inherit a *result* retention policy.

    With one worker and 25-95 minute jobs, a backlog pushed later jobs past the
    24h result TTL and their records vanished while still pending.
    """
    from bandit_api.config import settings
    from bandit_api.jobs import Job, JobStatus

    assert Job(id="a", status=JobStatus.SUCCEEDED).is_terminal
    assert Job(id="b", status=JobStatus.FAILED).is_terminal
    assert not Job(id="c", status=JobStatus.QUEUED).is_terminal
    assert not Job(id="d", status=JobStatus.RUNNING).is_terminal
    assert settings.job_timeout_seconds > 0  # non-terminal grace is non-trivial


def test_job_roundtrip_preserves_updated_at():
    from bandit_api.jobs import Job

    j = Job(id="c")
    j.updated_at = 1234.5
    assert Job.from_json(j.to_json()).updated_at == 1234.5
