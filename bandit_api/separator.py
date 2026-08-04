"""CPU separation driver for BandIt v2.

Two things this adds over upstream ``inference.py``:

1. **Device independence.** Upstream hardcodes ``.to("cuda")``.

2. **Bounded memory on long inputs.** ``StandardTensorChunkedInferenceHandler``
   accumulates every windowed chunk for every stem in RAM before folding, so
   peak memory grows linearly with input duration. At the default 8s/1s
   settings that is ~3 GB per stem-hour of stereo audio -- a feature-length
   film would need well over 100 GB. We therefore drive the handler over
   fixed-size segments with discarded context margins, which caps peak memory
   at a constant regardless of how long the input is.

   The margins (not a crossfade) are what keep segment boundaries clean: each
   segment is separated with ``margin_seconds`` of real audio on both sides and
   only the interior is kept, so every output sample was produced with full
   context.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import soundfile as sf
import torch
import torchaudio as ta

from .model import SAMPLE_RATE, STEMS, load_bandit
from .vendor.inference_handler import StandardTensorChunkedInferenceHandler

log = logging.getLogger(__name__)

ProgressCb = Callable[[float], None]

# hop_size_seconds is the dominant cost knob: the model re-processes each
# chunk_size window every hop, so compute scales with chunk/hop. Upstream's
# 8.0/1.0 means every second of audio passes through the network 8 times.
QUALITY_PRESETS: dict[str, float] = {
    "fast": 4.0,  # 2x overlap
    "balanced": 2.0,  # 4x overlap
    "best": 1.0,  # 8x overlap -- upstream default
}


@dataclass(frozen=True)
class SeparationConfig:
    quality: str = "balanced"
    chunk_size_seconds: float = 8.0
    inference_batch_size: int = 4

    # Outer segmentation, to keep peak memory independent of input duration.
    #
    # margin_seconds must be >= chunk_size_seconds for the segmented result to
    # match an unsegmented one. The RNNs are *bidirectional*, so every output
    # sample depends on its whole chunk, and overlap-add means a sample is
    # reconstructed from every chunk covering it -- a span of +/- chunk_size.
    # A smaller margin leaves the first kept samples partly reconstructed from
    # reflect-padded audio instead of real audio, which shows up as a seam at
    # every segment boundary.
    #
    # The cost is 2 * margin / segment of redundant compute (~13% at these
    # values), traded against peak RAM, which scales with segment_seconds.
    segment_seconds: float = 120.0
    margin_seconds: float = 8.0

    output_format: str = "wav"  # "wav" (PCM_16) or "flac"

    def __post_init__(self) -> None:
        if self.margin_seconds < self.chunk_size_seconds:
            raise ValueError(
                f"margin_seconds ({self.margin_seconds}) must be >= "
                f"chunk_size_seconds ({self.chunk_size_seconds}); a shorter "
                f"margin produces audible seams at segment boundaries"
            )

        # Both offsets must land on the analysis grid. The handler unfolds
        # chunks at a fixed `hop` stride from the segment start, so a segment
        # start that is not a multiple of hop shifts every chunk relative to
        # where it would sit in an unsegmented run. The masks are nonlinear in
        # chunk content, so a shifted grid yields genuinely different audio --
        # measured at ~8 dB SNR against the unsegmented result, i.e. badly
        # wrong, not subtly wrong.
        hop = self.hop_size_seconds
        for name, value in (
            ("segment_seconds", self.segment_seconds),
            ("margin_seconds", self.margin_seconds),
        ):
            if abs(value / hop - round(value / hop)) > 1e-9:
                raise ValueError(
                    f"{name} ({value}) must be an integer multiple of the "
                    f"{self.quality!r} hop size ({hop}s) to keep the chunk grid "
                    f"aligned across segments"
                )

    @property
    def hop_size_seconds(self) -> float:
        try:
            return QUALITY_PRESETS[self.quality]
        except KeyError:
            raise ValueError(
                f"unknown quality {self.quality!r}; expected one of "
                f"{sorted(QUALITY_PRESETS)}"
            ) from None


def configure_threads(n_threads: int | None = None) -> int:
    """Pin torch's intra-op thread count.

    Left alone, torch grabs every core and contends with the web server, Redis
    and the OS. Oversubscription measurably hurts throughput on RNN workloads,
    where the per-op parallelism is limited anyway.
    """
    if n_threads is None:
        n_threads = int(os.environ.get("BANDIT_THREADS", "0")) or max(
            1, (os.cpu_count() or 2) - 2
        )
    torch.set_num_threads(n_threads)
    # Interop parallelism is useless for a single sequential graph.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised; harmless
    return n_threads


class Separator:
    """Holds the model in memory across jobs. Construct once per worker."""

    def __init__(
        self,
        ckpt_path: str | Path,
        stems: Sequence[str] | None = None,
        device: str = "cpu",
        n_threads: int | None = None,
    ):
        self.device = device
        self.stems = list(stems or STEMS)
        if device == "cpu":
            self.n_threads = configure_threads(n_threads)
            log.info("torch intra-op threads: %d", self.n_threads)
        self.model = load_bandit(ckpt_path, stems=self.stems, device=device)

    def _handler(self, cfg: SeparationConfig) -> StandardTensorChunkedInferenceHandler:
        return StandardTensorChunkedInferenceHandler(
            chunk_size_seconds=cfg.chunk_size_seconds,
            hop_size_seconds=cfg.hop_size_seconds,
            inference_batch_size=cfg.inference_batch_size,
            fs=SAMPLE_RATE,
        ).to(self.device)

    @staticmethod
    def _decode_with_ffmpeg(path: Path) -> tuple[torch.Tensor, int]:
        """Decode anything libsndfile cannot open, via the ffmpeg CLI.

        torchaudio here only has the ``soundfile`` backend, which handles bare
        audio containers and nothing else -- so mp4/mov/mkv/webm, i.e. every
        video, fails to load. ffmpeg is already in the image for exactly this.

        Resampling happens inside ffmpeg so the caller never pays for a second
        pass, and ``-vn`` drops the video stream before it is ever decoded.
        """
        # Write beside the source (the data volume) rather than /tmp: a
        # feature-length decode is ~1 GB of PCM and the container's /tmp lives
        # on the overlay filesystem.
        scratch_dir = path.parent if os.access(path.parent, os.W_OK) else None
        fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=scratch_dir)
        os.close(fd)
        tmp = Path(tmp_name)

        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(path),
                    "-vn",
                    "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    str(tmp),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or tmp.stat().st_size == 0:
                raise RuntimeError(
                    f"could not decode {path.name}: no audio stream, or an "
                    f"unsupported format. ffmpeg said: "
                    f"{proc.stderr.strip()[:300] or '(nothing)'}"
                )
            audio, fs = ta.load(str(tmp))
            return audio, fs
        finally:
            tmp.unlink(missing_ok=True)

    def _load_audio(self, path: str | Path) -> tuple[torch.Tensor, int]:
        path = Path(path)
        try:
            audio, fs = ta.load(str(path))
        except Exception as exc:
            log.info("soundfile could not open %s (%s); decoding via ffmpeg",
                     path.name, type(exc).__name__)
            audio, fs = self._decode_with_ffmpeg(path)

        if fs != SAMPLE_RATE:
            audio = ta.functional.resample(audio, fs, SAMPLE_RATE)
        return audio, fs

    def separate_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        cfg: SeparationConfig | None = None,
        stems: Sequence[str] | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> dict[str, Path]:
        """Separate ``input_path``, writing one file per stem into ``output_dir``.

        Returns a mapping of stem name -> written path. Output is always at
        48 kHz, the rate every published checkpoint was trained at.
        """
        cfg = cfg or SeparationConfig()
        want = list(stems or self.stems)
        unknown = set(want) - set(self.stems)
        if unknown:
            raise ValueError(f"unknown stems {sorted(unknown)}; have {self.stems}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio, source_fs = self._load_audio(input_path)
        n_channels, n_samples = audio.shape
        handler = self._handler(cfg)

        # Set per call: the worker reuses one Separator across jobs, and jobs
        # can request different stem subsets.
        self.model.active_stems = want

        step = int(cfg.segment_seconds * SAMPLE_RATE)
        margin = int(cfg.margin_seconds * SAMPLE_RATE)

        suffix = "flac" if cfg.output_format == "flac" else "wav"
        subtype = "PCM_16"
        paths = {s: output_dir / f"{s}.{suffix}" for s in want}

        writers = {
            s: sf.SoundFile(
                str(p),
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=n_channels,
                format=suffix.upper(),
                subtype=subtype,
            )
            for s, p in paths.items()
        }

        n_segments = max(1, math.ceil(n_samples / step))

        try:
            pos = 0
            seg_index = 0
            with torch.inference_mode():
                while pos < n_samples:
                    seg_start = max(0, pos - margin)
                    seg_end = min(n_samples, pos + step + margin)
                    segment = audio[:, seg_start:seg_end].to(self.device)

                    # Report inside the segment too. A file shorter than
                    # segment_seconds is a single segment, and at these speeds
                    # that could otherwise mean an hour at 0% then 100%.
                    if progress_cb is not None:
                        base, span = seg_index / n_segments, 1.0 / n_segments

                        def on_batch(i: int, n: int, _b=base, _s=span) -> None:
                            progress_cb(min(1.0, _b + _s * (i / max(1, n))))

                        handler.progress_cb = on_batch

                    out = handler(segment[None, ...], self.model)

                    # Keep only the interior; the margins existed to give the
                    # boundary samples full receptive-field context.
                    left = pos - seg_start
                    right = left + min(step, n_samples - pos)

                    for stem in want:
                        piece = out["estimates"][stem]["audio"][0][:, left:right]
                        # Masked estimates can exceed unit scale; PCM_16 would
                        # wrap rather than clip.
                        piece = piece.clamp_(-1.0, 1.0)
                        writers[stem].write(piece.T.cpu().numpy())

                    del out, segment
                    pos += step
                    seg_index += 1

                    if progress_cb is not None:
                        progress_cb(min(1.0, pos / n_samples))
        finally:
            for w in writers.values():
                w.close()

        log.info(
            "separated %s (%.1fs @ %dHz -> %dHz, quality=%s) -> %s",
            Path(input_path).name,
            n_samples / SAMPLE_RATE,
            source_fs,
            SAMPLE_RATE,
            cfg.quality,
            output_dir,
        )
        return paths
