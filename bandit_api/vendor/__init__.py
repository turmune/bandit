"""Vendored subset of https://github.com/kwatcharasupat/bandit-v2 (Apache-2.0).

Only the modules required for inference are included. Upstream changes:

* ``base.py``              -- ``pl.LightningModule`` -> ``nn.Module``
* ``utils.py``             -- ``librosa.{hz_to_midi,midi_to_hz}`` -> local numpy
                              equivalents (drops librosa/numba/llvmlite)
* ``inference_handler.py`` -- streaming/file handlers and the ``torchaudio.io
                              .StreamReader`` import removed (StreamReader is
                              gone in torchaudio >= 2.2); tqdm progress bar
                              replaced with an optional ``progress_cb``; honours
                              the model's ``active_stems`` filter; debug
                              ``print`` in ``_fold`` removed
* ``tfmodel.py``,
  ``bandsplit.py``,
  ``maskestim.py``         -- ``checkpoint_sequential`` bypassed when grad is
                              disabled (it is a training-time memory/compute
                              trade with nothing to recompute at inference)
* ``bandit.py``            -- ``separate()`` honours an optional
                              ``active_stems`` list so callers can decode a
                              subset; relative import of ``base`` fixed for the
                              flattened layout

See LICENSE-bandit-v2 at the repo root.
"""
