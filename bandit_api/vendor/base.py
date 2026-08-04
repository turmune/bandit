"""Vendored from kwatcharasupat/bandit-v2 (src/models/base.py).

Upstream subclasses ``pl.LightningModule``. For inference we only need
``nn.Module``, which drops the pytorch_lightning dependency entirely.
"""

from torch import nn


class BaseEndToEndModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
