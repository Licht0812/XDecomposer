"""PatchTST-style encoder-only baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaselineBase
from .backbone import PretrainedXRDBackbone


class PatchTSTBaseline(BaselineBase):
    """Use the pretrained MAE patch encoder directly with a source-wise MLP head."""

    def __init__(
        self,
        backbone: PretrainedXRDBackbone,
        num_sources: int = 4,
        dropout: float = 0.1,
        output_activation: str = "relu",
    ) -> None:
        super().__init__(backbone, num_sources, dropout, output_activation)
        self.source_embed = nn.Parameter(torch.zeros(num_sources, self.d_model))
        self.patch_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.patch_len),
        )
        nn.init.normal_(self.source_embed, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.backbone(x)
        h = z[:, None, :, :] + self.source_embed[None, :, None, :]
        patch_pred = self.patch_head(h)
        return self._finish(patch_pred), self._activity_logits(z)
