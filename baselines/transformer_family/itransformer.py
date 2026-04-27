"""Latent-space iTransformer baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaselineBase
from .backbone import PretrainedXRDBackbone


class ITransformerBaseline(BaselineBase):
    """Treat MAE latent channels as variate tokens and model them with self-attention."""

    def __init__(
        self,
        backbone: PretrainedXRDBackbone,
        num_sources: int = 4,
        itransformer_dim: int = 256,
        itransformer_layers: int = 2,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        output_activation: str = "relu",
    ) -> None:
        super().__init__(backbone, num_sources, dropout, output_activation)
        d_ff = d_ff or itransformer_dim * 4

        self.to_variate_tokens = nn.Linear(self.num_patches, itransformer_dim)
        self.variate_embed = nn.Parameter(torch.zeros(1, self.d_model, itransformer_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=itransformer_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=itransformer_layers)
        self.to_patch_tokens = nn.Linear(itransformer_dim, self.num_patches)
        self.source_embed = nn.Parameter(torch.zeros(num_sources, self.d_model))
        self.patch_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.patch_len),
        )
        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.variate_embed, std=0.02)
        nn.init.normal_(self.source_embed, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.backbone(x)
        u = z.transpose(1, 2)
        u = self.to_variate_tokens(u) + self.variate_embed
        u = self.blocks(u)
        z_modeled = self.to_patch_tokens(u).transpose(1, 2)

        h = z_modeled[:, None, :, :] + self.source_embed[None, :, None, :]
        patch_pred = self.patch_head(h)
        return self._finish(patch_pred), self._activity_logits(z)
