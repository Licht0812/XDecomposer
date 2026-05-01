"""Canonical Transformer encoder-decoder baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaselineBase
from .backbone import PretrainedXRDBackbone

class TransformerBaseline(BaselineBase):
    """MAE encoder memory + learned source/patch queries + direct source decoder."""

    def __init__(
        self,
        backbone: PretrainedXRDBackbone,
        num_sources: int = 4,
        decoder_layers: int = 4,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        output_activation: str = "softplus",
    ) -> None:
        if output_activation == "relu":
            output_activation = "softplus"
        super().__init__(backbone, num_sources, dropout, output_activation)
        d_ff = d_ff or self.d_model * 4

        self.source_embed = nn.Parameter(torch.zeros(num_sources, self.d_model))
        self.decoder_pos = nn.Parameter(torch.zeros(self.num_patches, self.d_model))
        self.memory_condition = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.source_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, self.patch_len),
        )
        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.source_embed, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)
        nn.init.zeros_(self.memory_condition[-1].weight)
        nn.init.zeros_(self.memory_condition[-1].bias)
        nn.init.xavier_uniform_(self.source_head[-1].weight)
        nn.init.constant_(self.source_head[-1].bias, 0.01)

    def _prepare_mixture(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B, 1, L] input, got {x.shape}")
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected [B, L] or [B, 1, L] input, got {x.shape}")
        return self.backbone.normalize_xrd(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mixture = self._prepare_mixture(x)
        z = self.backbone(mixture)
        batch_size = z.shape[0]

        condition = self.memory_condition(z.mean(dim=1))
        query = self.source_embed[:, None, :] + self.decoder_pos[None, :, :]
        query = query.reshape(1, self.num_sources * self.num_patches, self.d_model)
        query = query.expand(batch_size, -1, -1) + condition[:, None, :]

        h = self.decoder(tgt=query, memory=z)
        h = h.reshape(batch_size, self.num_sources, self.num_patches, self.d_model)
        patch_pred = self.source_head(h)
        preds = self._finish(patch_pred)
        return preds, self._activity_logits(z)
