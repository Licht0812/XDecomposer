"""Common output protocol shared by all baseline models."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import PretrainedXRDBackbone, unpatchify_1d

class BaselineBase(nn.Module):
    def __init__(
        self,
        backbone: PretrainedXRDBackbone,
        num_sources: int,
        dropout: float = 0.1,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_sources = num_sources
        self.xrd_length = backbone.xrd_length
        self.patch_len = backbone.patch_len
        self.stride = backbone.stride
        self.padding = backbone.padding
        self.num_patches = backbone.num_patches
        self.d_model = backbone.d_model
        self.output_activation = output_activation

        self.activity_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_sources),
        )

    def _activity_logits(self, z: Tensor) -> Tensor:
        return self.activity_head(z.mean(dim=1))

    def _finish(self, patch_pred: Tensor) -> Tensor:
        preds = unpatchify_1d(patch_pred, self.stride, self.padding, self.xrd_length)
        if self.output_activation == "relu":
            preds = F.relu(preds)
        elif self.output_activation == "softplus":
            preds = F.softplus(preds)
        elif self.output_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported output_activation: {self.output_activation}")
        return preds
