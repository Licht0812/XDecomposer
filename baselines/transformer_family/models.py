"""Factory and public exports for Transformer-family baseline models."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .backbone import PretrainedXRDBackbone, unpatchify_1d
from .itransformer import ITransformerBaseline
from .patchtst import PatchTSTBaseline
from .transformer import TransformerBaseline

def build_transformer_family_baseline(
    name: str,
    mae_checkpoint: str,
    num_sources: int = 4,
    xrd_length: int = 3500,
    patch_len: int = 50,
    stride: int = 25,
    d_model: Optional[int] = None,
    n_heads: Optional[int] = None,
    n_layers: Optional[int] = None,
    dropout: float = 0.1,
    freeze_backbone: bool = False,
    output_activation: str = "relu",
    transformer_decoder_layers: int = 4,
    transformer_d_ff: Optional[int] = None,
    itransformer_dim: int = 256,
    itransformer_layers: int = 2,
    itransformer_heads: int = 8,
    itransformer_d_ff: Optional[int] = None,
) -> nn.Module:
    backbone = PretrainedXRDBackbone(
        mae_ckpt_path=mae_checkpoint,
        xrd_length=xrd_length,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        freeze=freeze_backbone,
    )

    name = name.lower()
    if name == "transformer":
        return TransformerBaseline(
            backbone=backbone,
            num_sources=num_sources,
            decoder_layers=transformer_decoder_layers,
            n_heads=n_heads or backbone.n_heads,
            d_ff=transformer_d_ff,
            dropout=dropout,
            output_activation=output_activation,
        )
    if name == "itransformer":
        return ITransformerBaseline(
            backbone=backbone,
            num_sources=num_sources,
            itransformer_dim=itransformer_dim,
            itransformer_layers=itransformer_layers,
            n_heads=itransformer_heads,
            d_ff=itransformer_d_ff,
            dropout=dropout,
            output_activation=output_activation,
        )
    if name == "patchtst":
        return PatchTSTBaseline(
            backbone=backbone,
            num_sources=num_sources,
            dropout=dropout,
            output_activation=output_activation,
        )
    raise ValueError(f"Unknown baseline '{name}'. Choose from transformer, itransformer, patchtst.")

# Backward-compatible short alias used by existing train/eval runners.
build_baseline = build_transformer_family_baseline

__all__ = [
    "ITransformerBaseline",
    "PatchTSTBaseline",
    "PretrainedXRDBackbone",
    "TransformerBaseline",
    "build_baseline",
    "build_transformer_family_baseline",
    "unpatchify_1d",
]
