"""Shared pretrained MAE encoder backbone for baseline models."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.xrd_transformer import XRDMaskedAutoencoder

def strip_module_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_mae_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint, "config": {}}

def merge_mae_config(checkpoint: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(checkpoint.get("config", {}) or {})
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config

def unpatchify_1d(
    patches: Tensor,
    stride: int,
    padding: int,
    xrd_length: int,
) -> Tensor:
    """Overlap-add reconstruction for patch tensors.

    Args:
        patches: [B, P, patch_len] or [B, K, P, patch_len].
    Returns:
        [B, L] or [B, K, L] matching the input rank.
    """
    original_dim = patches.dim()
    if original_dim == 4:
        batch_size, num_sources, num_patches, patch_len = patches.shape
        patches = patches.reshape(batch_size * num_sources, num_patches, patch_len)
    elif original_dim == 3:
        batch_size, num_patches, patch_len = patches.shape
        num_sources = None
    else:
        raise ValueError(f"Expected patches with 3 or 4 dims, got {patches.shape}")

    total_padded_len = (num_patches - 1) * stride + patch_len
    fold_input = patches.transpose(1, 2)
    output = F.fold(
        fold_input,
        output_size=(1, total_padded_len),
        kernel_size=(1, patch_len),
        stride=(1, stride),
    ).squeeze(2).squeeze(1)
    counts = F.fold(
        torch.ones_like(fold_input),
        output_size=(1, total_padded_len),
        kernel_size=(1, patch_len),
        stride=(1, stride),
    ).squeeze(2).squeeze(1)
    output = output / counts.clamp_min(1e-8)

    if padding > 0:
        output = output[:, padding:-padding]

    if output.shape[-1] > xrd_length:
        output = output[:, :xrd_length]
    elif output.shape[-1] < xrd_length:
        output = F.pad(output, (0, xrd_length - output.shape[-1]))

    if original_dim == 4:
        output = output.reshape(batch_size, num_sources, xrd_length)
    return output

class PretrainedXRDBackbone(nn.Module):
    """Deterministic MAE encoder-side feature extractor."""

    def __init__(
        self,
        mae_ckpt_path: str,
        xrd_length: Optional[int] = None,
        patch_len: Optional[int] = None,
        stride: Optional[int] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if not mae_ckpt_path:
            raise ValueError("mae_ckpt_path is required for baseline backbone initialization")

        checkpoint = load_mae_checkpoint(mae_ckpt_path)
        config = merge_mae_config(
            checkpoint,
            {
                "xrd_length": xrd_length,
                "patch_len": patch_len,
                "stride": stride,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "dropout": dropout,
            },
        )

        state_dict = strip_module_prefix(checkpoint["model_state_dict"])
        inferred_d_ff = None
        ffn_key = "encoder.layers.0.feed_forward.0.weight"
        if ffn_key in state_dict:
            inferred_d_ff = int(state_dict[ffn_key].shape[0])

        self.xrd_length = int(config.get("xrd_length", 3500))
        self.patch_len = int(config.get("patch_len", 50))
        self.stride = int(config.get("stride", 25))
        self.padding = self.patch_len // 2
        self.d_model = int(config.get("d_model", 768))
        self.n_heads = int(config.get("n_heads", 12))
        self.n_layers = int(config.get("n_layers", 4))
        self.d_ff = int(config.get("d_ff", inferred_d_ff or 1024))
        self.dropout = float(config.get("dropout", 0.1))
        self.num_patches = (self.xrd_length + 2 * self.padding - self.patch_len) // self.stride + 1

        mae = XRDMaskedAutoencoder(
            xrd_length=self.xrd_length,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            decoder_d_model=int(config.get("decoder_dim", 512)),
            decoder_n_layers=int(config.get("decoder_layers", 4)),
            decoder_n_heads=int(config.get("decoder_heads", 8)),
            d_ff=self.d_ff,
            dropout=self.dropout,
        )

        missing, unexpected = mae.load_state_dict(state_dict, strict=False)
        critical_missing = [
            key
            for key in missing
            if key.startswith("patch_embed.") or key.startswith("pos_encoding.") or key.startswith("encoder.")
        ]
        if critical_missing:
            raise RuntimeError(f"MAE checkpoint misses encoder-side keys: {critical_missing[:8]}")
        if unexpected:
            unexpected_encoder = [
                key
                for key in unexpected
                if key.startswith("patch_embed.") or key.startswith("pos_encoding.") or key.startswith("encoder.")
            ]
            if unexpected_encoder:
                raise RuntimeError(f"Unexpected encoder-side keys in MAE checkpoint: {unexpected_encoder[:8]}")

        self.patch_embed = copy.deepcopy(mae.patch_embed)
        self.pos_encoding = copy.deepcopy(mae.pos_encoding)
        self.encoder = copy.deepcopy(mae.encoder)
        self.normalize_input = bool(getattr(mae, "normalize_input", True))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def normalize_xrd(self, x: Tensor) -> Tensor:
        if not self.normalize_input:
            return x
        max_val = x.max(dim=1, keepdim=True)[0].clamp_min(1e-8)
        return torch.clamp(x / max_val, 0.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B, 1, L] input, got {x.shape}")
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected [B, L] or [B, 1, L] input, got {x.shape}")

        x = self.normalize_xrd(x)
        z = self.patch_embed(x)
        z = z + self.pos_encoding.pe[:, : z.shape[1], :].to(dtype=z.dtype, device=z.device)
        z = self.encoder(z)
        return z
