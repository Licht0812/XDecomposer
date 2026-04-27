"""XRD Transformer Model for Masked Reconstruction Learning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class ConvPatchEmbed(nn.Module):
    """
    Convolutional Patch Embedding layer.
    Transforms raw signal into token embeddings using 1D Convolution.
    Structure: Conv1d -> Transpose -> LayerNorm
    """
    def __init__(self, patch_len: int = 50, stride: int = 25, d_model: int = 256, padding: int = 16):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=1, 
            out_channels=d_model, 
            kernel_size=patch_len, 
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [Batch, Length] Raw XRD patterns
        Returns:
            x: [Batch, Num_Patches, D_Model]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class XRDTransformerLayer(nn.Module):
    """Single transformer encoder layer for XRD processing."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        
        x_norm = self.norm2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x

class XRDTransformerEncoder(nn.Module):
    """Multi-layer transformer encoder for XRD patterns."""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            XRDTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x

class XRDMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for XRD Pre-training.
    Masks random patches of the input XRD pattern and reconstructs the missing pixels.
    """
    def __init__(
        self,
        xrd_length: int = 3500,
        patch_len: int = 50,
        stride: int = 25,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        decoder_n_layers: int = 4,
        decoder_d_model: int = 128,
        decoder_n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        normalize_input: bool = True,
        mask_ratio: float = 0.6,
        ohem_ratio: float = 0.0,
        alpha: float = 10.0,
        lambda_cos: float = 0.5,
        lambda_deriv: float = 0.1,
    ):
        super().__init__()
        self.xrd_length = xrd_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.normalize_input = normalize_input
        self.mask_ratio = mask_ratio
        self.ohem_ratio = ohem_ratio
        
        self.alpha = alpha
        self.lambda_cos = lambda_cos
        self.lambda_deriv = lambda_deriv

        self.padding = patch_len // 2
        self.num_patches = (xrd_length + 2 * self.padding - patch_len) // stride + 1

        # Encoder
        self.patch_embed = ConvPatchEmbed(patch_len, stride, d_model, self.padding)
        self.pos_encoding = PositionalEncoding(d_model, dropout=0., max_len=self.num_patches + 1)
        self.encoder = XRDTransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)

        # Decoder
        self.decoder_d_model = decoder_d_model
        self.decoder_embed = nn.Linear(d_model, decoder_d_model, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        self.decoder_pos_encoding = PositionalEncoding(decoder_d_model, dropout=0., max_len=self.num_patches + 1)
        self.decoder = XRDTransformerEncoder(decoder_n_layers, decoder_d_model, decoder_n_heads, d_ff, dropout)
        self.decoder_pred = nn.Linear(decoder_d_model, patch_len, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform per-sample random masking."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x: Tensor, mask_ratio: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_encoding.pe[:, :x.shape[1], :]
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        x = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = x + self.decoder_pos_encoding.pe[:, :x.shape[1], :]
        x = self.decoder(x)
        x = self.decoder_pred(x)
        return x

    def normalize_xrd(self, xrd: Tensor) -> Tensor:
        if not self.normalize_input:
            return xrd
        
        # Max Scaling
        max_val = xrd.max(dim=1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-8)
        xrd_norm = xrd / max_val
        return torch.clamp(xrd_norm, 0.0, 1.0)

    def patchify(self, imgs: Tensor) -> Tensor:
        if self.padding > 0:
            imgs = F.pad(imgs, (self.padding, self.padding))
        return imgs.unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        target = self.patchify(imgs)
        
        # Avoid circular import
        from src.losses import calculate_reconstruction_loss
        loss = calculate_reconstruction_loss(
            pred, target, mask,
            alpha=self.alpha,
            lambda_cos=self.lambda_cos,
            lambda_deriv=self.lambda_deriv
        )
        return loss

    def forward(self, imgs: Tensor, mask_ratio: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        imgs = self.normalize_xrd(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
