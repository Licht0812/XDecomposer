import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-GELU block."""

    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Deconv-BN-GELU block."""

    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class PhaseQueryHead(nn.Module):
    """Phase query head."""

    def __init__(self, embed_dim, num_sources, num_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_sources, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.latent_proj = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)
        latent, _ = self.attn(query=queries, key=x, value=x)
        latent = self.norm(latent + queries)
        logits = self.classifier(latent).squeeze(-1)
        gamma, beta = self.latent_proj(latent).chunk(2, dim=-1)
        return logits, gamma, beta


class XDecomposer(nn.Module):
    """FiLM-based XRD decomposition model."""

    def __init__(
        self,
        mae_embed_dim=768,
        cnn_channels=None,
        cnn_strides=None,
        cnn_kernels=None,
        num_sources=2,
        xrd_length=3500,
        use_rope=False,
        use_transformer=True,
        use_film=True,
        use_skip_connections=True,
        mask_type="soft",
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 256, 512]
        if cnn_strides is None:
            cnn_strides = [2] * len(cnn_channels)
        if cnn_kernels is None:
            cnn_kernels = [8] * len(cnn_channels)
        if len(cnn_strides) != len(cnn_channels) or len(cnn_kernels) != len(cnn_channels):
            raise ValueError(
                f"Length mismatch: channels={len(cnn_channels)}, "
                f"strides={len(cnn_strides)}, kernels={len(cnn_kernels)}"
            )

        self.use_rope = use_rope
        self.use_transformer = use_transformer
        self.use_film = use_film
        self.use_skip_connections = use_skip_connections
        self.mask_type = mask_type
        self.num_sources = num_sources
        self.xrd_length = xrd_length

        self.encoder = nn.ModuleList()
        in_channels = 1
        current_len = xrd_length
        for out_channels, stride, kernel in zip(cnn_channels, cnn_strides, cnn_kernels):
            padding = kernel // 2 - 1 if stride == 2 else kernel // 2
            if kernel == 8 and stride == 2:
                padding = 3
            self.encoder.append(
                ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            )
            in_channels = out_channels
            current_len = (current_len + 2 * padding - kernel) // stride + 1

        self.enc_out_dim = cnn_channels[-1]
        self.enc_seq_len = current_len

        self.project_in = nn.Linear(self.enc_out_dim, mae_embed_dim)
        self.transformer_blocks = nn.ModuleList([])
        self.transformer_norm = nn.LayerNorm(mae_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.enc_seq_len + 50, mae_embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.project_out = nn.Linear(mae_embed_dim, self.enc_out_dim)

        self.decoder = nn.ModuleList()
        rev_channels = list(reversed(cnn_channels))
        rev_strides = list(reversed(cnn_strides))
        rev_kernels = list(reversed(cnn_kernels))
        for i in range(len(rev_channels) - 1):
            kernel = rev_kernels[i]
            stride = rev_strides[i]
            padding = kernel // 2 - 1 if stride == 2 else kernel // 2
            if kernel == 8 and stride == 2:
                padding = 3
            in_dim = rev_channels[i] + (rev_channels[i + 1] if self.use_skip_connections else 0)
            out_dim = rev_channels[i + 1]
            self.decoder.append(
                DeconvBlock(
                    in_dim,
                    out_dim,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=0 if stride == 1 else 1,
                )
            )

        last_kernel = rev_kernels[-1]
        last_stride = rev_strides[-1]
        last_padding = last_kernel // 2 - 1 if last_stride == 2 else last_kernel // 2
        if last_kernel == 8 and last_stride == 2:
            last_padding = 3
        output_padding = 0 if last_stride == 1 else 1
        self.final_deconv = nn.ConvTranspose1d(
            rev_channels[-1],
            num_sources,
            kernel_size=last_kernel,
            stride=last_stride,
            padding=last_padding,
            output_padding=output_padding,
        )
        self.activity_head = PhaseQueryHead(embed_dim=mae_embed_dim, num_sources=num_sources)

    def forward(self, x):
        input_mix = x.clone()
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        x = x.permute(0, 2, 1)
        x = self.project_in(x)

        seq_len = x.shape[1]
        if seq_len <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :seq_len, :]
        else:
            pos_embed = self.pos_embed.permute(0, 2, 1)
            pos_embed = F.interpolate(pos_embed, size=seq_len, mode="linear", align_corners=False)
            x = x + pos_embed.permute(0, 2, 1)

        if self.use_transformer:
            for block in self.transformer_blocks:
                x = block(x)
        x = self.transformer_norm(x)

        activity_logits, gamma, beta = self.activity_head(x)
        if self.use_film:
            relevance_scores = torch.matmul(gamma, x.transpose(1, 2))
            if self.mask_type == "soft":
                latent_mask = torch.softmax(relevance_scores / (x.shape[-1] ** 0.5), dim=1)
            elif self.mask_type == "hard":
                hard_indices = torch.argmax(relevance_scores, dim=1, keepdim=True)
                latent_mask = torch.zeros_like(relevance_scores).scatter_(1, hard_indices, 1.0)
            elif self.mask_type == "direct":
                latent_mask = torch.ones_like(relevance_scores) / relevance_scores.shape[1]
            else:
                latent_mask = torch.softmax(relevance_scores / (x.shape[-1] ** 0.5), dim=1)

            act_probs = torch.sigmoid(activity_logits).unsqueeze(-1)
            effective_mask = latent_mask * act_probs
            global_gamma = torch.matmul(effective_mask.transpose(1, 2), gamma)
            global_beta = torch.matmul(effective_mask.transpose(1, 2), beta)
            x = x * (1 + global_gamma) + global_beta

        x = self.project_out(x)
        x = x.permute(0, 2, 1)

        for i, block in enumerate(self.decoder):
            target_skip = skips[-(i + 2)]
            if x.shape[2] != target_skip.shape[2]:
                x = F.interpolate(x, size=target_skip.shape[2], mode="linear", align_corners=False)
            if self.use_skip_connections:
                x = torch.cat([x, target_skip], dim=1)
            x = block(x)

        x = self.final_deconv(x)
        if x.shape[2] != self.xrd_length:
            x = F.interpolate(x, size=self.xrd_length, mode="linear", align_corners=False)
        if input_mix.shape[2] != self.xrd_length:
            input_mix = F.interpolate(input_mix, size=self.xrd_length, mode="linear", align_corners=False)

        if self.mask_type == "direct":
            out = F.relu(x)
        else:
            mask = torch.sigmoid(x)
            if self.mask_type == "hard":
                mask = (mask > 0.5).float()
            out = input_mix * mask

        return out, activity_logits


def build_xdecomposer(
    pretrained_mae,
    num_sources=2,
    cnn_channels=None,
    cnn_kernels=None,
    cnn_strides=None,
    use_transformer=True,
    use_film=True,
    use_skip_connections=True,
    mask_type="soft",
):
    """Build XDecomposer from a pretrained encoder."""

    if cnn_channels is None:
        cnn_channels = [64, 128, 256, 512]

    model = XDecomposer(
        mae_embed_dim=pretrained_mae.d_model,
        cnn_channels=cnn_channels,
        cnn_kernels=cnn_kernels,
        cnn_strides=cnn_strides,
        num_sources=num_sources,
        xrd_length=pretrained_mae.xrd_length,
        use_transformer=use_transformer,
        use_film=use_film,
        use_skip_connections=use_skip_connections,
        mask_type=mask_type,
        use_rope=getattr(pretrained_mae, "use_rope", False),
    )
    model.transformer_blocks = copy.deepcopy(pretrained_mae.encoder.layers)
    return model
