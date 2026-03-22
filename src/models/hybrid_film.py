import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Optional

class ConvBlock(nn.Module):
    """Basic Convolution Block: Conv1d -> BatchNorm -> GELU"""
    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    """Basic Transposed Convolution Block: ConvTranspose1d -> BatchNorm -> GELU"""
    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, padding, output_padding=output_padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class PhaseQueryHead(nn.Module):
    """
    Latent Query Attention Mechanism.
    Outputs:
        1. Activity Logits (Classification)
        2. Latent Codes (The 'z' vector for each phase)
    """
    def __init__(self, embed_dim, num_sources, num_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_sources, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Latent Projection (Map Query to FiLM params: Gamma & Beta)
        # Output dim = 2 * embed_dim (for gamma and beta)
        self.latent_proj = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, x):
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        
        # 1. Cross Attention -> Get Refined Latent Codes 'z'
        # z: [B, K, D]
        z, _ = self.attn(query=q, key=x, value=x)
        z = self.norm(z + q)
        
        # 2. Activity Classification
        logits = self.classifier(z).squeeze(-1)
        
        # 3. Generate FiLM parameters from Latent Codes
        # params: [B, K, 2*D] -> split into gamma, beta: [B, K, D]
        film_params = self.latent_proj(z)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        return logits, gamma, beta

class HybridDemucsMAE(nn.Module):
    """
    Hybrid Architecture:
    Raw Signal -> CNN Encoder -> Linear Adapter -> MAE Transformer (Bottleneck) -> Linear Adapter -> CNN Decoder -> Output
    
    This preserves the semantic understanding of the pre-trained MAE while leveraging
    CNN's ability to handle high-resolution signal details via U-Net skip connections.
    """
    def __init__(self, 
                 mae_embed_dim=768, 
                 cnn_channels=[64, 128, 256, 512],
                 cnn_strides=None, # Auto-configured based on len(cnn_channels)
                 cnn_kernels=None, # Auto-configured based on len(cnn_channels)
                 num_sources=2,
                 xrd_length=3500,
                 use_rope=False): # Add use_rope support
        super().__init__()
        
        self.use_rope = use_rope
        
        # Auto-configure strides and kernels if not provided
        if cnn_strides is None:
            cnn_strides = [2] * len(cnn_channels)
        if cnn_kernels is None:
            cnn_kernels = [8] * len(cnn_channels)
            
        # Ensure lengths match
        if len(cnn_strides) != len(cnn_channels) or len(cnn_kernels) != len(cnn_channels):
            raise ValueError(f"Length mismatch: channels={len(cnn_channels)}, strides={len(cnn_strides)}, kernels={len(cnn_kernels)}")
        
        self.num_sources = num_sources
        self.xrd_length = xrd_length
        
        # --- 1. CNN Encoder (Downsampling) ---
        self.encoder = nn.ModuleList()
        in_c = 1
        current_len = xrd_length
        
        for out_c, stride, k in zip(cnn_channels, cnn_strides, cnn_kernels):
            pad = k // 2 - 1 if stride == 2 else k // 2 # Simple padding logic, might need tuning
            # Adjust padding for stride 2 kernel 8 -> pad 3
            if k==8 and stride==2: pad=3
            
            self.encoder.append(ConvBlock(in_c, out_c, kernel_size=k, stride=stride, padding=pad))
            in_c = out_c
            current_len = (current_len + 2*pad - k) // stride + 1
            
        self.enc_out_dim = cnn_channels[-1]
        self.enc_seq_len = current_len # Keep track for PE interpolation
        
        # --- 2. Adapter (In) ---
        self.project_in = nn.Linear(self.enc_out_dim, mae_embed_dim)
        
        # --- 3. Bottleneck (Transformer) ---
        # Container for pre-trained blocks
        self.transformer_blocks = nn.ModuleList([]) 
        self.transformer_norm = nn.LayerNorm(mae_embed_dim)
        
        # Learnable Positional Encoding for the bottleneck latent space
        # We initialize it large enough
        self.pos_embed = nn.Parameter(torch.zeros(1, self.enc_seq_len + 50, mae_embed_dim)) 
        nn.init.normal_(self.pos_embed, std=0.02)

        # --- 4. Adapter (Out) ---
        self.project_out = nn.Linear(mae_embed_dim, self.enc_out_dim)
        
        # --- 5. CNN Decoder (Upsampling + Skips) ---
        self.decoder = nn.ModuleList()
        rev_channels = list(reversed(cnn_channels))
        rev_strides = list(reversed(cnn_strides))
        rev_kernels = list(reversed(cnn_kernels))
        
        for i in range(len(rev_channels) - 1):
            k = rev_kernels[i]
            s = rev_strides[i]
            pad = k // 2 - 1 if s == 2 else k // 2
            if k==8 and s==2: pad=3
            
            # Input dim = current_layer + skip_connection
            # current: rev_channels[i] (e.g. 512)
            # skip: rev_channels[i+1] (e.g. 256)
            in_dim = rev_channels[i] + rev_channels[i+1]
            out_dim = rev_channels[i+1]
            
            self.decoder.append(
                DeconvBlock(in_dim, out_dim, kernel_size=k, stride=s, padding=pad, output_padding=0 if s==1 else 1) 
            )
        
        # Final layer
        k_last = rev_kernels[-1]
        s_last = rev_strides[-1]
        # Auto padding logic for final layer
        # Output Padding logic: (L_in - 1) * s - 2p + k + op = L_out
        # We want to match input size of first encoder layer which is 3500.
        # But 'x' here is output of last decoder block.
        
        # Simpler approach: Use same padding logic as blocks, rely on interpolate if needed.
        pad_last = k_last // 2 - 1 if s_last == 2 else k_last // 2
        if k_last==8 and s_last==2: pad_last=3
        
        # Special case for S=1 (no upsampling, just convolution) -> output_padding=0
        op = 0 if s_last == 1 else 1

        # No more skips for final layer, input is just the output of last decoder block
        self.final_deconv = nn.ConvTranspose1d(
            rev_channels[-1], num_sources, kernel_size=k_last, stride=s_last, padding=pad_last, output_padding=op
        )
        
        # --- 6. Activity Head (Latent Disentanglement via FiLM) ---
        # Instead of simple pooling, we use PhaseQueryHead to generate FiLM modulation params.
        self.activity_head = PhaseQueryHead(embed_dim=mae_embed_dim, num_sources=num_sources)

    def forward(self, x):
        # x: [B, 1, L]
        input_mix = x.clone() # Keep reference for masking
        input_len = x.shape[-1]
        
        # --- Encoder Path ---
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            
        # x: [B, C_enc, L_enc]
        
        # --- Bottleneck Path ---
        # 1. Prepare for Transformer: [B, L, C]
        x = x.permute(0, 2, 1) # [B, L_enc, C_enc]
        
        # 2. Project
        x = self.project_in(x) # [B, L_enc, D_model]
        
        # 3. Add Positional Encoding
        # Note: In HybridDemucs, we typically use learnable pos embedding for the Bottleneck.
        # But if the pretrained MAE used RoPE (no abs pos), we should probably respect that configuration?
        # Option A: Always add learnable Abs Pos in Bottleneck (Hybrid logic).
        # Option B: Use RoPE if pretrained MAE used RoPE.
        # Current implementation: ALWAYS adds learnable pos_embed.
        
        # If use_rope is True, the Transformer blocks will apply RoPE.
        # We can still add absolute pos embedding here if we want "Hybrid Pos".
        # Or disable it if we want Pure RoPE.
        # Let's keep existing logic: HybridDemucs ALWAYS has a learnable pos_embed parameter initialized.
        # But we can choose not to add it if we want strict RoPE.
        
        # For now, we KEEP adding it because HybridDemucs logic explicitly added a NEW learnable embedding
        # tailored for the bottleneck space (175 length), different from MAE's sinusoidal (3500/patch length).
        
        seq_len = x.shape[1]
        if seq_len <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate if needed
            pe = self.pos_embed.permute(0, 2, 1) # [1, D, MaxL]
            pe = F.interpolate(pe, size=seq_len, mode='linear', align_corners=False)
            pe = pe.permute(0, 2, 1)
            x = x + pe
            
        # 4. Transformer Blocks
        for blk in self.transformer_blocks:
            x = blk(x)
        x = self.transformer_norm(x)
        
        # --- Activity Head & Latent Space Modulation (FiLM) ---
        # 1. Get Activity Logits and FiLM Parameters (Gamma, Beta)
        # x (Bottleneck Features): [B, L, D]
        activity_logits, gamma, beta = self.activity_head(x) 
        
        # 2. Latent Disentanglement via Competition
        # We want to know which phase "owns" which part of the latent feature 'x'.
        # We compute a similarity map between Latent Codes (Gamma) and Features (x).
        # Gamma: [B, K, D], x: [B, L, D] -> Similarity: [B, K, L]
        
        # We use the predicted 'gamma' as the representation of the phase in latent space.
        relevance_scores = torch.matmul(gamma, x.transpose(1, 2))
        
        # Softmax Competition: Forces features to belong to specific phases
        # "latent_mask": [B, K, L]
        latent_mask = torch.softmax(relevance_scores / (x.shape[-1] ** 0.5), dim=1)
        
        # Weight by Activity Probability (Soft Gating)
        act_probs = torch.sigmoid(activity_logits).unsqueeze(-1) # [B, K, 1]
        effective_mask = latent_mask * act_probs
        
        # 3. Apply FiLM (Feature-wise Linear Modulation)
        # We aggregate the modulation parameters based on the mask.
        # Ideally, we would split 'x' into K streams, modulate each, and decode.
        # To keep it efficient (single decoder), we modulate 'x' by the *weighted sum* of active phases.
        
        # Aggregate Gamma/Beta for the single stream:
        # Gamma [B, K, D] * Mask [B, K, L] -> Sum over K -> [B, D, L] -> Transpose -> [B, L, D]
        # This creates a spatially-varying modulation map!
        
        global_gamma = torch.matmul(effective_mask.transpose(1, 2), gamma) # [B, L, K] @ [B, K, D] -> [B, L, D]
        global_beta = torch.matmul(effective_mask.transpose(1, 2), beta)   # [B, L, D]
        
        # Apply Modulation: x_new = x * (1 + gamma) + beta
        x_modulated = x * (1 + global_gamma) + global_beta
        
        # 5. Project back
        x = self.project_out(x_modulated)

        
        # 6. Prepare for CNN: [B, C, L]
        x = x.permute(0, 2, 1) # [B, C_enc, L_enc]
        
        # --- Decoder Path ---
        # skips = skips[::-1] # Do NOT reverse if using negative indexing from end
        
        # We start decoding. First input to decoder is the Bottleneck output.
        # But wait, U-Net usually concatenates the *input* of the corresponding encoder layer?
        # Standard U-Net: Encoder1 -> Encoder2 -> Bottleneck -> Decoder2(cat Encoder2) -> Decoder1(cat Encoder1)
        # Here:
        # Encoder output is skips[-1]. 
        # Bottleneck takes skips[-1] and processes it.
        # So the first Decoder block should take (Processed_Bottleneck + Skip_N-1)?
        # Actually, in Demucs, the bottleneck *is* the deepest layer. 
        # The first decoder layer corresponds to the last encoder layer.
        # Let's align indices carefully.
        # skips[0]: output of enc_layer_0 (shallowest)
        # skips[-1]: output of enc_layer_last (deepest, input to bottleneck)
        
        # The first decode step usually upsamples from bottleneck depth to (depth-1).
        # So we should concat with skips[-2].
        
        # Let's loop.
        # Decoder has len(cnn_channels) - 1 blocks + final layer.
        
        for i, block in enumerate(self.decoder):
            # We want skip from layer N-2-i
            # i=0 -> want skip[-2]
            # i=1 -> want skip[-3]
            skip_idx = 1 + i # Index in reversed skips (skips[::-1])
            # skips_rev = [skip_last, skip_last-1, ...]
            # skip_target = skips_rev[skip_idx]
            
            # Correction: 
            # We already have 'x' which is the processed version of skips[-1].
            # We want to upsample it and combine with skips[-2].
            
            target_skip = skips[-(i + 2)]
            
            # Check length mismatch due to padding/striding
            if x.shape[2] != target_skip.shape[2]:
                x = F.interpolate(x, size=target_skip.shape[2], mode='linear', align_corners=False)
            
            x = torch.cat([x, target_skip], dim=1)
            x = block(x)
            
        # Final layer
        x = self.final_deconv(x)
        
        # Final size check
        if x.shape[2] != self.xrd_length:
            x = F.interpolate(x, size=self.xrd_length, mode='linear', align_corners=False)
            
        # --- Output Masking ---
        # Instead of direct regression, we predict a mask applied to the input mixture.
        # This enforces Pred <= Input (physically consistent for XRD) and prevents leakage into zero-background regions.
        # x is the raw decoder output (logits for mask).
        
        mask = torch.sigmoid(x) # [B, K, L] in range (0, 1)
        
        # Ensure input_mix has correct shape for broadcasting
        # input_mix: [B, 1, L]
        if input_mix.shape[2] != self.xrd_length:
             input_mix = F.interpolate(input_mix, size=self.xrd_length, mode='linear', align_corners=False)
             
        out = input_mix * mask # [B, K, L]
            
        return out, activity_logits

def build_hybrid_model(pretrained_mae, num_sources=2, cnn_channels=[64, 128, 256, 512], cnn_kernels=None, cnn_strides=None):
    """
    Factory function to build HybridDemucsMAE and load pretrained weights.
    """
    # Check if pretrained MAE has use_rope attribute
    use_rope = getattr(pretrained_mae, 'use_rope', False)
    
    # 1. Create model
    model = HybridDemucsMAE(
        mae_embed_dim=pretrained_mae.d_model,
        cnn_channels=cnn_channels,
        cnn_kernels=cnn_kernels,
        cnn_strides=cnn_strides,
        num_sources=num_sources,
        xrd_length=pretrained_mae.xrd_length,
        use_rope=use_rope
    )
    
    # 2. Transfer weights
    # We copy the Encoder blocks from MAE
    # pretrained_mae.encoder is XRDTransformerEncoder, which has .layers
    model.transformer_blocks = copy.deepcopy(pretrained_mae.encoder.layers)
    
    # Copy normalization layer if exists/compatible, or just init
    # MAE encoder doesn't have a final norm in the class definition I saw?
    # Let's check src/models/transformer.py
    # XRDTransformerEncoder loop: `for layer in self.layers: x = layer(x)`
    # It returns raw x.
    # So we should probably keep model.transformer_norm as initialized (LayerNorm).
    
    # 3. Initialize Adapters
    # already done in __init__
    
    return model
