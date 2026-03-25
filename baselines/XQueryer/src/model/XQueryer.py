import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
import torch.fft

class Xmodel(nn.Module):

    def __init__(self, embed_dim=3500, nhead=5, num_encoder_layers=3, dim_feedforward=768,
                 dropout=0., activation="relu", num_slots=4, feature_dim=256, num_classes=100315):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.conv = ConvModule(drop_rate=dropout)

        # Learnable slot tokens (queries for each potential phase)
        self.slot_tokens = nn.Parameter(torch.zeros(1, num_slots, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, dim_feedforward))

        # -------------encoder----------------
        sa_layer = CrossAttnLayer(embed_dim, nhead, dim_feedforward, dropout, activation)
        self.encoder = SelfAttnModule(sa_layer, num_encoder_layers,)
        # ------------------------------------

        self.norm_after = nn.LayerNorm(embed_dim)

        # Heads for each slot
        self.xrd_head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, embed_dim),
            nn.Sigmoid() # XRD intensity is normalized [0, 1]
        )
        
        self.ratio_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Softmax(dim=1) # Ratios should sum to 1 across slots? 
            # Or Sigmoid if we treat them independently and normalize later
        )

        self.feature_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, feature_dim)
        )

        # Classification head for training slot features (not used during inference retrieval)
        self.feat_cls_head = nn.Linear(feature_dim, num_classes)

        self._reset_parameters()
        self.init_weights()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights(self):
        trunc_normal_(self.slot_tokens, std=.02)

        self.pos_embed.requires_grad = False

        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.array(range(self.pos_embed.shape[2])))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).T.unsqueeze(0))

    def forward(self, x, elem):
        # x: (N, 3500)
        x = x[:, :3500]
        N = x.shape[0]
        
        sampling_rate = 1.0 
        x = x.unsqueeze(1) # N*1*3500
        
        # Multi-scale and Frequency filtering
        x1 = self.conv(x) 
        x2 = self.conv(SignalProcessor(x, sampling_rate).filter_high_frequencies(percentage=0.3))
        x3 = self.conv(SignalProcessor(x, sampling_rate).filter_high_frequencies(percentage=0.6))
        x4 = self.conv(SignalProcessor(x, sampling_rate).filter_high_frequencies(percentage=0.9))

        x = torch.cat((x1, x2, x3, x4,), dim=1) # N*768*3500

        # Encoder expects (L, N, E)
        # Here x is (N, 768, 3500), we treat 768 as the sequence length L? 
        # Actually in original code: x = x.permute(1, 0, 2).contiguous() -> (768, N, 3500)
        # So E = 3500, L = 768
        x = x.permute(1, 0, 2).contiguous() # 768*N*3500

        pos_embed = self.pos_embed.permute(2, 0, 1).contiguous().repeat(1, N, 1) 

        # Prepare element info as query? No, the user wants Slots to be queries.
        # But CrossAttnLayer currently uses elem to generate queries. 
        # We need to modify CrossAttnLayer to accept our slot_tokens as queries.
        
        slots = self.slot_tokens.repeat(N, 1, 1).permute(1, 0, 2).contiguous() # num_slots, N, embed_dim

        # Modify: Cross-attention between slots (Q) and XRD features (K, V)
        # We also want to incorporate element info. 
        # Let's pass elem to the encoder as well.
        elem = elem.unsqueeze(1).permute(1, 0, 2).contiguous() # 1*N*92

        feats = self.encoder(slots, pos_embed, elem, keys=x)
        feats = self.norm_after(feats) # num_slots, N, embed_dim
        
        # feats is (num_slots, N, embed_dim), we want (N, num_slots, embed_dim)
        feats = feats.permute(1, 0, 2).contiguous()
        
        # Output for each slot
        pred_xrds = self.xrd_head(feats) # N, num_slots, 3500
        pred_ratios = self.ratio_head(feats).squeeze(-1) # N, num_slots
        pred_features = self.feature_head(feats) # N, num_slots, feature_dim
        feat_logits = self.feat_cls_head(pred_features) # N, num_slots, num_classes
     
        return {
            "xrds": pred_xrds,
            "ratios": pred_ratios,
            "features": pred_features,
            "feat_logits": feat_logits
        }



class ConvModule(nn.Module):
    def __init__(self, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=17, stride=1, padding=(17 - 1) // 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=33, stride=1, padding=(33 - 1) // 2)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=65, stride=1, padding=(65 - 1) // 2)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=129, stride=1, padding=(129 - 1) // 2)
        self.bn4 = nn.BatchNorm1d(32)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=257, stride=1, padding=(257 - 1) // 2)
        self.bn5 = nn.BatchNorm1d(32)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=513, stride=1, padding=(513 - 1) // 2)
        self.bn6 = nn.BatchNorm1d(32)
        self.act6 = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = Layer(64, 64, kernel_size=3, stride=2, downsample=True)
        # self.layer2 = Layer(64, 128, kernel_size=3, stride=2, downsample=True)
        # self.layer3 = Layer(256, 256, kernel_size=3, stride=2, downsample=True)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.act3(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.act4(x4)

        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        x5 = self.act5(x5)

        x6 = self.conv6(x)
        x6 = self.bn6(x6)
        x6 = self.act6(x6)



        #x = self.maxpool(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        # x = self.layer3(x)
        #x = self.maxpool2(x)
        return torch.cat((x1, x2, x3, x4, x5, x6), dim=1)


class SelfAttnModule(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
   

    def forward(self, src, pos, elem, keys):
        output = src

        for layer in self.layers:
            output = layer(output, pos, elem, keys)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CrossAttnLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # element_map to incorporate chemical information into the queries
        self.element_map = nn.Sequential(  
            nn.Linear(92, d_model),   
            nn.Dropout(0.5),  
            nn.ReLU(),
        )

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos, elem, keys):
        # src: Slot tokens (num_slots, N, embed_dim)
        # elem: Element embedding (1, N, 92)
        # keys: XRD features (768, N, 3500) - Wait, embed_dim is 3500.
        
        # Incorporate element info into slot tokens (Query)
        q_elem = self.element_map(elem) # 1, N, d_model
        q = src + q_elem # num_slots, N, d_model
        
        # Keys and Values are XRD features with positional embedding
        k = v = with_pos_embed(keys, pos)
        
        src2 = self.cross_attn(q, k, value=v)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class Layer(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample):
        super(Layer, self).__init__()
        self.block1 = BasicBlock(inchannel, outchannel, kernel_size=kernel_size, stride=stride, downsample=downsample)
        self.block2 = BasicBlock(outchannel, outchannel, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=2),
            nn.BatchNorm1d(outchannel)
        ) if downsample else None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out).astype(np.float32)  # (M, D/2)
    emb_cos = np.cos(out).astype(np.float32)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class SignalProcessor:
    def __init__(self, signals, sampling_rate):
        """
        Initializes the SignalProcessor with a signal and its sampling rate.

        Parameters:
        signals (torch.Tensor): The time-domain signals with shape (N, 1, 1000).
        sampling_rate (float): The sampling rate in Hz.
        """
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.frequency = torch.fft.fftfreq(signals.shape[-1], d=1/sampling_rate)
        self.fourier_transforms = torch.fft.fft(signals, dim=-1)
    
    def filter_high_frequencies(self, percentage=0.2):
        """
        Filters out the top given percentage of high frequencies from the signal.

        Parameters:
        percentage (float): The percentage of high frequencies to filter out.

        Returns:
        torch.Tensor: The filtered signals in the time domain.
        """
        n = self.signals.shape[-1]
        cutoff_index = int(n * (1 - percentage) / 2)
        filtered_fourier_transforms = self.fourier_transforms.clone()
        filtered_fourier_transforms[..., cutoff_index:-cutoff_index] = 0
        
        filtered_signals = torch.fft.ifft(filtered_fourier_transforms, dim=-1)
        return filtered_signals.real
