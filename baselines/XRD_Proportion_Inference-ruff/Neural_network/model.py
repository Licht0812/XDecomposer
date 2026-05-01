import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalNet(nn.Module):
    """
    Baseline model for proportion inference.
    Input: [B, 1, 3500], output: [B, 4].
    """
    def __init__(self, out_features=4):
        super(OriginalNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=8, padding=100)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=5)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=3)
        # Match the original linear input size
        self.fc1 = nn.Linear(32*25, 1024)
        self.fc2 = nn.Linear(1024, out_features)

    def forward(self, x):
        # x shape: [B, 1, 3500]
        x = F.relu(self.conv1(x)) # [B, 32, 462] (approx)
        x = F.relu(self.conv2(x)) # [B, 32, 92] (approx)
        x = F.relu(self.conv3(x)) # [B, 32, 30] (approx)

        # Resize to the expected linear input
        x = F.interpolate(x, size=25, mode='linear', align_corners=False)
        x = x.view(-1, 32*25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BaselineSeparationNet(nn.Module):
    """
    Baseline encoder-decoder for phase separation.
    Uses the same convolution settings as OriginalNet.
    Input: [B, 1, 3500], output: [B, 4, 3500].
    """
    def __init__(self, out_channels=4):
        super(BaselineSeparationNet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv1d(1, 32, kernel_size=8, stride=8, padding=100)
        self.enc2 = nn.Conv1d(32, 32, kernel_size=5, stride=5)
        self.enc3 = nn.Conv1d(32, 32, kernel_size=3, stride=3)

        # Decoder
        self.dec3 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=3)
        self.dec2 = nn.ConvTranspose1d(32, 32, kernel_size=5, stride=5)
        self.dec1 = nn.ConvTranspose1d(32, out_channels, kernel_size=8, stride=8, padding=100)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        L_orig = x.shape[2]

        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))

        # Decoder
        d3 = self.relu(self.dec3(e3))
        if d3.shape[2] != e2.shape[2]:
            d3 = F.interpolate(d3, size=e2.shape[2], mode='linear', align_corners=False)

        d2 = self.relu(self.dec2(d3))
        if d2.shape[2] != e1.shape[2]:
            d2 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=False)

        d1 = self.dec1(d2)
        if d1.shape[2] != L_orig:
            d1 = F.interpolate(d1, size=L_orig, mode='linear', align_corners=False)

        return self.relu(d1)

def get_model(name="baseline", **kwargs):
    if name == "baseline":
        return BaselineSeparationNet(**kwargs)
    elif name == "original":
        return OriginalNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
