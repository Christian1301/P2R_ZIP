# P2R_ZIP/models/p2r_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)

class P2RHead(nn.Module):
    """
    Decoder leggero per densità (o logit punti).
    """
    def __init__(self, in_ch: int, gate: str = "multiply"):
        super().__init__()
        add = 1 if gate == "concat" else 0
        c = in_ch + add
        self.dec = nn.Sequential(
            ConvBlock(c, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
        )
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        h = self.dec(x)
        den = torch.relu(self.out(h))
        return den
