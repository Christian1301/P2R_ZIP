# -*- coding: utf-8 -*-
# ============================================================
# P2RHead Module
# "Point-to-Region Supervision for Crowd Counting" (2023)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def conv_3x3(in_channels, out_channels, bn=False):
    padding = 1
    layers = [
        nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=padding,
            bias=not bn
        )
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    block = nn.Sequential(*layers)

    if not bn:
        init.constant_(block[0].bias, 0.)
    return block

class P2RHead(nn.Module):
    """
    Decoder P2R conforme al paper:
    - PixelShuffle per l'upsampling
    - Sigmoid finale per produrre mappa di densità normalizzata [0,1]
    """
    def __init__(self, in_channel=128, fea_channel=64, up_scale=1, out_channel=1, debug=False):
        super().__init__()
        self.debug = debug
        self.up_scale = up_scale
        self.base_stride = 8  
        self.log_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        self.layer1 = conv_3x3(in_channel, fea_channel, bn=False)
        self.layer2 = conv_3x3(fea_channel, fea_channel, bn=False)
        self.conv_out = nn.Conv2d(
            fea_channel,
            out_channel * (up_scale ** 2),
            kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

        init.constant_(self.conv_out.bias, 0.)

    def forward(self, x):
        if self.debug:
            print(f"[P2RHead] Input shape: {tuple(x.shape)}")

        h = self.layer1(x)
        h = self.layer2(h)
        out = self.conv_out(h)
        out = self.pixel_shuffle(out)

        scale = torch.exp(self.log_scale)
        out = torch.relu(out) * scale

        if self.debug:
            print(f"[P2RHead] Output shape: {tuple(out.shape)} (upscale x{self.up_scale})")
            print(f"[P2RHead] Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
            print(f"[P2RHead] Density scale (relu): {scale.item():.6f}")
            print(f"[P2RHead] Density scale (exp): {scale.item():.6f}")

        return out