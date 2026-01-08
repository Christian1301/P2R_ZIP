# models/p2r_head.py
# -*- coding: utf-8 -*-
"""
P2RHead Module - VERSIONE OTTIMIZZATA V9

Modifiche:
- Inizializzazione log_scale più conservativa (2.5 invece di 4.0)
  per ridurre il bias di sovrastima iniziale e facilitare il fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv_3x3(in_channels, out_channels, bn=True):
    """Blocco conv 3x3 con opzionale BatchNorm e ReLU."""
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

    # Inizializzazione Kaiming
    init.kaiming_normal_(block[0].weight, mode='fan_out', nonlinearity='relu')
    if not bn and block[0].bias is not None:
        init.constant_(block[0].bias, 0.)
    
    return block


class P2RHead(nn.Module):
    """
    Decoder P2R con inizializzazione bilanciata.
    """
    
    def __init__(
        self, 
        in_channel: int = 512,
        fea_channel: int = 64, 
        up_scale: int = 2,
        out_channel: int = 1, 
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        self.up_scale = up_scale
        self.base_stride = 8
        
        # MODIFICA V9: Inizializzazione a 2.5 (scala ~12) invece di 4.0 (scala ~55)
        # Questo riduce il rischio di esplosione del conteggio nelle prime fasi
        self.log_scale = nn.Parameter(
            torch.tensor(2.5, dtype=torch.float32), 
            requires_grad=True
        )
        
        # GroupNorm per stabilizzare le feature in input
        self.input_norm = nn.GroupNorm(32, in_channel)
        
        # Decoder
        self.layer1 = conv_3x3(in_channel, fea_channel, bn=True)
        self.layer2 = conv_3x3(fea_channel, fea_channel, bn=True)
        
        self.conv_out = nn.Conv2d(
            fea_channel,
            out_channel * (up_scale ** 2),
            kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

        # Inizializzazione output layer
        init.kaiming_normal_(self.conv_out.weight, mode='fan_out', nonlinearity='relu')
        if self.conv_out.bias is not None:
            init.constant_(self.conv_out.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizza input per stabilità
        x = self.input_norm(x)
        
        # Decoder
        h = self.layer1(x)
        h = self.layer2(h)
        out = self.conv_out(h)
        out = self.pixel_shuffle(out)

        # Output: ReLU per positività, poi scala
        scale = torch.exp(self.log_scale)
        out = torch.relu(out) * scale

        return out
    
    def get_scale(self) -> float:
        return torch.exp(self.log_scale).item()