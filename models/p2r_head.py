# models/p2r_head.py
# -*- coding: utf-8 -*-
"""
P2RHead Module - VERSIONE CORRETTA V2

PROBLEMA RISOLTO:
La versione originale inizializzava log_scale a -1.0 (scala=0.37) e lo clampava
a [-1.5, 1.0] (scala max=2.7). Questo impediva al modello di predire conteggi
nell'ordine delle centinaia.

SOLUZIONE:
- log_scale inizializzato a 4.0 (scala ~55)
- Nessun clamp restrittivo di default
- GroupNorm per stabilizzare le feature in input
- BatchNorm nei layer conv per training più stabile
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
    Decoder P2R con inizializzazione corretta per crowd counting.
    
    Modifiche rispetto all'originale:
    1. log_scale inizializzato a 4.0 (era -1.0)
    2. GroupNorm sulle feature in input per stabilità
    3. BatchNorm nei layer conv
    4. Output: ReLU * exp(log_scale)
    """
    
    def __init__(
        self, 
        in_channel: int = 512,      # Default per VGG backbone (era 128)
        fea_channel: int = 64, 
        up_scale: int = 2,          # Default 2 per upscale 2x (era 1)
        out_channel: int = 1, 
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        self.up_scale = up_scale
        self.base_stride = 8
        
        # CORREZIONE CRITICA: log_scale inizializzato MOLTO più alto
        # exp(4.0) ≈ 55, ragionevole per density counting
        # L'originale usava -1.0 → exp(-1.0) ≈ 0.37, troppo basso!
        self.log_scale = nn.Parameter(
            torch.tensor(4.0, dtype=torch.float32), 
            requires_grad=True
        )
        
        # GroupNorm per stabilizzare le feature in input
        self.input_norm = nn.GroupNorm(32, in_channel)
        
        # Decoder con BatchNorm per training più stabile
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
            init.constant_(self.conv_out.bias, 0.1)  # Bias leggermente positivo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"[P2RHead] Input: shape={tuple(x.shape)}, "
                  f"range=[{x.min().item():.4f}, {x.max().item():.4f}]")

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

        if self.debug:
            print(f"[P2RHead] Output: shape={tuple(out.shape)}, "
                  f"range=[{out.min().item():.4f}, {out.max().item():.4f}], "
                  f"scale={scale.item():.4f}")

        return out
    
    def get_scale(self) -> float:
        """Ritorna il fattore di scala corrente."""
        return torch.exp(self.log_scale).item()