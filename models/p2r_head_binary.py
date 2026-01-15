#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2R Head Binario - Come nel paper originale

Il paper P2R usa un output a 2 canali con differenza:
- Output: [B, 2, H, W]
- Density = channel[1] - channel[0]
- Conteggio = (density > 0).sum()

Questo è fondamentalmente diverso dalla density regression:
- NON predice "quante persone per cella"
- Predice "c'è almeno una persona in questa cella?" (binario)

La P2RLoss assegna target binari basati sul matching point-to-region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Blocco conv 3x3 + BN opzionale + ReLU."""
    
    def __init__(self, in_ch, out_ch, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # Init come nel paper
        if not use_bn:
            nn.init.constant_(self.conv.bias, 0.)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class P2RHeadBinary(nn.Module):
    """
    P2R Head con output binario (2 canali + differenza).
    
    Architettura dal paper:
    - 2x Conv3x3 + ReLU
    - Conv3x3 → 2 canali
    - PixelShuffle per upsampling
    - Output finale = ch[1] - ch[0]
    
    Args:
        in_channels: canali input (dal backbone/fusion)
        hidden_channels: canali intermedi (default 64)
        upscale_factor: fattore di upsampling (default 2)
        use_bn: usa BatchNorm (default False, come paper)
    """
    
    def __init__(
        self, 
        in_channels=256, 
        hidden_channels=64, 
        upscale_factor=2,
        use_bn=False
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        out_channels = 2 * (upscale_factor ** 2)  # 2 canali * upscale^2 per PixelShuffle
        
        self.decoder = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, use_bn=use_bn),
            ConvBlock(hidden_channels, hidden_channels, use_bn=use_bn),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Init bias finale a 0
        nn.init.constant_(self.decoder[-2].bias, 0.)
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] features dal backbone
            
        Returns:
            dict con:
                - 'logits': [B, 1, H*up, W*up] - differenza ch1 - ch0 (logits per BCE)
                - 'density_binary': [B, 1, H*up, W*up] - stesso di logits (per compatibilità)
        """
        x = self.decoder(features)  # [B, 2, H*up, W*up]
        
        # Differenza come nel paper
        ch0, ch1 = x[:, 0:1], x[:, 1:2]
        logits = ch1 - ch0  # [B, 1, H*up, W*up]
        
        return {
            'logits': logits,
            'density_binary': logits,  # Alias per compatibilità
        }


class P2RHeadBinaryWithFusion(nn.Module):
    """
    P2R Head con multi-scale fusion + output binario.
    
    Combina features da multiple scale del backbone prima del decoder.
    Replica l'architettura UpSample_P2P + SimpleDecoder del paper.
    """
    
    def __init__(
        self,
        in_channels_list=[512, 512],  # Canali da ogni scala
        fuse_channels=256,
        hidden_channels=64,
        upscale_factor=2,
        use_bn=False
    ):
        super().__init__()
        
        # Fusion layer (1x1 conv per allineare canali + somma)
        self.align_layers = nn.ModuleList([
            nn.Conv2d(in_ch, fuse_channels, kernel_size=1, bias=not use_bn)
            for in_ch in in_channels_list
        ])
        
        # Fuse conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(fuse_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_bn else nn.Identity()  # Paper non usa ReLU qui
        )
        
        # Decoder binario
        self.decoder = P2RHeadBinary(
            in_channels=fuse_channels,
            hidden_channels=hidden_channels,
            upscale_factor=upscale_factor,
            use_bn=use_bn
        )
        
        # Init
        for layer in self.align_layers:
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.)
    
    def forward(self, features_list):
        """
        Args:
            features_list: lista di [B, C_i, H_i, W_i] features multi-scala
            
        Returns:
            dict dal decoder binario
        """
        # Allinea e somma features
        aligned = []
        target_size = features_list[0].shape[-2:]
        
        for feat, align in zip(features_list, self.align_layers):
            x = align(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)
            aligned.append(x)
        
        # Somma
        fused = aligned[0]
        for x in aligned[1:]:
            fused = fused + x
        
        # Fuse conv
        fused = self.fuse_conv(fused)
        
        # Decoder
        return self.decoder(fused)


# =============================================================================
# Funzioni di utilità
# =============================================================================

def count_from_binary_logits(logits, threshold=0.0):
    """
    Conta persone da logits binari.
    
    Args:
        logits: [B, 1, H, W] output del P2R head
        threshold: soglia per considerare una cella "occupata" (default 0 = prob > 0.5)
        
    Returns:
        counts: [B] tensor con conteggi per ogni immagine
    """
    # Conta celle con logit > threshold
    binary_map = (logits > threshold).float()
    counts = binary_map.sum(dim=(1, 2, 3))
    return counts


def binary_logits_to_points(logits, threshold=0.0, scale_factor=1.0):
    """
    Estrae coordinate dei punti predetti dai logits.
    
    Args:
        logits: [B, 1, H, W] output del P2R head
        threshold: soglia
        scale_factor: fattore per convertire coordinate in pixel originali
        
    Returns:
        list di [N, 2] tensors con coordinate (y, x)
    """
    B = logits.shape[0]
    points_list = []
    
    for i in range(B):
        mask = logits[i, 0] > threshold  # [H, W]
        coords = torch.nonzero(mask)  # [N, 2] - (y, x)
        
        if scale_factor != 1.0:
            coords = coords.float() * scale_factor
        
        points_list.append(coords)
    
    return points_list