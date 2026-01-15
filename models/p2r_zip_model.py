#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2R-ZIP Model V2 - Con P2R Head Binario

Questa versione usa il P2R head binario come nel paper originale:
- Output: logits binari (persona sì/no per cella)
- Conteggio: (logits > 0).sum()
- Loss: BCE con matching point-to-region

Il modello mantiene la struttura ZIP per Stage 1, ma Stage 2/3 usano
l'output binario per il counting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import heads
from models.p2r_head_binary import P2RHeadBinary


class VGG16BNBackbone(nn.Module):
    """
    Backbone VGG16-BN con estrazione features multi-scala.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        
        # Split in stage (come nel paper P2R)
        # Stage indices per VGG16-BN: 0-32 (conv1-4), 33-42 (conv5)
        self.stage1 = nn.Sequential(*features[:33])  # Fino a pool4
        self.stage2 = nn.Sequential(*features[33:43])  # Conv5
        
        self.out_channels = [512, 512]  # Canali output di ogni stage
    
    def forward(self, x):
        """
        Returns:
            features_list: lista di tensori multi-scala
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        return [f1, f2]


class FusionLayer(nn.Module):
    """
    Layer di fusione multi-scala (come UpSample_P2P nel paper).
    """
    
    def __init__(self, in_channels_list, out_channels, use_bn=False):
        super().__init__()
        
        self.align_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=not use_bn)
            for in_ch in in_channels_list
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            # Paper non usa ReLU dopo fusion
        )
        
        # Init
        for layer in self.align_layers:
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.)
    
    def forward(self, features_list):
        target_size = features_list[0].shape[-2:]
        
        aligned = []
        for feat, align in zip(features_list, self.align_layers):
            x = align(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)
            aligned.append(x)
        
        fused = sum(aligned)
        return self.fuse(fused)


class ZIPHead(nn.Module):
    """
    ZIP Head per Stage 1 (invariato).
    """
    
    def __init__(
        self,
        in_channels=256,
        hidden_channels=128,
        num_classes=2,
        lambda_scale=1.2,
        lambda_max=8.0,
        use_softplus=True,
        lambda_noise_std=0.0
    ):
        super().__init__()
        
        self.lambda_scale = lambda_scale
        self.lambda_max = lambda_max
        self.use_softplus = use_softplus
        self.lambda_noise_std = lambda_noise_std
        
        # Pi head (classificazione occupato/vuoto)
        self.pi_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )
        
        # Lambda head (conteggio atteso)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )
    
    def forward(self, features):
        logit_pi = self.pi_head(features)
        
        raw_lambda = self.lambda_head(features)
        
        if self.use_softplus:
            lam = F.softplus(raw_lambda) * self.lambda_scale
        else:
            lam = torch.exp(raw_lambda.clamp(max=3.0)) * self.lambda_scale
        
        lam = lam.clamp(max=self.lambda_max)
        
        if self.training and self.lambda_noise_std > 0:
            noise = torch.randn_like(lam) * self.lambda_noise_std
            lam = (lam + noise).clamp(min=0.01)
        
        return {
            'logit_pi': logit_pi,
            'lambda': lam
        }


class P2R_ZIP_Model_V2(nn.Module):
    """
    Modello P2R-ZIP V2 con P2R head binario.
    
    Architettura:
    - Backbone: VGG16-BN
    - Fusion: Multi-scale feature fusion
    - ZIP Head: Per Stage 1 (classificazione blocchi)
    - P2R Head: Output binario per Stage 2/3
    
    Args:
        backbone_name: nome backbone (solo 'vgg16_bn' supportato)
        fusion_channels: canali dopo fusion (default 256)
        pi_thresh: soglia per maschera ZIP (default 0.3)
        use_ste_mask: usa Straight-Through Estimator per maschera
        zip_head_kwargs: kwargs per ZIP head
    """
    
    def __init__(
        self,
        backbone_name='vgg16_bn',
        fusion_channels=256,
        pi_thresh=0.3,
        use_ste_mask=True,
        zip_head_kwargs=None
    ):
        super().__init__()
        
        self.pi_thresh = pi_thresh
        self.use_ste_mask = use_ste_mask
        
        # Backbone
        self.backbone = VGG16BNBackbone(pretrained=True)
        
        # Fusion
        self.fusion = FusionLayer(
            in_channels_list=self.backbone.out_channels,
            out_channels=fusion_channels,
            use_bn=False
        )
        
        # ZIP Head
        zip_kwargs = zip_head_kwargs or {}
        self.zip_head = ZIPHead(
            in_channels=fusion_channels,
            **zip_kwargs
        )
        
        # P2R Head BINARIO
        self.p2r_head = P2RHeadBinary(
            in_channels=fusion_channels,
            hidden_channels=64,
            upscale_factor=2,
            use_bn=False
        )
        
        # Downsampling factor totale
        # VGG16: /32, fusion mantiene, P2R upscale x2 → /16
        self.down_factor = 16
    
    def forward(self, x, return_all=True):
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] input images
            return_all: se True, restituisce tutti gli output
            
        Returns:
            dict con:
                - 'p2r_logits': [B, 1, H/16, W/16] logits binari
                - 'logit_pi_maps': [B, 2, H/32, W/32] logits ZIP pi
                - 'lambda_maps': [B, 1, H/32, W/32] lambda ZIP
                - 'zip_mask': [B, 1, H/16, W/16] maschera soft/hard
        """
        # Backbone
        features_list = self.backbone(x)
        
        # Fusion
        fused = self.fusion(features_list)
        
        # ZIP head
        zip_out = self.zip_head(fused)
        logit_pi = zip_out['logit_pi']
        lam = zip_out['lambda']
        
        # P2R head (binario)
        p2r_out = self.p2r_head(fused)
        p2r_logits = p2r_out['logits']
        
        # Calcola maschera ZIP
        pi_prob = torch.softmax(logit_pi, dim=1)[:, 1:2]  # Prob "occupato"
        
        if self.use_ste_mask:
            # Hard mask con STE
            hard_mask = (pi_prob > self.pi_thresh).float()
            zip_mask = hard_mask - pi_prob.detach() + pi_prob
        else:
            # Soft mask
            zip_mask = pi_prob
        
        # Upsample mask per matchare P2R output
        if zip_mask.shape[-2:] != p2r_logits.shape[-2:]:
            zip_mask = F.interpolate(
                zip_mask, 
                size=p2r_logits.shape[-2:], 
                mode='nearest'
            )
        
        outputs = {
            'p2r_logits': p2r_logits,
            'logit_pi_maps': logit_pi,
            'lambda_maps': lam,
            'zip_mask': zip_mask,
        }
        
        # Output mascherato opzionale
        outputs['p2r_logits_masked'] = p2r_logits * zip_mask
        
        return outputs
    
    def count(self, x, use_mask=False, threshold=0.0):
        """
        Conta persone in un'immagine.
        
        Args:
            x: [B, 3, H, W] input
            use_mask: applica maschera ZIP
            threshold: soglia per logits (default 0 = prob > 0.5)
            
        Returns:
            counts: [B] tensor con conteggi
        """
        outputs = self.forward(x)
        
        if use_mask:
            logits = outputs['p2r_logits_masked']
        else:
            logits = outputs['p2r_logits']
        
        counts = (logits > threshold).float().sum(dim=(1, 2, 3))
        return counts


# =============================================================================
# Factory function
# =============================================================================

def create_p2r_zip_model_v2(config):
    """
    Crea modello P2R-ZIP V2 da config.
    """
    model_cfg = config.get('MODEL', {})
    zip_head_cfg = config.get('ZIP_HEAD', {})
    
    return P2R_ZIP_Model_V2(
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        fusion_channels=256,
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.3),
        use_ste_mask=model_cfg.get('USE_STE_MASK', True),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
            'lambda_noise_std': zip_head_cfg.get('LAMBDA_NOISE_STD', 0.0),
        }
    )
P2R_ZIP_Model = P2R_ZIP_Model_V2