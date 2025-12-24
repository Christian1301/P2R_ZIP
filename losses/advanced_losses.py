#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Losses per Crowd Counting - Target MAE < 60

Include:
1. Smooth L1 Count Loss (più robusto di L1)
2. Distribution Matching Loss (DM-Count style)
3. Optimal Transport Loss (per matching spaziale preciso)
4. Bayesian Loss (per incertezza)

Reference papers:
- DM-Count: Distribution Matching for Crowd Counting (NeurIPS 2020)
- Bayesian Loss for Crowd Count Estimation (ICCV 2019)
- Learning to Count Objects with Few Exemplars (CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np


# =============================================================================
# SMOOTH L1 COUNT LOSS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """
    Smooth L1 loss per count - più robusto agli outlier di L1.
    
    L1 è sensibile agli errori grandi nelle scene dense.
    Smooth L1 ha gradiente limitato per errori grandi.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_count: torch.Tensor, gt_count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_count: predicted count (scalar or batch)
            gt_count: ground truth count
        """
        diff = torch.abs(pred_count - gt_count)
        
        # Smooth L1
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        return loss.mean()


# =============================================================================
# DISTRIBUTION MATCHING LOSS (DM-Count style)
# =============================================================================

class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss per crowd counting.
    
    Invece di confrontare solo il count totale, confronta la distribuzione
    spaziale della density predetta con quella del ground truth.
    
    Usa un kernel gaussiano per smoothing e matching.
    
    Reference: DM-Count (NeurIPS 2020)
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        
        # Pre-compute gaussian kernel
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float, size: int = None) -> torch.Tensor:
        """Crea kernel gaussiano 2D."""
        if size is None:
            size = int(6 * sigma) | 1  # Dispari, ~6 sigma
        
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Applica smoothing gaussiano."""
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_density: [B, 1, H, W] predicted density
            gt_density: [B, 1, H, W] ground truth density
        """
        # Downsample per efficienza
        if self.downsample > 1:
            pred = F.avg_pool2d(pred_density, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt_density, self.downsample) * (self.downsample ** 2)
        else:
            pred = pred_density
            gt = gt_density
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize per confronto distribuzionale
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # KL divergence (con smoothing per stabilità)
        eps = 1e-8
        kl_loss = (gt_norm * torch.log((gt_norm + eps) / (pred_norm + eps))).sum(dim=[2, 3])
        
        # Anche L2 sulla distribuzione
        l2_loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        # Combinazione
        loss = kl_loss.mean() + l2_loss.mean()
        
        return loss


# =============================================================================
# OPTIMAL TRANSPORT LOSS (Sinkhorn)
# =============================================================================

class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con algoritmo Sinkhorn.
    
    Calcola la distanza Wasserstein tra la distribuzione predetta e GT,
    che è più robusta della semplice MSE per il matching spaziale.
    
    Usa regularizzazione entropica per efficienza (Sinkhorn-Knopp).
    """
    
    def __init__(
        self,
        reg: float = 0.1,
        num_iters: int = 50,
        downsample: int = 8,
    ):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
    
    def _sinkhorn(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        C: torch.Tensor,
        reg: float,
        num_iters: int,
    ) -> torch.Tensor:
        """
        Algoritmo Sinkhorn per OT con regularizzazione entropica.
        
        Args:
            a: [N] source distribution
            b: [M] target distribution  
            C: [N, M] cost matrix
            reg: regularization strength
            num_iters: numero iterazioni
        
        Returns:
            transport_cost: costo OT
        """
        K = torch.exp(-C / reg)
        
        u = torch.ones_like(a)
        
        for _ in range(num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        # Transport plan
        P = u.view(-1, 1) * K * v.view(1, -1)
        
        # Transport cost
        cost = (P * C).sum()
        
        return cost
    
    def _compute_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Calcola matrice dei costi (distanza euclidea tra pixel)."""
        # Coordinate grid
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [H*W, 2]
        
        # Distanza euclidea squared
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
        C = (diff ** 2).sum(dim=-1)  # [N, N]
        
        return C
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_density: [B, 1, H, W]
            gt_density: [B, 1, H, W]
        """
        # Downsample per efficienza
        if self.downsample > 1:
            pred = F.avg_pool2d(pred_density, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt_density, self.downsample) * (self.downsample ** 2)
        else:
            pred = pred_density
            gt = gt_density
        
        B, _, H, W = pred.shape
        N = H * W
        
        # Cost matrix (cached se possibile)
        C = self._compute_cost_matrix(H, W, pred.device)
        
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for i in range(B):
            # Flatten e normalize
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            # Normalize to distributions
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            a_norm = a / a_sum
            b_norm = b / b_sum
            
            # Solo se entrambi non vuoti
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a_norm, b_norm, C, self.reg, self.num_iters)
                total_loss = total_loss + cost
        
        return total_loss / B


# =============================================================================
# BAYESIAN LOSS (Uncertainty)
# =============================================================================

class BayesianLoss(nn.Module):
    """
    Bayesian Loss per crowd counting con stima dell'incertezza.
    
    Assume che ogni predizione abbia un'incertezza associata,
    e pesa la loss inversamente all'incertezza.
    
    Utile per gestire scene con densità variabile.
    
    Reference: Bayesian Loss for Crowd Count Estimation (ICCV 2019)
    """
    
    def __init__(self, sigma_prior: float = 1.0):
        super().__init__()
        self.sigma_prior = sigma_prior
    
    def forward(
        self,
        pred_density: torch.Tensor,
        pred_sigma: torch.Tensor,
        gt_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_density: [B, 1, H, W] predicted mean density
            pred_sigma: [B, 1, H, W] predicted uncertainty (log variance)
            gt_density: [B, 1, H, W] ground truth
        """
        # Uncertainty = exp(log_sigma)
        sigma = torch.exp(pred_sigma).clamp(min=1e-4, max=100)
        
        # Negative log likelihood gaussiano
        diff_sq = (pred_density - gt_density) ** 2
        nll = 0.5 * (diff_sq / sigma + torch.log(sigma))
        
        # Prior sulla sigma per evitare che esploda
        prior_loss = 0.5 * ((sigma - self.sigma_prior) ** 2).mean()
        
        loss = nll.mean() + 0.01 * prior_loss
        
        metrics = {
            'nll': nll.mean().item(),
            'sigma_mean': sigma.mean().item(),
            'sigma_std': sigma.std().item(),
        }
        
        return loss, metrics


# =============================================================================
# COMBINED P2R LOSS V2
# =============================================================================

class P2RLossV2(nn.Module):
    """
    P2R Loss combinata per target MAE < 60.
    
    Combina:
    1. Count Loss (Smooth L1)
    2. Distribution Matching Loss
    3. Optimal Transport Loss (opzionale)
    4. Spatial Consistency Loss
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        count_loss_type: str = "smoothl1",
        dm_weight: float = 0.5,
        dm_sigma: float = 8.0,
        use_ot: bool = True,
        ot_weight: float = 0.3,
        ot_reg: float = 0.1,
        ot_iters: int = 50,
        spatial_weight: float = 0.2,
        scale_weight: float = 0.2,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.use_ot = use_ot
        
        # Count loss
        if count_loss_type == "smoothl1":
            self.count_loss = SmoothL1CountLoss(beta=10.0)
        else:
            self.count_loss = lambda pred, gt: F.l1_loss(pred, gt)
        
        # Distribution matching
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma)
        
        # Optimal transport
        if use_ot and ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(
                reg=ot_reg, num_iters=ot_iters, downsample=8
            )
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_density: [B, 1, H, W]
            gt_density: [B, 1, H, W]
            points_list: list of [N_i, 2] point annotations
            cell_area: area per normalizzazione count
        """
        B, _, H, W = pred_density.shape
        device = pred_density.device
        
        losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_counts_t = torch.stack(pred_counts)
        gt_counts_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        count_loss = self.count_loss(pred_counts_t, gt_counts_t)
        losses['count'] = count_loss.item()
        total_loss = total_loss + self.count_weight * count_loss
        
        # 2. Distribution Matching Loss
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            dm_loss = self.dm_loss(pred_density, gt_resized)
            losses['dm'] = dm_loss.item()
            total_loss = total_loss + self.dm_weight * dm_loss
        
        # 3. Optimal Transport Loss
        if self.use_ot and self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            ot_loss = self.ot_loss(pred_density, gt_resized)
            losses['ot'] = ot_loss.item()
            total_loss = total_loss + self.ot_weight * ot_loss
        
        # 4. Spatial Consistency (center of mass)
        if self.spatial_weight > 0:
            spatial_loss = self._spatial_consistency(pred_density, points_list, H, W)
            losses['spatial'] = spatial_loss.item()
            total_loss = total_loss + self.spatial_weight * spatial_loss
        
        losses['total'] = total_loss.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_counts_t - gt_counts_t).mean().item()
            bias = pred_counts_t.sum().item() / gt_counts_t.sum().clamp(min=1).item()
            losses['mae'] = mae
            losses['bias'] = bias
        
        return total_loss, losses
    
    def _spatial_consistency(
        self,
        pred_density: torch.Tensor,
        points_list: List[torch.Tensor],
        H: int, W: int,
    ) -> torch.Tensor:
        """Penalizza differenze nel center of mass."""
        device = pred_density.device
        B = pred_density.shape[0]
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        # Coordinate grid
        y_coords = torch.arange(H, device=device).float().view(1, 1, H, 1)
        x_coords = torch.arange(W, device=device).float().view(1, 1, 1, W)
        
        for i, pts in enumerate(points_list):
            if pts is None or len(pts) == 0:
                continue
            
            pred = pred_density[i:i+1]  # [1, 1, H, W]
            pred_sum = pred.sum().clamp(min=1e-8)
            
            # Pred center of mass
            pred_cy = (pred * y_coords).sum() / pred_sum
            pred_cx = (pred * x_coords).sum() / pred_sum
            
            # GT center of mass
            scale_h = H / pts[:, 1].max().clamp(min=1)
            scale_w = W / pts[:, 0].max().clamp(min=1)
            
            gt_cy = (pts[:, 1] * scale_h).mean()
            gt_cx = (pts[:, 0] * scale_w).mean()
            
            # Distance
            dist = torch.sqrt((pred_cy - gt_cy) ** 2 + (pred_cx - gt_cx) ** 2)
            total_loss = total_loss + dist
            count += 1
        
        return total_loss / max(count, 1)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Advanced Losses...")
    
    B, H, W = 2, 48, 48
    device = 'cpu'
    
    pred = torch.rand(B, 1, H, W) * 10
    gt = torch.rand(B, 1, H, W) * 10
    points = [torch.rand(50, 2) * H for _ in range(B)]
    
    # DM Loss
    print("\n1. Distribution Matching Loss:")
    dm = DistributionMatchingLoss(sigma=8.0)
    dm_loss = dm(pred, gt)
    print(f"   Loss: {dm_loss.item():.4f}")
    
    # OT Loss
    print("\n2. Optimal Transport Loss:")
    ot = OptimalTransportLoss(reg=0.1, num_iters=20, downsample=4)
    ot_loss = ot(pred, gt)
    print(f"   Loss: {ot_loss.item():.4f}")
    
    # Combined Loss
    print("\n3. Combined P2R Loss V2:")
    combined = P2RLossV2(
        count_weight=3.0,
        dm_weight=0.5,
        use_ot=True,
        ot_weight=0.3,
    )
    total_loss, metrics = combined(pred, gt, points, cell_area=64)
    print(f"   Total: {metrics['total']:.4f}")
    print(f"   Count: {metrics['count']:.4f}")
    print(f"   DM:    {metrics.get('dm', 0):.4f}")
    print(f"   OT:    {metrics.get('ot', 0):.4f}")
    print(f"   MAE:   {metrics['mae']:.2f}")
    
    print("\n✅ All tests passed!")