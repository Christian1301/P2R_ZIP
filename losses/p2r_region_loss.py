# losses/p2r_region_loss.py
# -*- coding: utf-8 -*-
"""
P2R Loss - VERSIONE CORRETTA V2

PROBLEMA RISOLTO:
La versione originale usava BCE con normalizzazione [0,1], che elimina
l'informazione sulla magnitudine. Il modello imparava la FORMA ma non la SCALA,
predicendo sempre ~3-4 persone indipendentemente dal GT.

SOLUZIONE:
Questa versione usa supervisione DIRETTA sul conteggio totale tramite loss L1/L2,
permettendo al modello di apprendere la magnitudine corretta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

ENABLE_LIGHT_DEBUG_LOG = True


class P2RLoss(nn.Module):
    """
    P2R Loss con supervisione diretta sul conteggio.
    
    Combina:
    1. Count Loss (L1): |pred_count - gt_count| - peso ALTO
    2. Scale Loss: penalizza errore relativo
    3. Spatial Loss: MSE sulla forma normalizzata - peso BASSO
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        chunk_size: int = 2048,
        scale_weight: float = 0.5,      # Peso per errore relativo
        pos_weight: float = 1.0,        # Non usato in questa versione
        min_radius: float = 8.0,        # Per soft target spaziale
        max_radius: float = 64.0,
        cost_point: float = 8.0,
        cost_class: float = 1.0,
        count_weight: float = 2.0,      # NUOVO: peso per count loss (ALTO)
        spatial_weight: float = 0.1,    # NUOVO: peso per spatial loss (BASSO)
    ):
        super().__init__()
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.scale_weight = float(scale_weight)
        self.pos_weight = float(pos_weight)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.count_weight = float(count_weight)
        self.spatial_weight = float(spatial_weight)

    def _compute_soft_target(
        self,
        points: torch.Tensor,
        H: int, W: int,
        down_h: float, down_w: float,
        device: torch.device
    ) -> torch.Tensor:
        """Calcola un target soft basato sulla distanza dai punti."""
        
        if points is None or points.numel() == 0:
            return torch.zeros((H, W), device=device)
        
        # Coordinate dei pixel nello spazio input
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Centro dei pixel nello spazio input
        yy_in = yy * down_h + down_h / 2
        xx_in = xx * down_w + down_w / 2
        
        # Per ogni pixel, calcola distanza minima da qualsiasi punto
        min_dist = torch.full((H, W), float('inf'), device=device)
        
        # Limita il numero di punti per evitare OOM
        max_points = min(points.shape[0], 2000)
        step = max(1, points.shape[0] // max_points)
        
        for i in range(0, points.shape[0], step):
            px, py = points[i, 0], points[i, 1]
            dist = torch.sqrt((xx_in - px)**2 + (yy_in - py)**2)
            min_dist = torch.minimum(min_dist, dist)
        
        # Converti distanza in probabilità (più vicino = più alto)
        target = torch.sigmoid((self.min_radius - min_dist) / (self.min_radius * 0.5))
        
        return target

    def forward(
        self,
        dens: torch.Tensor,
        points: List[torch.Tensor],
        down: Tuple[float, float] = (8.0, 8.0),
        masks: Optional[torch.Tensor] = None,
        crop_den_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        device = dens.device
        B, _, H, W = dens.shape
        
        if isinstance(down, (list, tuple)):
            down_h, down_w = float(down[0]), float(down[1])
        else:
            down_h = down_w = float(down)
        
        cell_area = down_h * down_w
        
        total_count_loss = torch.tensor(0.0, device=device)
        total_scale_loss = torch.tensor(0.0, device=device)
        total_spatial_loss = torch.tensor(0.0, device=device)
        
        pred_counts = []
        gt_counts = []
        
        for i in range(B):
            pts = points[i]
            pred = dens[i, 0]  # [H, W]
            
            # Ground truth count
            if pts is not None and hasattr(pts, 'numel') and pts.numel() > 0:
                pts = pts.to(device)
                if pts.ndim == 1:
                    pts = pts.view(-1, 2) if pts.numel() >= 2 else torch.zeros((0, 2), device=device)
                gt_count = float(pts.shape[0])
            else:
                pts = torch.zeros((0, 2), device=device)
                gt_count = 0.0
            
            # Predicted count
            pred_count = pred.sum() / cell_area
            
            # 1. Count Loss (L1) - SUPERVISIONE DIRETTA SUL CONTEGGIO
            count_loss = torch.abs(pred_count - gt_count)
            total_count_loss = total_count_loss + count_loss
            
            # 2. Scale Loss - penalizza la differenza relativa
            if gt_count > 0:
                relative_error = torch.abs(pred_count - gt_count) / gt_count
                scale_loss = relative_error
            else:
                # Per immagini vuote, penalizza predizioni non-zero
                scale_loss = pred_count * 0.1
            total_scale_loss = total_scale_loss + scale_loss
            
            # 3. Spatial Loss - forma della distribuzione (peso basso)
            if gt_count > 0 and self.spatial_weight > 0:
                target = self._compute_soft_target(pts, H, W, down_h, down_w, device)
                
                # Normalizza pred e target per confrontare solo la forma
                pred_max = pred.max()
                target_max = target.max()
                
                if pred_max > 1e-8 and target_max > 1e-8:
                    pred_norm = pred / pred_max
                    target_norm = target / target_max
                    spatial_loss = F.mse_loss(pred_norm, target_norm)
                else:
                    spatial_loss = torch.tensor(0.0, device=device)
            else:
                # Per immagini vuote, penalizza qualsiasi attivazione
                spatial_loss = pred.mean() if gt_count == 0 else torch.tensor(0.0, device=device)
            
            total_spatial_loss = total_spatial_loss + spatial_loss
            
            pred_counts.append(pred_count.item())
            gt_counts.append(gt_count)
        
        # Loss totale
        avg_count = total_count_loss / B
        avg_scale = total_scale_loss / B
        avg_spatial = total_spatial_loss / B
        
        total_loss = (self.count_weight * avg_count + 
                      self.scale_weight * avg_scale + 
                      self.spatial_weight * avg_spatial)
        
        # Debug log
        if ENABLE_LIGHT_DEBUG_LOG:
            pc_mean = np.mean(pred_counts) if pred_counts else 0
            gc_mean = np.mean(gt_counts) if gt_counts else 0
            ratio = pc_mean / max(gc_mean, 1e-6)
            print(f"[P2R DEBUG] total_loss={total_loss.item():.4f}, "
                  f"count_l1={avg_count.item():.2f}, scale={avg_scale.item():.4f}, "
                  f"spatial={avg_spatial.item():.4f}, "
                  f"pred={pc_mean:.1f}, gt={gc_mean:.1f}, ratio={ratio:.3f}")
        
        return total_loss


# Alias per retrocompatibilità
P2RLossFixed = P2RLoss