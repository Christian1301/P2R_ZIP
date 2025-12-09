# p2r_region_loss_fixed.py
# -*- coding: utf-8 -*-
"""
P2R Loss CORRETTA - Risolve il problema della scale_penalty dominante.

Modifiche chiave:
1. scale_weight default ridotto da 0.01 a 0.001
2. Normalizzazione della scale_penalty per batch size
3. Aggiunta di per-image loss capping per evitare outlier
4. BCE loss pesata in modo più bilanciato
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ENABLE_LIGHT_DEBUG_LOG = True


def _safe_normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalizza in [0,1] una mappa non negativa."""
    x = torch.clamp(x, min=0.0)
    xmax = torch.amax(x)
    if torch.isfinite(xmax) and xmax > 0:
        return x / (xmax + eps)
    return torch.zeros_like(x)


def _check_no_nan_inf(t: torch.Tensor, name: str = "tensor"):
    if torch.isnan(t).any():
        raise ValueError(f"{name} contiene NaN")
    if torch.isinf(t).any():
        raise ValueError(f"{name} contiene Inf")


class L2DIS:
    """Calcola distanze L2 normalizzate."""
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        return torch.norm(x_col - y_row, dim=-1) / max(self.factor, 1e-6)


class P2RLossFixed(nn.Module):
    """
    P2R Loss con correzioni per stabilità.
    
    Modifiche rispetto all'originale:
    1. scale_weight molto ridotto (default 0.005 invece di 0.01)
    2. Loss capping per singola immagine
    3. Scale penalty normalizzata per GT count
    4. Soft target invece di hard threshold
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        chunk_size: int = 4096,
        scale_weight: float = 0.005,      # RIDOTTO da 0.01
        pos_weight: float = 2.0,
        min_radius: float = 8.0,           # AUMENTATO da 4.0
        max_radius: float = 64.0,
        cost_point: float = 8.0,
        cost_class: float = 1.0,
        max_loss_per_image: float = 50.0,  # NUOVO: cap per outlier
        use_soft_target: bool = True,      # NUOVO: target soft
    ):
        super().__init__()
        self.cost = L2DIS(1.0)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.reduction = reduction
        self.chunk_size = int(chunk_size)
        self.scale_weight = float(scale_weight)
        self.pos_weight = float(pos_weight)
        self.max_loss_per_image = float(max_loss_per_image)
        self.use_soft_target = use_soft_target

    @torch.no_grad()
    def _min_distances_streaming(
        self,
        A_coord: torch.Tensor,
        B_coord: torch.Tensor,
        chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcola min distanze in streaming per ridurre memoria."""
        device = A_coord.device
        bsz, n_pix, _ = A_coord.shape
        _, n_pts, _ = B_coord.shape

        if n_pts == 0:
            very_large = torch.full((1, n_pix, 1), float('inf'), device=device)
            zero_idx = torch.zeros((1, n_pix, 1), dtype=torch.long, device=device)
            return very_large, zero_idx

        minC_list, mcidx_list = [], []

        for start in range(0, n_pix, chunk_size):
            end = min(start + chunk_size, n_pix)
            C_chunk = self.cost(A_coord[:, start:end, :], B_coord)
            minC, mcidx = torch.min(C_chunk, dim=-1, keepdim=True)
            minC_list.append(minC)
            mcidx_list.append(mcidx)

        return torch.cat(minC_list, dim=1), torch.cat(mcidx_list, dim=1)

    def forward(
        self,
        dens: torch.Tensor,
        points,
        down=16,
        masks: Optional[torch.Tensor] = None,
        crop_den_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        device = dens.device
        _check_no_nan_inf(dens, "dens")

        if isinstance(down, (list, tuple)):
            down_h, down_w = float(down[0]), float(down[1])
        else:
            down_h = down_w = float(down)

        B = len(points)
        assert dens.ndim == 4 and dens.shape[1] == 1

        total_loss = dens.new_tensor(0.0)
        pred_counts = []
        gt_counts = []
        bce_losses = []
        scale_penalties = []

        for i in range(B):
            den = dens[i].permute(1, 2, 0).contiguous()
            H, W = den.shape[:2]

            seq: torch.Tensor = points[i]
            if not torch.is_tensor(seq):
                raise TypeError(f"points[{i}] deve essere torch.Tensor")

            if seq.ndim == 1:
                seq = seq.reshape(0, 2) if seq.numel() == 0 else seq.reshape(1, -1)

            seq = seq.to(device, dtype=torch.float32)

            # Caso senza punti
            if seq.numel() == 0 or seq.shape[0] == 0:
                loss_empty = F.binary_cross_entropy(
                    _safe_normalize_01(den),
                    torch.zeros_like(den),
                    reduction="mean",
                ) * 0.5  # Peso ridotto per immagini vuote
                total_loss = total_loss + loss_empty
                pred_counts.append(dens[i].sum() / (down_h * down_w))
                gt_counts.append(dens.new_tensor(0.0))
                continue

            H_in = H * down_h
            W_in = W * down_w
            seq[:, 0] = torch.clamp(seq[:, 0], 0, W_in - 1)
            seq[:, 1] = torch.clamp(seq[:, 1], 0, H_in - 1)

            A = den.view(1, -1, 1)

            # Coordinate pixel nello spazio input
            rows = torch.arange(H, device=device, dtype=torch.float32)
            cols = torch.arange(W, device=device, dtype=torch.float32)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            center_offset_h = (down_h - 1.0) / 2.0
            center_offset_w = (down_w - 1.0) / 2.0
            rr_in = rr * down_h + center_offset_h
            cc_in = cc * down_w + center_offset_w
            A_coord = torch.stack([rr_in, cc_in], dim=-1).view(1, -1, 2)

            B_coord = torch.stack([seq[:, 1], seq[:, 0]], dim=-1).unsqueeze(0)

            minC, _ = self._min_distances_streaming(A_coord, B_coord, self.chunk_size)

            # CORREZIONE: Target SOFT invece di hard threshold
            if self.use_soft_target:
                # Sigmoid-like soft target basato sulla distanza
                T = torch.sigmoid((self.min_radius - minC) / (self.min_radius * 0.5))
                T = T.view_as(A)
            else:
                T = (minC < self.min_radius).float().view_as(A)

            # Pesi per BCE
            Wt = torch.ones_like(T)
            if self.pos_weight != 1.0:
                Wt = torch.where(T > 0.5, Wt * self.pos_weight, Wt)

            A_norm = _safe_normalize_01(A)

            loss_bce = F.binary_cross_entropy(A_norm, T, weight=Wt, reduction="mean")
            bce_losses.append(loss_bce.item())
            
            # CORREZIONE: Cap sulla loss per singola immagine
            loss_i = torch.clamp(loss_bce, max=self.max_loss_per_image)
            total_loss = total_loss + loss_i

            pred_counts.append(dens[i].sum() / (down_h * down_w))
            gt_counts.append(dens.new_tensor(float(seq.shape[0])))

        # Scale penalty CORRETTA
        if len(pred_counts) > 0:
            pred_counts_t = torch.stack(pred_counts)
            gt_counts_t = torch.stack(gt_counts)

            # CORREZIONE: Normalizza la scale penalty per il GT count medio
            # Questo evita che immagini dense dominino la loss
            gt_mean = gt_counts_t.mean().clamp(min=1.0)
            relative_errors = torch.abs(pred_counts_t - gt_counts_t) / gt_mean
            
            scale_penalty = self.scale_weight * relative_errors.mean()
            scale_penalties.append(scale_penalty.item())
            
            # CORREZIONE: Cap anche sulla scale penalty
            scale_penalty = torch.clamp(scale_penalty, max=5.0)
            total_loss = total_loss + scale_penalty
        else:
            scale_penalty = dens.new_tensor(0.0)

        if self.reduction == "mean":
            total_loss = total_loss / max(B, 1)

        # Debug log migliorato
        if ENABLE_LIGHT_DEBUG_LOG:
            with torch.no_grad():
                pc = torch.stack(pred_counts).mean().item() if pred_counts else 0.0
                gc = torch.stack(gt_counts).mean().item() if gt_counts else 0.0
                bce_avg = sum(bce_losses) / len(bce_losses) if bce_losses else 0.0
                sp_avg = sum(scale_penalties) / len(scale_penalties) if scale_penalties else 0.0
                
                # NUOVO: Mostra il rapporto BCE/scale_penalty
                ratio = bce_avg / (sp_avg + 1e-8)
                
                print(f"[P2R DEBUG] total_loss={total_loss.item():.4f}, "
                      f"BCE_avg={bce_avg:.4f}, scale_penalty={sp_avg:.4f}, "
                      f"BCE/SP_ratio={ratio:.2f}, "
                      f"pred={pc:.1f}, gt={gc:.1f}")

        return total_loss