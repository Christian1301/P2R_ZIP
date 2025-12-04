# losses/composite_loss.py

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zip_nll import zip_nll

EPS = 1e-8  

def _bin_count(gt_den_maps: torch.Tensor, bins: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Assigns each value in gt_den_maps to a bin index [0, N-1].
    Handles counts exactly equal to bin edges correctly.
    """
    if not bins or bins[0][0] != 0 or bins[0][1] != 0:
        raise ValueError("Bins must start with [[0, 0]].")

    bin_edges = torch.tensor([b[0] for b in bins] + [bins[-1][1] + EPS], device=gt_den_maps.device)

    gt_counts_flat = gt_den_maps.flatten() 
    binned_indices_1_based = torch.bucketize(gt_counts_flat, bin_edges, right=False)

    binned_indices_0_based = binned_indices_1_based - 1

    num_bins = len(bins)
    binned_indices_clamped = torch.clamp(binned_indices_0_based, 0, num_bins - 1)

    return binned_indices_clamped.reshape(gt_den_maps.shape[0], gt_den_maps.shape[2], gt_den_maps.shape[3]).long()


class ZIPCompositeLoss(nn.Module):
    """
    Loss composita per lo Stage 1 (ZIP):

    - Cross-entropy sull'occupazione per blocco (background / occupato)
    - Zero-Inflated Poisson NLL sui conteggi per blocco
    - Loss sui conteggi globali (somma π*λ vs somma densità GT)
    """

    def __init__(
        self,
        bins,
        weight_ce: float = 1.0,
        weight_nll: float = 1.0,
        zip_block_size: int = 16,
        count_weight: float = 1.0,
    ):
        super().__init__()
        self.bins = bins
        self.weight_ce = weight_ce
        self.weight_nll = weight_nll
        self.zip_block_size = zip_block_size  # mantenuto per compatibilità eventuale
        self.count_weight = count_weight

    def forward(self, preds, gt_density):
        """
        preds: dict con
          - 'logit_pi_maps': (B, 2, H_pi, W_pi)  -> logits per classi [background, occupato]
          - 'lambda_maps':   (B, 1, H_l, W_l)    -> intensità Poisson per blocco
        gt_density: (B, 1, H_gt, W_gt)          -> mappa di densità full-res
        """
        device = gt_density.device

        logit_pi = preds["logit_pi_maps"]
        lam_maps = preds["lambda_maps"]

        B, C, H_pi, W_pi = logit_pi.shape
        _, _, H_gt, W_gt = gt_density.shape

        # ---------------------------------------------------
        # 1) Conteggi per blocco a risoluzione H_pi x W_pi
        # ---------------------------------------------------
        # Scegliamo kernel/stride in modo da approssimare H_pi, W_pi
        bs_h = max(1, H_gt // H_pi)
        bs_w = max(1, W_gt // W_pi)

        gt_blocks = F.avg_pool2d(
            gt_density,
            kernel_size=(bs_h, bs_w),
            stride=(bs_h, bs_w),
        ) * float(bs_h * bs_w)   # => numero di persone per blocco

        # Se c'è mismatch di dimensioni, riallineiamo
        if gt_blocks.shape[-2:] != (H_pi, W_pi):
            gt_blocks = F.interpolate(gt_blocks, size=(H_pi, W_pi), mode="nearest")

        # ---------------------------------------------------
        # 2) Target occupazione: 0 se blocco vuoto, 1 se occupato
        # ---------------------------------------------------
        occ_target = (gt_blocks > 0.0).long().squeeze(1)   # (B, H_pi, W_pi)

        ce_loss = torch.tensor(0.0, device=device)
        if self.weight_ce > 0.0:
            # classi: 0 = background, 1 = occupato
            ce_loss = F.cross_entropy(logit_pi, occ_target, reduction="mean") * self.weight_ce

        # ---------------------------------------------------
        # 3) NLL Zero-Inflated Poisson sui conteggi per blocco
        # ---------------------------------------------------
        pi_soft = torch.softmax(logit_pi, dim=1)
        pi_occ = pi_soft[:, 1:2]  # P(blocco occupato), shape (B,1,H_pi,W_pi)

        nll = zip_nll(pi_occ, lam_maps, gt_blocks, reduction="mean") * self.weight_nll

        # ---------------------------------------------------
        # 4) Loss sui conteggi globali: somma(π * λ) vs somma densità GT
        # ---------------------------------------------------
        pred_count = (pi_occ * lam_maps).sum(dim=(1, 2, 3))   # [B]
        gt_count = gt_density.sum(dim=(1, 2, 3))              # [B]

        count_loss = torch.tensor(0.0, device=device)
        if self.count_weight > 0.0:
            count_loss = F.smooth_l1_loss(pred_count, gt_count) * self.count_weight

        total = nll + ce_loss + count_loss

        loss_dict = {
            "zip_nll_loss": float(nll.detach().cpu()),
            "zip_ce_loss": float(ce_loss.detach().cpu()),
            "zip_count_loss": float(count_loss.detach().cpu()),
        }

        return total, loss_dict
