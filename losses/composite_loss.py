# P2R_ZIP/losses/composite_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange

EPS = 1e-8

def _bin_count(gt_den_maps: torch.Tensor, bins: List[Tuple[float, float]]) -> torch.Tensor:
    bin_edges = torch.tensor([b[0] for b in bins] + [bins[-1][1]], device=gt_den_maps.device)
    gt_den_maps = gt_den_maps.squeeze(1)
    binned_counts = torch.bucketize(gt_den_maps, bin_edges, right=True) - 1
    return torch.clamp(binned_counts, 0, len(bins) - 1).long()

class ZIPCompositeLoss(nn.Module):
    def __init__(self, bins: List[Tuple[float, float]], weight_ce: float = 1.0, zip_block_size: int = 16):
        super().__init__()
        self.bins = bins
        self.weight_ce = weight_ce
        self.zip_block_size = zip_block_size
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred: dict, target_density: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        logit_pi_maps = pred["logit_pi_maps"]
        logit_bin_maps = pred["logit_bin_maps"]
        lambda_maps = pred["lambda_maps"]
        pred_density_zip = pred["pred_density_zip"]
        
        target_counts = F.avg_pool2d(target_density, kernel_size=self.zip_block_size) * (self.zip_block_size**2)

        B = target_counts.shape[0]
        pi_maps = logit_pi_maps.softmax(dim=1)
        pi_zero = pi_maps[:, 0:1]
        pi_not_zero = pi_maps[:, 1:]

        is_zero = (target_counts < 0.5).float()

        log_p0 = torch.log(pi_zero + pi_not_zero * torch.exp(-lambda_maps) + EPS)
        zero_loss = -log_p0 * is_zero

        log_py_positive = torch.log(pi_not_zero + EPS) - lambda_maps + target_counts * torch.log(lambda_maps + EPS) - torch.lgamma(target_counts + 1.0)
        nonzero_loss = -log_py_positive * (1.0 - is_zero)

        nll_loss = (zero_loss + nonzero_loss).sum(dim=(-1, -2)).mean()

        gt_class_maps = _bin_count(target_counts, bins=self.bins)
        gt_class_maps_flat = rearrange(gt_class_maps, "B H W -> (B H W)")
        logit_bin_maps_flat = rearrange(logit_bin_maps, "B C H W -> (B H W) C")

        # --- RIGA CORRETTA ---
        # Il pattern ora è "B C H W -> (B C H W)", che appiattisce tutte le dimensioni.
        mask = (rearrange(target_counts, "B C H W -> (B C H W)") > 0.5)
        
        ce_loss = torch.tensor(0.0, device=target_counts.device)
        if mask.sum() > 0:
            target_ce = gt_class_maps_flat[mask] - 1
            pred_ce = logit_bin_maps_flat[mask]
            if len(pred_ce) > 0:
                ce_loss = self.ce_loss_fn(pred_ce, target_ce).sum() / B

        pred_total_count = pred_density_zip.sum(dim=(-1, -2, -3))
        gt_total_count = target_counts.sum(dim=(-1, -2, -3))
        count_loss = F.l1_loss(pred_total_count, gt_total_count)

        total_loss = nll_loss + self.weight_ce * ce_loss + count_loss

        loss_dict = {
            "zip_total_loss": total_loss.detach(),
            "zip_nll_loss": nll_loss.detach(),
            "zip_ce_loss": ce_loss.detach(),
            "zip_count_loss": count_loss.detach()
        }
        return total_loss, loss_dict