# P2R_ZIP/losses/composite_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange

EPS = 1e-8

def _bin_count(gt_den_maps: torch.Tensor, bins: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Assigns each value in gt_den_maps to a bin index [0, N-1].
    Handles counts exactly equal to bin edges correctly.
    """
    if not bins or bins[0][0] != 0 or bins[0][1] != 0:
        raise ValueError("Bins must start with [[0, 0]].")

    # Create bin edges: [b0_start, b1_start, b2_start, ..., bN-1_start, bN-1_end + epsilon]
    bin_edges = torch.tensor([b[0] for b in bins] + [bins[-1][1] + EPS], device=gt_den_maps.device)

    gt_counts_flat = gt_den_maps.flatten() # Flatten for bucketize

    # Find the index+1 of the bin where each count falls.
    # right=False means edge value falls into bin on the right [start, end)
    # Using right=False ensures count 0 falls correctly into bucket index 1.
    binned_indices_1_based = torch.bucketize(gt_counts_flat, bin_edges, right=False)

    # Shift indices to be 0-based: Index 0 now corresponds to bin [0, 0]
    binned_indices_0_based = binned_indices_1_based - 1

    # Clamp indices to be within [0, N-1] just in case of edge issues
    num_bins = len(bins)
    binned_indices_clamped = torch.clamp(binned_indices_0_based, 0, num_bins - 1)

    # Reshape back to original map dimensions (without channel dim)
    return binned_indices_clamped.reshape(gt_den_maps.shape[0], gt_den_maps.shape[2], gt_den_maps.shape[3]).long()


class ZIPCompositeLoss(nn.Module):
    def __init__(self, bins: List[Tuple[float, float]], weight_ce: float = 1.0, zip_block_size: int = 16):
        super().__init__()
        if not bins or bins[0][0] != 0 or bins[0][1] != 0:
             raise ValueError("Bins must start with [[0, 0]] for ZIP loss.")
        self.bins = bins
        self.num_bins = len(bins)
        # Number of classes for CE loss is the number of bins excluding the [0,0] bin
        self.num_ce_classes = self.num_bins - 1
        if self.num_ce_classes <= 0:
             raise ValueError("At least two bins (including [[0, 0]]) are required for CE loss.")
        self.weight_ce = weight_ce
        self.zip_block_size = zip_block_size
        # reduction='none' allows manual masking and averaging
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred: dict, target_density: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        logit_pi_maps = pred["logit_pi_maps"]         # [B, 2, Hb, Wb]
        logit_bin_maps = pred["logit_bin_maps"]       # [B, N-1, Hb, Wb] <- Predicts N-1 classes
        lambda_maps = pred["lambda_maps"]             # [B, 1, Hb, Wb]
        # pred_density_zip is (1-pi_zero)*lambda, used for count loss & validation

        # Ensure prediction channel count matches expected CE classes
        if logit_bin_maps.shape[1] != self.num_ce_classes:
            raise ValueError(f"Prediction dimension mismatch: logit_bin_maps has {logit_bin_maps.shape[1]} channels, expected {self.num_ce_classes} (num_bins - 1)")

        # Calculate GT counts per block
        target_counts = F.avg_pool2d(target_density, kernel_size=self.zip_block_size) * (self.zip_block_size**2) # [B, 1, Hb, Wb]

        # --- NLL Loss ---
        B, _, Hb, Wb = target_counts.shape
        pi_maps = logit_pi_maps.softmax(dim=1)
        pi_zero = pi_maps[:, 0:1]
        pi_not_zero = pi_maps[:, 1:]

        is_zero_gt = (target_counts < 0.5).float() # Mask for GT counts == 0

        log_p0 = torch.log(pi_zero + pi_not_zero * torch.exp(-torch.clamp(lambda_maps, max=80)) + EPS) # Clamp lambda to avoid exp overflow
        zero_loss = -log_p0 * is_zero_gt

        safe_lambda = torch.clamp(lambda_maps, min=EPS)
        safe_target_counts = torch.clamp(target_counts, min=0.0) # Ensure target >= 0 for lgamma
        log_py_positive = (torch.log(pi_not_zero + EPS)
                           - safe_lambda
                           + safe_target_counts * torch.log(safe_lambda)
                           - torch.lgamma(safe_target_counts + 1.0))
        nonzero_loss = -log_py_positive * (1.0 - is_zero_gt)

        nll_loss = (zero_loss + nonzero_loss).sum(dim=(-1, -2)).mean() # Sum over H, W; mean over B

        # --- CE Loss ---
        # Get GT bin indices [0, N-1]
        gt_bin_indices = _bin_count(target_counts, bins=self.bins) # [B, Hb, Wb]

        # Target for CE loss needs to be [0, N-2], corresponding to logit_bin_maps channels
        # Shift indices down by 1. Indices that were 0 (for bin [0,0]) become -1.
        target_ce_indices = gt_bin_indices - 1 # Shape [B, Hb, Wb], values [-1, N-2]

        # Flatten targets and predictions
        target_ce_flat = rearrange(target_ce_indices, "B H W -> (B H W)")
        logit_bin_maps_flat = rearrange(logit_bin_maps, "B C H W -> (B H W) C") # Shape [B*H*W, N-1]

        # Create mask for valid CE targets: GT count > 0.5 AND target index >= 0
        mask_positive_gt = rearrange(target_counts > 0.5, "B C H W -> (B C H W)") # Include C in the flattening
        mask_valid_ce_target = target_ce_flat >= 0
        final_ce_mask = mask_positive_gt & mask_valid_ce_target

        ce_loss = torch.tensor(0.0, device=target_counts.device)
        num_valid_targets = final_ce_mask.sum()

        if num_valid_targets > 0:
            # Select valid targets and corresponding predictions
            valid_target_ce = target_ce_flat[final_ce_mask] # Values in [0, N-2]
            valid_pred_ce = logit_bin_maps_flat[final_ce_mask] # Shape [num_valid, N-1]

            # Calculate CE loss per element
            ce_loss_per_element = self.ce_loss_fn(valid_pred_ce, valid_target_ce)

            # Average loss over the valid elements
            ce_loss = ce_loss_per_element.mean()

        # --- Count L1 Loss ---
        # Use (1-pi_zero)*lambda which represents the expected count when not zero-inflated
        pred_expected_count_per_block = (1.0 - pi_zero) * lambda_maps
        pred_total_count = torch.sum(pred_expected_count_per_block, dim=(1, 2, 3)) # Sum over H, W, C=1
        gt_total_count = torch.sum(target_counts, dim=(1, 2, 3)) # Sum over H, W, C=1
        count_loss = F.l1_loss(pred_total_count, gt_total_count)

        # --- Loss Totale ---
        total_loss = nll_loss + self.weight_ce * ce_loss + count_loss

        loss_dict = {
            "zip_total_loss": total_loss.detach(),
            "zip_nll_loss": nll_loss.detach(),
            "zip_ce_loss": ce_loss.detach(),
            "zip_count_loss": count_loss.detach()
        }
        return total_loss, loss_dict