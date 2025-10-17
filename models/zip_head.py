# P2R_ZIP/models/zip_head.py
import torch
import torch.nn as nn
from typing import List, Tuple

class ZIPHead(nn.Module):
    def __init__(self, in_ch: int, bins: List[Tuple[float, float]]):
        super().__init__()
        if not all(len(b) == 2 for b in bins):
            raise ValueError(f"I bin devono essere una lista di tuple di lunghezza 2, ma abbiamo ricevuto {bins}")
        self.bins = bins

        inter = max(64, in_ch // 4)
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )

        self.pi_head = nn.Conv2d(inter, 2, 1)
        self.bin_head = nn.Conv2d(inter, len(self.bins) - 1, 1)

    def forward(self, feat, bin_centers: torch.Tensor):
        h = self.shared(feat)
        logit_pi_maps = self.pi_head(h)
        logit_bin_maps = self.bin_head(h)
        centers = bin_centers
        if centers.dim() == 1:
            centers = centers.view(1, -1, 1, 1)

        # se i centers includono lo 0 (K+1), togli lo 0
        if centers.shape[1] == logit_bin_maps.shape[1] + 1:
            centers = centers[:, 1:, :, :]

        assert centers.shape[1] == logit_bin_maps.shape[1], \
            f"centers {centers.shape} vs logits {logit_bin_maps.shape}"

        p_bins = logit_bin_maps.softmax(dim=1)
        lambda_maps = (p_bins * centers).sum(dim=1, keepdim=True)

        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": logit_bin_maps,
            "lambda_maps": lambda_maps
        }