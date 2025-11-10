# P2R_ZIP/models/zip_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ZIPHead(nn.Module):
    """
    ZIP Head migliorata:
    - λ (lambda) con range espanso e softplus
    - probabilità π bilanciate
    - regolarizzazione leggera su λ per maggiore variabilità
    """
    def __init__(
        self,
        in_ch: int,
        bins: List[Tuple[float, float]],
        lambda_scale: float = 0.5,       
        lambda_max: float = 8.0,         
        use_softplus: bool = True,
        epsilon: float = 1e-6,
        lambda_noise_std: float = 0.0,
    ):
        super().__init__()
        if not all(len(b) == 2 for b in bins):
            raise ValueError("I bin devono essere tuple di lunghezza 2 (es. [(0,0),(1,5),...])")
        if bins[0][0] != 0 or bins[0][1] != 0:
            raise ValueError("Il primo bin deve essere [0, 0]")

        self.bins = bins
        self.lambda_scale = lambda_scale
        self.lambda_max = lambda_max
        self.use_softplus = use_softplus
        self.epsilon = epsilon
        self.lambda_noise_std = lambda_noise_std
        self._softplus_offset = math.log(2.0)  

        inter = max(64, in_ch // 4)

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        self.pi_head = nn.Conv2d(inter, 2, 1)
        self.bin_head = nn.Conv2d(inter, len(bins) - 1, 1)
        
        if self.pi_head.bias is not None and self.pi_head.bias.numel() == 2:
            with torch.no_grad():
                self.pi_head.bias[0] = 1.5
                self.pi_head.bias[1] = -1.5

    def forward(self, feat: torch.Tensor, bin_centers: torch.Tensor):
        h = self.shared(feat)

        logit_pi_maps = self.pi_head(h)     
        logit_bin_maps = self.bin_head(h)   

        centers = bin_centers
        if centers.dim() == 1:
            centers = centers.view(1, -1, 1, 1)

        if centers.shape[1] == logit_bin_maps.shape[1] + 1:
            centers_positive = centers[:, 1:, :, :]
        else:
            centers_positive = centers

        assert centers_positive.shape[1] == logit_bin_maps.shape[1], \
            f"Dimensioni non corrispondenti: {centers_positive.shape[1]} vs {logit_bin_maps.shape[1]}"

        p_bins = F.softmax(logit_bin_maps, dim=1)

        lambda_raw = (p_bins * centers_positive).sum(dim=1, keepdim=True)

        lambda_pre = lambda_raw * self.lambda_scale
        if self.use_softplus:
            lambda_maps = F.softplus(lambda_pre) - self._softplus_offset
        else:
            lambda_maps = F.relu(lambda_pre)

        lambda_maps = torch.clamp(lambda_maps, min=self.epsilon)

        lambda_maps = torch.clamp(lambda_maps, min=self.epsilon, max=self.lambda_max)

        if self.training and self.lambda_noise_std > 0.0:
            noise = torch.randn_like(lambda_maps) * self.lambda_noise_std
            lambda_maps = torch.clamp(lambda_maps + noise, min=self.epsilon, max=self.lambda_max)

        if hasattr(self, "debug") and self.debug:
            print(f"[ZIPHead] λ range: [{lambda_maps.min().item():.3f}, {lambda_maps.max().item():.3f}], "
                  f"π logits mean: {logit_pi_maps.mean().item():.3f}")

        return {
            "logit_pi_maps": logit_pi_maps,   
            "logit_bin_maps": logit_bin_maps, 
            "lambda_maps": lambda_maps        
        }
