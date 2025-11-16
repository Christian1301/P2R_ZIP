# P2R_ZIP/models/p2r_zip_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .backbone import BackboneWrapper
from .zip_head import ZIPHead
from .p2r_head import P2RHead


class P2R_ZIP_Model(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        backbone_name="vgg16_bn",
        pi_thresh=0.5,
        gate="multiply",
        upsample_to_input=True,
        debug=False,
        pi_mode: str = "hard",
        pi_soft_gamma: float = 1.0,
        detach_pi_mask: bool = False,
        zip_head_kwargs: Optional[dict] = None,
        p2r_head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.bins = bins
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32).view(1, -1, 1, 1)
        )

        self.backbone = BackboneWrapper(backbone_name)
        zip_head_kwargs = zip_head_kwargs or {}
        self.zip_head = ZIPHead(self.backbone.out_channels, bins=self.bins, **zip_head_kwargs)
        p2r_head_kwargs = p2r_head_kwargs or {}
        if "in_channel" not in p2r_head_kwargs:
            p2r_head_kwargs["in_channel"] = 512
        if "fea_channel" not in p2r_head_kwargs:
            p2r_head_kwargs["fea_channel"] = 64
        if "up_scale" not in p2r_head_kwargs:
            p2r_head_kwargs["up_scale"] = 2
        self.p2r_head = P2RHead(**p2r_head_kwargs)

        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input
        self.debug = debug
        self.pi_mode = pi_mode.lower() if isinstance(pi_mode, str) else "hard"
        try:
            self.pi_soft_gamma = float(pi_soft_gamma)
        except (TypeError, ValueError):
            self.pi_soft_gamma = 1.0
        if self.pi_soft_gamma <= 0:
            self.pi_soft_gamma = 1.0
        self.detach_pi_mask = bool(detach_pi_mask)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)

        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:]

        if self.pi_thresh is not None:
            hard_mask = (pi_not_zero > self.pi_thresh).float()
        else:
            hard_mask = None

        if self.pi_mode == "soft":
            if self.pi_thresh is None:
                mask = torch.clamp(pi_not_zero, 0.0, 1.0)
            else:
                denom = max(1.0 - float(self.pi_thresh), 1e-6)
                mask = torch.clamp((pi_not_zero - self.pi_thresh) / denom, 0.0, 1.0)
            if self.pi_soft_gamma != 1.0:
                mask = torch.pow(mask, self.pi_soft_gamma)
        elif self.pi_mode == "prob":
            mask = torch.clamp(pi_not_zero, 0.0, 1.0)
        elif self.pi_mode == "none":
            mask = torch.ones_like(pi_not_zero)
        else:  # default hard mask
            if hard_mask is None:
                mask = (pi_not_zero > 0.5).float()
            else:
                mask = hard_mask

        if self.detach_pi_mask and self.pi_mode in {"soft", "prob"}:
            mask = mask.detach()

        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        if self.debug:
            active_ratio = mask.mean().item() * 100
            print(f"[DEBUG] Active blocks ratio: {active_ratio:.2f}% (th={self.pi_thresh})")

        if self.gate == "multiply":
            if mask.shape[1] not in (1, feat.shape[1]):
                raise ValueError(
                    "La maschera ZIP deve avere 1 canale o lo stesso numero di canali delle feature."
                )
            gated = feat * mask
        elif self.gate == "concat":
            gated = torch.cat([feat, mask.expand_as(feat[:, :1, :, :])], dim=1)
        else:
            raise ValueError("gate deve essere 'multiply' o 'concat'")

        dens = self.p2r_head(gated)
        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)

        if lambda_maps.shape != pi_not_zero.shape:
            raise ValueError(
                "lambda_maps e pi_not_zero devono avere la stessa shape per produrre la densità ZIP"
            )
        pred_density_zip = pi_not_zero * lambda_maps

        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens
        }