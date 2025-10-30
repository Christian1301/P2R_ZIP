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

        # ✅ ora rispetta il valore da config
        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input
        self.debug = debug

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)

        # --- ZIP head ---
        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]

        # --- Maschera occupazione ---
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:]  # probabilità che il blocco sia occupato
        mask = (pi_not_zero > self.pi_thresh).float()

        # ✅ Allinea la maschera alla risoluzione delle feature
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        # --- Debug: percentuale blocchi attivi ---
        if self.debug:
            active_ratio = mask.mean().item() * 100
            print(f"[DEBUG] Active blocks ratio: {active_ratio:.2f}% (th={self.pi_thresh})")

        # --- Gating ---
        if self.gate == "multiply":
            gated = feat * mask
        elif self.gate == "concat":
            gated = torch.cat([feat, mask.expand_as(feat[:, :1, :, :])], dim=1)
        else:
            raise ValueError("gate deve essere 'multiply' o 'concat'")

        # --- P2R head ---
        dens = self.p2r_head(gated)
        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)

        pred_density_zip = pi_not_zero * lambda_maps

        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens
        }