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
        pi_thresh: Optional[float] = 0.5,
        gate: str = "multiply",
        upsample_to_input: bool = True,
        debug: bool = False,
        zip_head_kwargs: Optional[dict] = None,
        p2r_head_kwargs: Optional[dict] = None,
        soft_pi_gate: bool = False,
        pi_gate_power: float = 1.0,
        pi_gate_min: float = 0.0,
        apply_gate_to_output: bool = False,
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
            c_in = self.backbone.out_channels
            if gate == "concat":
                c_in += 1 
            p2r_head_kwargs["in_channel"] = c_in
        if "fea_channel" not in p2r_head_kwargs:
            p2r_head_kwargs["fea_channel"] = 64
        if "up_scale" not in p2r_head_kwargs:
            p2r_head_kwargs["up_scale"] = 2
        self.p2r_head = P2RHead(**p2r_head_kwargs)

        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input
        self.debug = debug
        self.soft_pi_gate = bool(soft_pi_gate)
        self.pi_gate_power = float(pi_gate_power)
        self.pi_gate_min = float(pi_gate_min)
        self.apply_gate_to_output = bool(apply_gate_to_output)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)

        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:] 
        mask = pi_not_zero
        if self.pi_thresh is not None:
            active = (mask >= float(self.pi_thresh)).float()
        else:
            active = torch.ones_like(mask)

        if self.soft_pi_gate:
            gate_vals = mask
            if self.pi_gate_min > 0.0:
                gate_vals = torch.clamp(gate_vals, min=self.pi_gate_min)
            if abs(self.pi_gate_power - 1.0) > 1e-6:
                gate_vals = torch.pow(gate_vals, self.pi_gate_power)
            mask = gate_vals * active
        else:
            mask = active

        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        if self.debug:
            active_ratio = mask.mean().item() * 100
            print(f"[DEBUG] Active blocks ratio: {active_ratio:.2f}% (th={self.pi_thresh})")

        if self.gate == "multiply":
            gated = feat * mask
        elif self.gate == "concat":
            gated = torch.cat([feat, mask.expand_as(feat[:, :1, :, :])], dim=1)
        else:
            raise ValueError("gate deve essere 'multiply' o 'concat'")

        dens = self.p2r_head(gated)

        mask_for_output = None
        if self.apply_gate_to_output:
            mask_for_output = mask
            up_scale = getattr(self.p2r_head, "up_scale", 1)
            if up_scale != 1:
                mask_for_output = F.interpolate(mask_for_output, scale_factor=up_scale, mode="bilinear", align_corners=False)

        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)
            if self.apply_gate_to_output:
                mask_resized = F.interpolate(mask_for_output, size=(H, W), mode="bilinear", align_corners=False)
                dens = dens * mask_resized
        elif self.apply_gate_to_output:
            dens = dens * mask_for_output

        pred_density_zip = pi_not_zero * lambda_maps

        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens
        }