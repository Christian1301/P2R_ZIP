# P2R_ZIP/models/p2r_zip_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneWrapper
from .zip_head import ZIPHead
from .p2r_head import P2RHead

class P2R_ZIP_Model(nn.Module):
    def __init__(self, backbone_name="vgg16_bn", pi_thresh=0.5, gate="multiply", upsample_to_input=True):
        super().__init__()
        self.backbone = BackboneWrapper(backbone_name)
        self.zip_head  = ZIPHead(self.backbone.out_channels)
        self.p2r_head  = P2RHead(self.backbone.out_channels, gate=gate)
        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)                    # [B, Cb, Hb, Wb]
        pi, lam = self.zip_head(feat)              # [B,1,Hb,Wb]

        # costruiamo la mask a risoluzione feature decoder
        pi_up = F.interpolate(pi, size=feat.shape[-2:], mode="nearest")
        mask = (pi_up > self.pi_thresh).float()

        if self.gate == "multiply":
            gated = feat * mask
        elif self.gate == "concat":
            gated = torch.cat([feat, mask], dim=1)
        else:
            raise ValueError("gate must be 'multiply' or 'concat'")

        dens = self.p2r_head(gated)                # upsample interno 4x

        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)

        return {"pi": pi, "lam": lam, "density": dens}
