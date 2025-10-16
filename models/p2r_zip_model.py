# P2R_ZIP/models/p2r_zip_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .backbone import BackboneWrapper
from .zip_head import ZIPHead
from .p2r_head import P2RHead

class P2R_ZIP_Model(nn.Module):
    def __init__(self, bins: List[Tuple[float, float]], bin_centers: List[float], backbone_name="vgg16_bn", pi_thresh=0.5, gate="multiply", upsample_to_input=True):
        super().__init__()
        self.bins = bins
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32).view(1, -1, 1, 1))

        self.backbone = BackboneWrapper(backbone_name)
        self.zip_head = ZIPHead(self.backbone.out_channels, bins=self.bins)
        self.p2r_head = P2RHead(self.backbone.out_channels, gate=gate)
        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)

        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]

        pi_maps_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_maps_softmax[:, 1:]

        mask = (pi_not_zero > self.pi_thresh).float()

        if self.gate == "multiply":
            gated = feat * mask
        elif self.gate == "concat":
            if mask.shape[-2:] != feat.shape[-2:]:
                 mask = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
            gated = torch.cat([feat, mask], dim=1)
        else:
            raise ValueError("gate deve essere 'multiply' o 'concat'")

        dens = self.p2r_head(gated)

        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)

        pred_density_zip = pi_not_zero * lambda_maps
        
        # --- MODIFICA CHIAVE ---
        # Restituisci sempre il dizionario completo.
        # La logica su quale output usare (es. 'p2r_density' per l'inferenza)
        # sarà gestita dagli script di training/validazione.
        return {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens
        }