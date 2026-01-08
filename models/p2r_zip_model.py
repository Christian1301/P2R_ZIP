# P2R_ZIP/models/p2r_zip_model.py
# VERSIONE CORRETTA V3 - Logic Strict Gating (Output Masking)
#
# PROBLEMA RISOLTO:
# Anche con feature masking (gated = feat * mask), i bias dei layer convoluzionali
# producevano valori non-zero (es. 0.001) nelle zone vuote.
#
# SOLUZIONE:
# Output Masking: dens = dens * mask.
# Forza matematicamente lo zero assoluto nelle zone scartate dallo ZIP.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .backbone import BackboneWrapper
from .zip_head import ZIPHead
from .p2r_head import P2RHead


class STEMask(torch.autograd.Function):
    """
    Straight-Through Estimator per il masking.
    """
    @staticmethod
    def forward(ctx, soft_mask, threshold):
        hard_mask = (soft_mask > threshold).float()
        ctx.save_for_backward(soft_mask)
        return hard_mask
    
    @staticmethod
    def backward(ctx, grad_output):
        soft_mask, = ctx.saved_tensors
        grad_input = grad_output * soft_mask * (1 - soft_mask + 0.1)
        return grad_input, None


def ste_mask(soft_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return STEMask.apply(soft_mask, threshold)


class P2R_ZIP_Model(nn.Module):
    """
    Modello P2R-ZIP con Strict Gating logic.
    """
    
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        backbone_name: str = "vgg16_bn",
        pi_thresh: float = 0.5,
        gate: str = "multiply",
        upsample_to_input: bool = True,
        debug: bool = False,
        use_ste_mask: bool = True,
        mask_residual: float = 0.0,
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
        self.use_ste_mask = use_ste_mask
        self.mask_residual = mask_residual

    def _compute_mask(self, pi_not_zero: torch.Tensor) -> torch.Tensor:
        """Calcola la maschera ZIP."""
        if self.use_ste_mask:
            if self.training:
                mask = ste_mask(pi_not_zero, self.pi_thresh)
            else:
                mask = (pi_not_zero > self.pi_thresh).float()
        else:
            if self.training:
                mask = torch.clamp(pi_not_zero + 0.1, 0.0, 1.0)
            else:
                mask = (pi_not_zero > self.pi_thresh).float()
        
        if self.mask_residual > 0:
            mask = mask + self.mask_residual * (1 - mask)
        return mask

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> dict:
        B, C, H, W = x.shape
        
        # 1. Backbone
        feat = self.backbone(x)
        
        # 2. ZIP Head (Classificazione Blocchi)
        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]
        
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:]  # [B, 1, Hb, Wb]
        
        # 3. Maschera ZIP
        mask = self._compute_mask(pi_not_zero)
        
        # Resize mask per feature gating
        if mask.shape[-2:] != feat.shape[-2:]:
            mask_feat = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
        else:
            mask_feat = mask
        
        # 4. Feature Gating (Input Masking)
        # Questo aiuta il modello a non vedere il noise
        if self.gate == "multiply":
            gated = feat * mask_feat
        elif self.gate == "concat":
            gated = torch.cat([feat, mask_feat.expand_as(feat[:, :1, :, :])], dim=1)
        else:
            raise ValueError(f"gate ignoto: {self.gate}")
        
        # 5. P2R Head (Regressione)
        dens = self.p2r_head(gated)
        
        # Calcolo RAW per debug/visualizzazione (senza masking)
        dens_raw = None
        if return_intermediates or self.debug:
            with torch.no_grad():
                dens_raw = self.p2r_head(feat) # P2R su feature originali
                if self.upsample_to_input:
                    dens_raw = F.interpolate(dens_raw, size=(H, W), mode="bilinear", align_corners=False)

        # Upsample densità finale
        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)
        
        # 6. STRICT OUTPUT MASKING (La tua richiesta specifica)
        # Forza a 0.0 l'output dove ZIP dice "vuoto".
        # Elimina i bias residui del P2R.
        if mask.shape[-2:] != dens.shape[-2:]:
            mask_final = F.interpolate(mask, size=dens.shape[-2:], mode="nearest")
        else:
            mask_final = mask
            
        dens = dens * mask_final

        # Output ZIP density (solo per riferimento)
        pred_density_zip = pi_not_zero * lambda_maps
        
        outputs = {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens,              # Output finale mascherato
            "mask": mask,
            "pi_probs": pi_not_zero,
        }
        
        if dens_raw is not None:
            outputs["p2r_density_raw"] = dens_raw
        
        return outputs
    
    def get_count_from_density(self, density, cell_area=1.0):
        return torch.sum(density, dim=(1, 2, 3)) / cell_area