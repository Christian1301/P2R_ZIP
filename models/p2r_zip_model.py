# P2R_ZIP/models/p2r_zip_model.py
# VERSIONE CORRETTA V2 - Con Straight-Through Estimator (STE)
#
# PROBLEMA RISOLTO:
# La versione originale usava soft mask in training e hard mask in eval,
# causando mismatch e performance degradate (MAE 291 vs target 65).
#
# SOLUZIONE:
# Straight-Through Estimator: forward usa hard mask, backward usa soft gradients.
# Questo permette alla P2R head di "vedere" lo stesso masking in train e eval.

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
    
    Forward: applica hard threshold (binario)
    Backward: passa gradienti attraverso soft mask (differenziabile)
    
    Questo permette:
    - Training: gradienti fluiscono attraverso il π-head
    - Inference: comportamento identico al training (hard mask)
    """
    @staticmethod
    def forward(ctx, soft_mask, threshold):
        hard_mask = (soft_mask > threshold).float()
        ctx.save_for_backward(soft_mask)
        return hard_mask
    
    @staticmethod
    def backward(ctx, grad_output):
        soft_mask, = ctx.saved_tensors
        # Passa il gradiente attraverso la soft mask
        # Usa sigmoid derivative come proxy: σ(x)(1-σ(x))
        # Ma qui soft_mask è già una probabilità, quindi usiamo direttamente
        grad_input = grad_output * soft_mask * (1 - soft_mask + 0.1)  # +0.1 per evitare zero gradients
        return grad_input, None


def ste_mask(soft_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Applica STE masking."""
    return STEMask.apply(soft_mask, threshold)


class P2R_ZIP_Model(nn.Module):
    """
    Modello P2R-ZIP con masking unificato via STE.
    
    Modifiche V2:
    1. STE per train/eval consistency
    2. Opzione per bypass masking (usa raw density)
    3. Debug logging migliorato
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
        use_ste_mask: bool = True,  # NUOVO: abilita STE
        mask_residual: float = 0.0,  # NUOVO: residual connection (0 = no residual)
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
        """
        Calcola la maschera con comportamento CONSISTENTE tra train e eval.
        
        Usa STE: forward hard, backward soft.
        """
        if self.use_ste_mask:
            # STE: stesso comportamento in train e eval
            # Forward: hard threshold
            # Backward: gradienti passano attraverso soft mask
            if self.training:
                mask = ste_mask(pi_not_zero, self.pi_thresh)
            else:
                mask = (pi_not_zero > self.pi_thresh).float()
        else:
            # Comportamento legacy (DEPRECATO - causa mismatch)
            if self.training:
                mask = pi_not_zero + 0.1
                mask = torch.clamp(mask, 0.0, 1.0)
            else:
                mask = (pi_not_zero > self.pi_thresh).float()
        
        # Opzionale: residual connection per non azzerare completamente
        if self.mask_residual > 0:
            mask = mask + self.mask_residual * (1 - mask)
        
        return mask

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> dict:
        """
        Forward pass con output dettagliati.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_intermediates: Se True, ritorna anche feature intermedie
            
        Returns:
            dict con:
            - logit_pi_maps: [B, 2, Hb, Wb] logits π-head
            - logit_bin_maps: [B, num_bins-1, Hb, Wb] logits bin
            - lambda_maps: [B, 1, Hb, Wb] rate Poisson
            - pred_density_zip: [B, 1, Hb, Wb] densità ZIP (π * λ)
            - p2r_density: [B, 1, H', W'] densità P2R
            - p2r_density_raw: [B, 1, H', W'] densità P2R SENZA masking (per debug)
            - mask: [B, 1, Hf, Wf] maschera applicata
            - pi_probs: [B, 1, Hb, Wb] probabilità π (softmax)
        """
        B, C, H, W = x.shape
        
        # Backbone features
        feat = self.backbone(x)
        
        # ZIP head
        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]
        
        # Probabilità π (canale 1 = "pieno")
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:]  # [B, 1, Hb, Wb]
        
        # Maschera con STE
        mask = self._compute_mask(pi_not_zero)
        
        # Resize mask to feature size se necessario
        if mask.shape[-2:] != feat.shape[-2:]:
            mask_resized = F.interpolate(
                mask, size=feat.shape[-2:], 
                mode="bilinear", align_corners=False
            )
        else:
            mask_resized = mask
        
        # Debug logging
        if self.debug:
            with torch.no_grad():
                active_ratio = mask.mean().item() * 100
                pi_mean = pi_not_zero.mean().item()
                print(f"[P2R_ZIP] π_mean={pi_mean:.3f}, mask_active={active_ratio:.1f}%, "
                      f"thresh={self.pi_thresh}, ste={self.use_ste_mask}")
        
        # Gating
        if self.gate == "multiply":
            gated = feat * mask_resized
        elif self.gate == "concat":
            gated = torch.cat([feat, mask_resized.expand_as(feat[:, :1, :, :])], dim=1)
        else:
            raise ValueError(f"gate deve essere 'multiply' o 'concat', ricevuto: {self.gate}")
        
        # P2R head
        dens = self.p2r_head(gated)
        
        # Calcola anche density RAW (senza masking) per confronto
        if return_intermediates or self.debug:
            with torch.no_grad():
                dens_raw = self.p2r_head(feat)
                if self.upsample_to_input:
                    dens_raw = F.interpolate(dens_raw, size=(H, W), mode="bilinear", align_corners=False)
        else:
            dens_raw = None
        
        # Upsample se richiesto
        if self.upsample_to_input:
            dens = F.interpolate(dens, size=(H, W), mode="bilinear", align_corners=False)
        
        # ZIP density map
        pred_density_zip = pi_not_zero * lambda_maps
        
        outputs = {
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
            "p2r_density": dens,
            "mask": mask,
            "pi_probs": pi_not_zero,
        }
        
        if dens_raw is not None:
            outputs["p2r_density_raw"] = dens_raw
        
        return outputs
    
    def get_count_from_density(
        self, 
        density: torch.Tensor, 
        cell_area: float = 1.0,
        apply_mask: bool = False,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcola il conteggio dalla density map.
        
        Args:
            density: [B, 1, H, W] density map
            cell_area: area di ogni cella (per downsampling)
            apply_mask: se applicare la maschera π
            mask: maschera custom, se None usa self.mask
            
        Returns:
            [B] conteggio per ogni immagine
        """
        if apply_mask and mask is not None:
            if mask.shape[-2:] != density.shape[-2:]:
                mask = F.interpolate(mask, size=density.shape[-2:], mode="nearest")
            density = density * mask
        
        count = torch.sum(density, dim=(1, 2, 3)) / cell_area
        return count