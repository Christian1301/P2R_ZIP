# models/p2r_zip_model.py
# VERSIONE V5 - Fix Train/Eval Consistency
#
# MODIFICHE V5:
# - FIX CRITICO: Comportamento CONSISTENTE tra train e eval
# - Quando use_ste_mask=False: SEMPRE soft mask (sia train che eval)
# - Quando use_ste_mask=True: STE in train, hard mask in eval
# - Output masking applicato SOLO con use_ste_mask=True in eval

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
    Forward: hard threshold (0 o 1)
    Backward: gradiente passa inalterato
    """
    @staticmethod
    def forward(ctx, soft_mask, threshold):
        hard_mask = (soft_mask > threshold).float()
        ctx.save_for_backward(soft_mask)
        return hard_mask
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE puro: gradiente passa inalterato
        return grad_output, None


def ste_mask(soft_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Applica STE mask."""
    return STEMask.apply(soft_mask, threshold)


class P2R_ZIP_Model(nn.Module):
    """
    Modello P2R-ZIP con comportamento CONSISTENTE train/eval.
    
    Modalità di masking:
    - use_ste_mask=True:  Hard mask via STE (train) / Hard threshold (eval)
    - use_ste_mask=False: Soft mask (probabilità) sia in train che eval
    
    Questo garantisce che il modello veda lo STESSO comportamento
    durante training e validation, eliminando il mismatch.
    """
    
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        backbone_name: str = "vgg16_bn",
        pi_thresh: float = 0.5,
        gate: str = "multiply",
        upsample_to_input: bool = False,
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

        # Backbone
        self.backbone = BackboneWrapper(backbone_name)
        
        # ZIP Head
        zip_head_kwargs = zip_head_kwargs or {}
        self.zip_head = ZIPHead(
            self.backbone.out_channels, 
            bins=self.bins, 
            **zip_head_kwargs
        )
        
        # P2R Head
        p2r_head_kwargs = p2r_head_kwargs or {}
        if "in_channel" not in p2r_head_kwargs:
            p2r_head_kwargs["in_channel"] = 512
        if "fea_channel" not in p2r_head_kwargs:
            p2r_head_kwargs["fea_channel"] = 64
        if "up_scale" not in p2r_head_kwargs:
            p2r_head_kwargs["up_scale"] = 2
        self.p2r_head = P2RHead(**p2r_head_kwargs)

        # Configurazione
        self.pi_thresh = pi_thresh
        self.gate = gate
        self.upsample_to_input = upsample_to_input
        self.debug = debug
        self.use_ste_mask = use_ste_mask
        self.mask_residual = mask_residual

    def _compute_mask(self, pi_not_zero: torch.Tensor) -> torch.Tensor:
        """
        Calcola la maschera ZIP con comportamento CONSISTENTE train/eval.
        
        Args:
            pi_not_zero: Probabilità che il blocco contenga persone [B, 1, H, W]
        
        Returns:
            mask: Maschera da applicare alle feature
            
        Comportamento:
            - use_ste_mask=True:  Hard mask (STE in train, threshold in eval)
            - use_ste_mask=False: Soft mask (probabilità) SEMPRE
        """
        if self.use_ste_mask:
            # Modalità HARD: usa threshold
            if self.training:
                # In training: STE per permettere backprop
                mask = ste_mask(pi_not_zero, self.pi_thresh)
            else:
                # In eval: semplice threshold
                mask = (pi_not_zero > self.pi_thresh).float()
        else:
            # Modalità SOFT: usa probabilità direttamente
            # SEMPRE, sia in train che eval - questo garantisce consistenza
            mask = pi_not_zero
        
        # Opzionale: aggiungi residual per non azzerare completamente
        if self.mask_residual > 0:
            mask = mask + self.mask_residual * (1 - mask)
        
        return mask

    def _resize_mask(self, mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Ridimensiona la maschera alla dimensione target."""
        if mask.shape[-2:] != target_size:
            return F.interpolate(mask, size=target_size, mode="nearest")
        return mask

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> dict:
        """
        Forward pass del modello P2R-ZIP.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_intermediates: Se True, include output raw per debug
        
        Returns:
            dict con:
                - p2r_density: Mappa di densità finale
                - mask: Maschera applicata
                - pi_probs: Probabilità ZIP
                - logit_pi_maps, lambda_maps: Output ZIP head
                - (opzionale) p2r_density_raw: Densità senza masking
        """
        B, C, H, W = x.shape
        
        # =====================================================================
        # 1. BACKBONE - Estrazione feature
        # =====================================================================
        feat = self.backbone(x)
        
        # =====================================================================
        # 2. ZIP HEAD - Classificazione blocchi
        # =====================================================================
        zip_outputs = self.zip_head(feat, self.bin_centers)
        logit_pi_maps = zip_outputs["logit_pi_maps"]
        lambda_maps = zip_outputs["lambda_maps"]
        
        # Probabilità che il blocco contenga persone (canale 1)
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_not_zero = pi_softmax[:, 1:]  # [B, 1, Hb, Wb]
        
        # =====================================================================
        # 3. MASCHERA - Comportamento consistente train/eval
        # =====================================================================
        mask = self._compute_mask(pi_not_zero)
        
        # Ridimensiona maschera per feature gating
        mask_feat = self._resize_mask(mask, feat.shape[-2:])
        
        # =====================================================================
        # 4. FEATURE GATING - Applica maschera alle feature
        # =====================================================================
        if self.gate == "multiply":
            gated = feat * mask_feat
        elif self.gate == "concat":
            gated = torch.cat([feat, mask_feat.expand(-1, feat.shape[1], -1, -1)], dim=1)
        else:
            raise ValueError(f"Gate type '{self.gate}' non supportato. Usa 'multiply' o 'concat'.")
        
        # =====================================================================
        # 5. P2R HEAD - Regressione densità
        # =====================================================================
        dens = self.p2r_head(gated)
        
        # =====================================================================
        # 6. CALCOLO RAW (opzionale, per debug/visualizzazione)
        # =====================================================================
        dens_raw = None
        if return_intermediates or self.debug:
            with torch.no_grad():
                dens_raw = self.p2r_head(feat)
                if self.upsample_to_input:
                    dens_raw = F.interpolate(
                        dens_raw, size=(H, W), 
                        mode="bilinear", align_corners=False
                    )

        # =====================================================================
        # 7. UPSAMPLE (opzionale)
        # =====================================================================
        if self.upsample_to_input:
            dens = F.interpolate(
                dens, size=(H, W), 
                mode="bilinear", align_corners=False
            )
        
        # =====================================================================
        # 8. OUTPUT MASKING
        # =====================================================================
        # Applica output masking SOLO con use_ste_mask=True in eval
        # Questo mantiene consistenza: 
        #   - soft mask mode: NO output masking (feature gating basta)
        #   - hard mask mode: output masking in eval per "pulizia"
        
        if self.use_ste_mask and not self.training:
            mask_output = self._resize_mask(mask, dens.shape[-2:])
            dens = dens * mask_output

        # =====================================================================
        # 9. OUTPUT
        # =====================================================================
        # Densità ZIP (per riferimento/confronto)
        pred_density_zip = pi_not_zero * lambda_maps
        
        outputs = {
            # Output principali
            "p2r_density": dens,
            "mask": mask,
            "pi_probs": pi_not_zero,
            
            # ZIP head outputs
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": zip_outputs["logit_bin_maps"],
            "lambda_maps": lambda_maps,
            "pred_density_zip": pred_density_zip,
        }
        
        # Aggiungi raw se richiesto
        if dens_raw is not None:
            outputs["p2r_density_raw"] = dens_raw
        
        return outputs
    
    def get_count_from_density(self, density: torch.Tensor, cell_area: float = 1.0) -> torch.Tensor:
        """
        Calcola il conteggio dalla mappa di densità.
        
        Args:
            density: Mappa di densità [B, 1, H, W]
            cell_area: Area di ogni cella per normalizzazione
        
        Returns:
            count: Conteggio per ogni immagine nel batch [B]
        """
        return torch.sum(density, dim=(1, 2, 3)) / cell_area
    
    def get_trainable_params_info(self) -> dict:
        """Ritorna informazioni sui parametri trainabili."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "percent_trainable": 100 * trainable / total if total > 0 else 0
        }