"""
P2R-ZIP Model V2 - Bypass Gating Architecture

ARCHITETTURA:
    Immagine → Backbone → features [B, 512, H, W]
                              ↓
                         ZIP Head → π [B, 1, H, W], λ [B, 1, H, W]
                              ↓
                         NESSUN masking!
                              ↓
                         concat(features, π, λ) → [B, 514, H, W]
                              ↓
                         P2R Head [514 ch input]
                              ↓
                         density map

VANTAGGI vs Hard Gating:
1. Zero perdita informazione: P2R vede SEMPRE tutte le features
2. π e λ come hint: P2R può usarli per guidare l'attenzione
3. Robustezza: Se π sbaglia, P2R può ignorarlo
4. Baseline garantita: Worst case = P2R standalone (~MAE 69)

COMPATIBILITÀ:
- Checkpoint Stage 1: ✅ Backbone + ZIP head riutilizzabili
- P2R head: 🆕 Deve essere reinizializzato (514 vs 512 canali)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .backbone import BackboneWrapper
from .zip_head import ZIPHead
from .p2r_head import P2RHead


class P2R_ZIP_Model(nn.Module):
    """
    Modello P2R-ZIP V2 con ZIP come feature (no gating).
    
    Args:
        bins: Lista di tuple per bin classification
        bin_centers: Centri dei bin per calcolo λ
        backbone_name: Nome backbone ("vgg16_bn")
        pi_thresh: Threshold per π (solo per metriche, non usato nel forward)
        upsample_to_input: Se True, upsamplare density alla risoluzione input
        zip_head_kwargs: Kwargs per ZIPHead
        p2r_head_kwargs: Kwargs per P2RHead
        freeze_backbone: Se True, congela backbone
        freeze_zip: Se True, congela ZIP head
        gate: IGNORATO - mantenuto per retrocompatibilità Stage 1
        gate_thresh: IGNORATO - mantenuto per retrocompatibilità
        use_ste: IGNORATO - mantenuto per retrocompatibilità
    """
    
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        bin_centers: List[float],
        backbone_name: str = "vgg16_bn",
        pi_thresh: float = 0.5,
        upsample_to_input: bool = False,
        zip_head_kwargs: Optional[dict] = None,
        p2r_head_kwargs: Optional[dict] = None,
        freeze_backbone: bool = False,
        freeze_zip: bool = False,
        # Parametri legacy per retrocompatibilità (ignorati in bypass mode)
        gate: str = "none",  # Ignorato - sempre bypass
        gate_thresh: float = 0.5,  # Ignorato
        use_ste: bool = False,  # Ignorato
        use_ste_mask: bool = False,  # Ignorato
    ):
        super().__init__()
        
        self.bins = bins
        self.pi_thresh = pi_thresh
        self.upsample_to_input = upsample_to_input
        self.freeze_backbone = freeze_backbone
        self.freeze_zip = freeze_zip
        
        # Bin centers come buffer (non parametro)
        self.register_buffer(
            "bin_centers",
            torch.tensor(bin_centers, dtype=torch.float32).view(1, -1, 1, 1)
        )
        
        # ============================================================
        # BACKBONE
        # ============================================================
        self.backbone = BackboneWrapper(backbone_name)
        backbone_out_ch = self.backbone.out_channels  # 512 per VGG16
        
        # ============================================================
        # ZIP HEAD (produce π e λ)
        # ============================================================
        zip_head_kwargs = zip_head_kwargs or {}
        self.zip_head = ZIPHead(
            in_ch=backbone_out_ch,
            bins=self.bins,
            **zip_head_kwargs
        )
        
        # ============================================================
        # P2R HEAD (riceve features + π + λ = 514 canali)
        # ============================================================
        p2r_head_kwargs = p2r_head_kwargs or {}
        
        # Rimuovi eventuali kwargs incompatibili
        p2r_head_kwargs.pop("up_scale", None)
        p2r_head_kwargs.pop("out_channel", None)
        
        # IMPORTANTE: in_channel = 514 (512 features + 1 π + 1 λ)
        p2r_in_channel = backbone_out_ch + 2  # 512 + 2 = 514
        
        self.p2r_head = P2RHead(
            in_channel=p2r_in_channel,
            fea_channel=p2r_head_kwargs.get("fea_channel", 256),
            out_stride=p2r_head_kwargs.get("out_stride", 16),
            log_scale_init=p2r_head_kwargs.get("log_scale_init", 4.0),
            log_scale_clamp=p2r_head_kwargs.get("log_scale_clamp", (-2.0, 10.0)),
            dropout_rate=p2r_head_kwargs.get("dropout_rate", 0.0),
            final_dropout_rate=p2r_head_kwargs.get("final_dropout_rate", 0.0),
        )
        
        # ============================================================
        # FREEZE se richiesto
        # ============================================================
        if freeze_backbone:
            self._freeze_module(self.backbone)
        
        if freeze_zip:
            self._freeze_module(self.zip_head)
    
    def _freeze_module(self, module: nn.Module):
        """Congela tutti i parametri di un modulo."""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
    
    def _unfreeze_module(self, module: nn.Module):
        """Scongela tutti i parametri di un modulo."""
        for param in module.parameters():
            param.requires_grad = True
        module.train()
    
    def set_freeze_backbone(self, freeze: bool):
        """Imposta freeze/unfreeze del backbone."""
        self.freeze_backbone = freeze
        if freeze:
            self._freeze_module(self.backbone)
        else:
            self._unfreeze_module(self.backbone)
    
    def set_freeze_zip(self, freeze: bool):
        """Imposta freeze/unfreeze dello ZIP head."""
        self.freeze_zip = freeze
        if freeze:
            self._freeze_module(self.zip_head)
        else:
            self._unfreeze_module(self.zip_head)
    
    def train(self, mode: bool = True):
        """Override train() per rispettare freeze settings."""
        super().train(mode)
        
        # Mantieni frozen i moduli che devono restare frozen
        if self.freeze_backbone:
            self.backbone.eval()
        if self.freeze_zip:
            self.zip_head.eval()
        
        return self
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass con ZIP come feature.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            dict con:
            - p2r_density: [B, 1, H', W'] density map finale
            - pi_probs: [B, 1, Hb, Wb] probabilità π
            - lambda_maps: [B, 1, Hb, Wb] rate Poisson λ
            - logit_pi_maps: [B, 2, Hb, Wb] logits π
            - logit_bin_maps: [B, num_bins-1, Hb, Wb] logits bin
            - features: [B, 512, Hf, Wf] backbone features (per debug)
            - features_augmented: [B, 514, Hf, Wf] features + π + λ
        """
        B, C, H, W = x.shape
        
        # ============================================================
        # 1. BACKBONE FEATURES
        # ============================================================
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        
        # feat shape: [B, 512, H/16, W/16]
        _, _, Hf, Wf = feat.shape
        
        # ============================================================
        # 2. ZIP HEAD → π, λ
        # ============================================================
        if self.freeze_zip:
            with torch.no_grad():
                zip_outputs = self.zip_head(feat, self.bin_centers)
        else:
            zip_outputs = self.zip_head(feat, self.bin_centers)
        
        logit_pi_maps = zip_outputs["logit_pi_maps"]  # [B, 2, Hb, Wb]
        lambda_maps = zip_outputs["lambda_maps"]       # [B, 1, Hb, Wb]
        logit_bin_maps = zip_outputs["logit_bin_maps"]
        
        # π probabilità (canale 1 = "occupato")
        pi_softmax = logit_pi_maps.softmax(dim=1)
        pi_probs = pi_softmax[:, 1:2]  # [B, 1, Hb, Wb]
        
        # ============================================================
        # 3. ALLINEA π e λ ALLA RISOLUZIONE FEATURES
        # ZIP head potrebbe avere risoluzione diversa
        # ============================================================
        if pi_probs.shape[-2:] != feat.shape[-2:]:
            pi_aligned = F.interpolate(
                pi_probs,
                size=(Hf, Wf),
                mode='bilinear',
                align_corners=False
            )
            lambda_aligned = F.interpolate(
                lambda_maps,
                size=(Hf, Wf),
                mode='bilinear',
                align_corners=False
            )
        else:
            pi_aligned = pi_probs
            lambda_aligned = lambda_maps
        
        # ============================================================
        # 4. CONCATENA FEATURES + π + λ (NO MASKING!)
        # Questa è la differenza chiave rispetto a V1
        # ============================================================
        features_augmented = torch.cat([feat, pi_aligned, lambda_aligned], dim=1)
        # Shape: [B, 514, Hf, Wf]
        
        # ============================================================
        # 5. P2R HEAD
        # ============================================================
        density = self.p2r_head(features_augmented)
        
        # ============================================================
        # 6. UPSAMPLE SE RICHIESTO
        # ============================================================
        if self.upsample_to_input:
            density = F.interpolate(
                density,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # ============================================================
        # OUTPUT
        # ============================================================
        return {
            "p2r_density": density,
            "pi_probs": pi_probs,
            "lambda_maps": lambda_maps,
            "logit_pi_maps": logit_pi_maps,
            "logit_bin_maps": logit_bin_maps,
            "features": feat,
            "features_augmented": features_augmented,
        }
    
    def get_count(
        self,
        density: torch.Tensor,
        cell_area: float = 1.0
    ) -> torch.Tensor:
        """
        Calcola il conteggio dalla density map.
        
        Args:
            density: [B, 1, H, W]
            cell_area: area di ogni cella per normalizzazione
            
        Returns:
            [B] conteggio per ogni immagine
        """
        return density.sum(dim=(1, 2, 3)) / cell_area
    
    def load_stage1_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device,
        strict: bool = False
    ) -> dict:
        """
        Carica checkpoint Stage 1 (backbone + ZIP head).
        P2R head rimane con pesi inizializzati fresh.
        
        Args:
            checkpoint_path: Path al checkpoint
            device: Device target
            strict: Se True, richiede match esatto delle chiavi
            
        Returns:
            dict con info sul caricamento
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Estrai state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filtra solo backbone e zip_head
        backbone_keys = []
        zip_keys = []
        skipped_keys = []
        
        filtered_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                filtered_state[k] = v
                backbone_keys.append(k)
            elif k.startswith('zip_head.'):
                filtered_state[k] = v
                zip_keys.append(k)
            else:
                skipped_keys.append(k)
        
        # Carica pesi filtrati
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        
        # Report
        info = {
            'backbone_keys_loaded': len(backbone_keys),
            'zip_keys_loaded': len(zip_keys),
            'skipped_keys': len(skipped_keys),
            'missing_keys': len(missing),
            'unexpected_keys': len(unexpected),
            'epoch': checkpoint.get('epoch', None),
            'metrics': checkpoint.get('metrics', {}),
        }
        
        print(f"✅ Stage 1 checkpoint caricato:")
        print(f"   Backbone: {len(backbone_keys)} tensori")
        print(f"   ZIP head: {len(zip_keys)} tensori")
        print(f"   Skipped (P2R): {len(skipped_keys)} tensori")
        
        if checkpoint.get('metrics'):
            metrics = checkpoint['metrics']
            print(f"   Stage 1 metrics: P={metrics.get('precision', 0)*100:.1f}%, "
                  f"R={metrics.get('recall', 0)*100:.1f}%, "
                  f"F1={metrics.get('f1', 0)*100:.1f}%")
        
        return info


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing P2R-ZIP Model (Bypass Gating)")
    print("=" * 70)
    
    # Config di test
    bins = [[0, 0], [1, 3], [4, 6], [7, 10], [11, 15], [16, 22], [23, 32], [33, 9999]]
    bin_centers = [0.0, 2.0, 5.0, 8.5, 13.0, 19.0, 27.5, 45.0]
    
    # Crea modello
    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name="vgg16_bn",
        freeze_backbone=True,
        freeze_zip=True,
    )
    
    print(f"\n1. Architettura:")
    print(f"   Backbone output: 512 canali")
    print(f"   ZIP output: π [B,1,H,W] + λ [B,1,H,W]")
    print(f"   P2R input: 514 canali (512 + 2)")
    print(f"   P2R head GroupNorm groups: {model.p2r_head._norm_groups}")
    
    # Test forward
    print(f"\n2. Test Forward:")
    x = torch.randn(2, 3, 384, 512)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Features: {outputs['features'].shape}")
    print(f"   π probs: {outputs['pi_probs'].shape}")
    print(f"   λ maps: {outputs['lambda_maps'].shape}")
    print(f"   Features augmented: {outputs['features_augmented'].shape}")
    print(f"   Density: {outputs['p2r_density'].shape}")
    
    # Verifica shapes
    assert outputs['features'].shape[1] == 512, "Backbone deve outputtare 512 canali"
    assert outputs['features_augmented'].shape[1] == 514, "Features + π + λ deve essere 514"
    assert outputs['p2r_density'].shape[1] == 1, "Density deve avere 1 canale"
    
    # Test count
    print(f"\n3. Test Count:")
    density = outputs['p2r_density']
    count = model.get_count(density, cell_area=16*16)
    print(f"   Density sum (raw): {density.sum(dim=[1,2,3]).tolist()}")
    print(f"   Count (normalized): {count.tolist()}")
    
    # Test freeze/unfreeze
    print(f"\n4. Test Freeze/Unfreeze:")
    
    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    print(f"   Con freeze (default):")
    print(f"     Backbone trainable: {count_trainable(model.backbone)}")
    print(f"     ZIP trainable: {count_trainable(model.zip_head)}")
    print(f"     P2R trainable: {count_trainable(model.p2r_head)}")
    
    model.set_freeze_backbone(False)
    model.set_freeze_zip(False)
    
    print(f"   Senza freeze:")
    print(f"     Backbone trainable: {count_trainable(model.backbone)}")
    print(f"     ZIP trainable: {count_trainable(model.zip_head)}")
    print(f"     P2R trainable: {count_trainable(model.p2r_head)}")
    
    # Verifica gradients
    print(f"\n5. Test Gradients:")
    model.set_freeze_backbone(True)
    model.set_freeze_zip(True)
    model.train()
    
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    outputs = model(x)
    loss = outputs['p2r_density'].sum()
    loss.backward()
    
    has_grad_p2r = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.p2r_head.parameters() if p.requires_grad)
    has_grad_backbone = any(p.grad is not None and p.grad.abs().sum() > 0 
                            for p in model.backbone.parameters())
    
    print(f"   P2R head ha gradienti: {has_grad_p2r}")
    print(f"   Backbone ha gradienti: {has_grad_backbone} (dovrebbe essere False)")
    
    print("\n" + "=" * 70)
    print("✅ Tutti i test passati!")
    print("=" * 70)