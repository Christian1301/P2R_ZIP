import torch
import torch.nn as nn
import torch.nn.functional as F

class PiHeadLoss(nn.Module):
    """
    Loss per Stage 1 ispirata a ZIP-CLIP-EBC.
    Si concentra sulla classificazione binaria (Vuoto vs Pieno).
    """
    def __init__(
        self,
        pos_weight: float = 3.0,  
        block_size: int = 8,      
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        
        # BCE con pos_weight per bilanciare le classi
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        """Genera la maschera binaria GT dai punti/densità."""
        # Somma la densità nel blocco
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        # Se c'è almeno mezza persona (o soglia minima), è "pieno"
        gt_occupancy = (gt_counts_per_block > 1e-3).float()
        return gt_occupancy
    
    def forward(self, predictions, gt_density):
        # [B, 2, H, W] -> Canale 1 è la probabilità di "pieno"
        logit_pi_maps = predictions["logit_pi_maps"]
        logit_pieno = logit_pi_maps[:, 1:2, :, :] 
        
        # Calcola GT
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni se necessario (es. padding o arrotondamenti diversi)
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, 
                size=logit_pieno.shape[-2:], 
                mode='nearest'
            )
        
        # Sposta il peso sul device corretto
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
            
        loss = self.bce(logit_pieno, gt_occupancy)
        
        return loss, {"pi_bce_loss": loss.detach()}