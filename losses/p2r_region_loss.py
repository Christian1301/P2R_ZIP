"""
P2R Region Loss 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class P2RRegionLoss(nn.Module):
    """
    Loss per P2R density estimation.
    
    Combina:
    1. Count Loss (L1): |pred_count - gt_count| - peso ALTO
    2. Scale Loss: penalizza errore relativo per immagini con molte persone
    3. Spatial Loss: MSE sulla forma normalizzata - peso BASSO
    
    Args:
        count_weight: Peso per count loss (default 2.0 - ALTO)
        spatial_weight: Peso per spatial loss (default 0.15 - BASSO)
        scale_weight: Peso per scale loss (default 0.5)
        eps: Epsilon per stabilità numerica
    """
    
    def __init__(
        self,
        count_weight: float = 2.0,      # CRITICO: peso alto per count
        spatial_weight: float = 0.15,   # CRITICO: peso basso per spatial
        scale_weight: float = 0.5,
        eps: float = 1e-6
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.eps = eps
    
    def forward(
        self, 
        pred_density: torch.Tensor,
        points_list: List[torch.Tensor],
        downscale: int = 16
    ) -> Dict[str, torch.Tensor]:
        """
        Calcola la loss combinata.
        
        Args:
            pred_density: Predicted density map [B, 1, H, W]
            points_list: Lista di tensori con coordinate punti per ogni immagine
            downscale: Fattore di downscale del backbone
            
        Returns:
            Dict con 'total_loss', 'count_loss', 'spatial_loss', 'scale_loss'
        """
        batch_size = pred_density.shape[0]
        device = pred_density.device
        
        count_losses = []
        spatial_losses = []
        scale_losses = []
        
        for b in range(batch_size):
            pred = pred_density[b]  # [1, H, W]
            points = points_list[b]  # [N, 2] o empty
            
            # Ground truth count
            gt_count = points.shape[0] if len(points) > 0 else 0
            gt_count = torch.tensor(gt_count, dtype=torch.float32, device=device)
            
            # Predicted count (sum della density map)
            pred_count = pred.sum()
            
            # ============================================================
            # 1. COUNT LOSS (L1) - SUPERVISIONE DIRETTA SUL CONTEGGIO
            # Questo è il loss principale, peso alto (2.0)
            # ============================================================
            count_loss = torch.abs(pred_count - gt_count)
            count_losses.append(count_loss)
            
            # ============================================================
            # 2. SCALE LOSS - Penalizza errore relativo
            # Importante per immagini dense dove errore assoluto è alto
            # ============================================================
            if gt_count > 1:
                relative_error = torch.abs(pred_count - gt_count) / (gt_count + self.eps)
                scale_losses.append(relative_error)
            else:
                scale_losses.append(torch.zeros(1, device=device).squeeze())
            
            # ============================================================
            # 3. SPATIAL LOSS - Guida la localizzazione
            # Peso basso (0.15) per non dominare il training
            # ============================================================
            if gt_count > 0 and len(points) > 0:
                # Genera GT density map dai punti
                H, W = pred.shape[1], pred.shape[2]
                gt_density = self._generate_gt_density(
                    points, H, W, downscale, device
                )
                
                # Normalizza entrambe per confrontare solo la forma
                pred_norm = pred / (pred.sum() + self.eps)
                gt_norm = gt_density / (gt_density.sum() + self.eps)
                
                # MSE sulla forma normalizzata
                spatial_loss = F.mse_loss(pred_norm, gt_norm)
                spatial_losses.append(spatial_loss)
            else:
                # Per immagini vuote, penalizza predizioni non-zero
                spatial_loss = pred.mean()  # Dovrebbe essere ~0
                spatial_losses.append(spatial_loss)
        
        # Aggregazione losses
        count_loss = torch.stack(count_losses).mean()
        scale_loss = torch.stack(scale_losses).mean()
        spatial_loss = torch.stack(spatial_losses).mean()
        
        # ============================================================
        # TOTAL LOSS con pesi V9
        # count_weight=2.0 (ALTO), spatial_weight=0.15 (BASSO)
        # ============================================================
        total_loss = (
            self.count_weight * count_loss +
            self.scale_weight * scale_loss +
            self.spatial_weight * spatial_loss
        )
        
        return {
            'total_loss': total_loss,
            'count_loss': count_loss,
            'scale_loss': scale_loss,
            'spatial_loss': spatial_loss
        }
    
    def _generate_gt_density(
        self,
        points: torch.Tensor,
        H: int,
        W: int,
        downscale: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Genera una GT density map dai punti.
        Usa distribuzione uniforme nel blocco contenente il punto.
        """
        gt_density = torch.zeros(1, H, W, device=device)
        
        for point in points:
            x, y = point[0].item(), point[1].item()
            # Coordinate nella feature map
            fx = int(x / downscale)
            fy = int(y / downscale)
            
            # Clamp alle dimensioni
            fx = min(max(fx, 0), W - 1)
            fy = min(max(fy, 0), H - 1)
            
            # Aggiungi 1 al blocco (ogni persona = 1)
            gt_density[0, fy, fx] += 1.0
        
        return gt_density


class MultiScaleP2RLoss(nn.Module):
    """
    Multi-scale loss per P2R.
    
    Calcola la loss a diverse risoluzioni per forzare consistenza.
    CRITICO per la convergenza in V9.
    
    Args:
        scales: Lista di scale (es. [1, 2, 4])
        scale_weights: Pesi per ogni scala (es. [1.0, 0.5, 0.25])
        count_weight: Peso per count loss
        spatial_weight: Peso per spatial loss
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        scale_weights: List[float] = [1.0, 0.5, 0.25],
        count_weight: float = 2.0,
        spatial_weight: float = 0.15,
        scale_loss_weight: float = 0.5
    ):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_loss_weight = scale_loss_weight
        
        # Base loss per ogni scala
        self.base_loss = P2RRegionLoss(
            count_weight=1.0,  # Pesi applicati dopo
            spatial_weight=1.0,
            scale_weight=1.0
        )
    
    def forward(
        self,
        pred_density: torch.Tensor,
        points_list: List[torch.Tensor],
        downscale: int = 16
    ) -> Dict[str, torch.Tensor]:
        """
        Calcola multi-scale loss.
        
        Args:
            pred_density: Predicted density [B, 1, H, W]
            points_list: Lista punti GT per batch
            downscale: Downscale factor
            
        Returns:
            Dict con losses aggregate
        """
        device = pred_density.device
        
        total_count_loss = torch.zeros(1, device=device).squeeze()
        total_spatial_loss = torch.zeros(1, device=device).squeeze()
        total_scale_loss = torch.zeros(1, device=device).squeeze()
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1:
                pred_scaled = pred_density
            else:
                # Average pooling
                pred_scaled = F.avg_pool2d(pred_density, kernel_size=scale, stride=scale)
                # Moltiplica per scale^2 per preservare il count totale
                pred_scaled = pred_scaled * (scale ** 2)
            
            # Adatta downscale per questa scala
            scaled_downscale = downscale * scale
            
            # Calcola loss a questa scala
            losses = self.base_loss(pred_scaled, points_list, scaled_downscale)
            
            total_count_loss = total_count_loss + weight * losses['count_loss']
            total_spatial_loss = total_spatial_loss + weight * losses['spatial_loss']
            total_scale_loss = total_scale_loss + weight * losses['scale_loss']
        
        # Normalizza per somma pesi
        weight_sum = sum(self.scale_weights)
        total_count_loss = total_count_loss / weight_sum
        total_spatial_loss = total_spatial_loss / weight_sum
        total_scale_loss = total_scale_loss / weight_sum
        
        # ============================================================
        # TOTAL LOSS con pesi V9
        # ============================================================
        total_loss = (
            self.count_weight * total_count_loss +
            self.spatial_weight * total_spatial_loss +
            self.scale_loss_weight * total_scale_loss
        )
        
        return {
            'total_loss': total_loss,
            'count_loss': total_count_loss,
            'spatial_loss': total_spatial_loss,
            'scale_loss': total_scale_loss
        }


def build_p2r_loss(cfg) -> nn.Module:
    """
    Factory function per costruire la loss da config.
    
    Args:
        cfg: Config dict/object con P2R_LOSS section
        
    Returns:
        Loss module
    """
    loss_cfg = cfg.get('P2R_LOSS', {})
    
    use_multiscale = loss_cfg.get('USE_MULTI_SCALE', True)
    
    if use_multiscale:
        return MultiScaleP2RLoss(
            scales=loss_cfg.get('SCALES', [1, 2, 4]),
            scale_weights=loss_cfg.get('SCALE_WEIGHTS', [1.0, 0.5, 0.25]),
            count_weight=loss_cfg.get('COUNT_WEIGHT', 2.0),
            spatial_weight=loss_cfg.get('SPATIAL_WEIGHT', 0.15),
            scale_loss_weight=loss_cfg.get('SCALE_WEIGHT', 0.5)
        )
    else:
        return P2RRegionLoss(
            count_weight=loss_cfg.get('COUNT_WEIGHT', 2.0),
            spatial_weight=loss_cfg.get('SPATIAL_WEIGHT', 0.15),
            scale_weight=loss_cfg.get('SCALE_WEIGHT', 0.5)
        )


# Test del modulo
if __name__ == "__main__":
    # Test P2RRegionLoss
    loss_fn = P2RRegionLoss(count_weight=2.0, spatial_weight=0.15)
    
    pred = torch.rand(2, 1, 24, 32) * 10  # Simula density
    points = [
        torch.tensor([[100, 150], [200, 250], [300, 100]]),  # 3 persone
        torch.tensor([[50, 60]])  # 1 persona
    ]
    
    losses = loss_fn(pred, points, downscale=16)
    
    print("=== P2RRegionLoss Test ===")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Count loss: {losses['count_loss'].item():.4f}")
    print(f"Spatial loss: {losses['spatial_loss'].item():.4f}")
    print(f"Scale loss: {losses['scale_loss'].item():.4f}")
    
    # Test MultiScaleP2RLoss
    ms_loss_fn = MultiScaleP2RLoss(
        scales=[1, 2, 4],
        scale_weights=[1.0, 0.5, 0.25],
        count_weight=2.0,
        spatial_weight=0.15
    )
    
    ms_losses = ms_loss_fn(pred, points, downscale=16)
    
    print("\n=== MultiScaleP2RLoss Test ===")
    print(f"Total loss: {ms_losses['total_loss'].item():.4f}")
    print(f"Count loss: {ms_losses['count_loss'].item():.4f}")
    print(f"Spatial loss: {ms_losses['spatial_loss'].item():.4f}")
    
    print("\n✅ P2R Loss V15 tests passed!")
