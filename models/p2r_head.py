"""
P2R Head V15 - Point-to-Region Density Estimation
Basato su V9 (MAE ~69) con correzioni critiche

CORREZIONI RISPETTO A V14:
1. log_scale inizializzato a 4.0 (era -1.0)
2. Aggiunto GroupNorm per stabilità features
3. BatchNorm nei layer conv
4. Clamping log_scale per evitare instabilità
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_channels, out_channels, stride=1, padding=1, bn=True):
    """Conv 3x3 con opzionale BatchNorm"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                  stride=stride, padding=padding, bias=not bn)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_1x1(in_channels, out_channels, bn=False):
    """Conv 1x1 per riduzione canali"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not bn)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class P2RHead(nn.Module):
    """
    P2R Head per density estimation.
    
    Predice una density map che, sommata, dà il conteggio totale.
    Usa un parametro log_scale learnable per scalare le predizioni.
    
    Args:
        in_channel: Canali in input dal backbone (512 per VGG16)
        fea_channel: Canali intermedi (default 256)
        out_stride: Stride del backbone (16 per VGG16)
        log_scale_init: Valore iniziale per log_scale (CRITICO: 4.0!)
        log_scale_clamp: Range per clamping log_scale
    """
    
    def __init__(
        self, 
        in_channel: int = 512, 
        fea_channel: int = 256,
        out_stride: int = 16,
        log_scale_init: float = 4.0,      # CRITICO: era -1.0 in V14!
        log_scale_clamp: tuple = (-2.0, 10.0)
    ):
        super().__init__()
        
        self.out_stride = out_stride
        self.log_scale_clamp = log_scale_clamp
        
        # ============================================================
        # CORREZIONE CRITICA: log_scale inizializzato a 4.0
        # exp(4.0) ≈ 55, ragionevole per crowd counting
        # V14 usava -1.0 → exp(-1.0) ≈ 0.37, TROPPO BASSO!
        # ============================================================
        self.log_scale = nn.Parameter(
            torch.tensor(log_scale_init, dtype=torch.float32),
            requires_grad=True
        )
        
        # ============================================================
        # CORREZIONE: GroupNorm per stabilizzare features in input
        # Normalizza le feature dal backbone prima di processarle
        # ============================================================
        self.input_norm = nn.GroupNorm(32, in_channel)
        
        # ============================================================
        # Encoder layers con BatchNorm per stabilità
        # ============================================================
        self.layer1 = conv_3x3(in_channel, fea_channel, bn=True)
        self.layer2 = conv_3x3(fea_channel, fea_channel, bn=True)
        self.layer3 = conv_3x3(fea_channel, fea_channel, bn=True)
        self.layer4 = conv_3x3(fea_channel, fea_channel, bn=True)
        
        # Final prediction layer (no activation, can be negative before scale)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(fea_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        # Skip connection per preservare dettagli spaziali
        self.skip_conv = conv_1x1(in_channel, fea_channel, bn=True)
        
        # Inizializzazione pesi
        self._init_weights()
    
    def _init_weights(self):
        """Inizializza i pesi con Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Feature map dal backbone [B, C, H, W]
            
        Returns:
            density: Density map scalata [B, 1, H, W]
        """
        # Input normalization per stabilità
        x = self.input_norm(features)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        # Encoder path
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Add skip connection
        x = x + skip
        
        # Raw density prediction (può essere negativo)
        raw_density = self.pred_layer(x)
        
        # ============================================================
        # Scala la density con exp(log_scale)
        # Con log_scale=4.0, scale≈55, valori ragionevoli per counting
        # ============================================================
        clamped_log_scale = torch.clamp(
            self.log_scale, 
            min=self.log_scale_clamp[0], 
            max=self.log_scale_clamp[1]
        )
        scale = torch.exp(clamped_log_scale)
        
        # ReLU dopo scaling per garantire density >= 0
        density = F.softplus(raw_density) * scale
        
        return density
    
    def get_scale(self) -> float:
        """Ritorna il valore corrente di scale (exp(log_scale))"""
        with torch.no_grad():
            clamped = torch.clamp(
                self.log_scale,
                min=self.log_scale_clamp[0],
                max=self.log_scale_clamp[1]
            )
            return torch.exp(clamped).item()
    
    def get_log_scale(self) -> float:
        """Ritorna il valore corrente di log_scale"""
        with torch.no_grad():
            return self.log_scale.item()
    
    def set_log_scale(self, value: float):
        """Imposta manualmente log_scale (per calibrazione)"""
        with torch.no_grad():
            self.log_scale.fill_(value)


class P2RHeadMultiScale(P2RHead):
    """
    Versione multi-scala del P2R Head.
    Combina predizioni a diverse risoluzioni per robustezza.
    """
    
    def __init__(
        self,
        in_channel: int = 512,
        fea_channel: int = 256,
        out_stride: int = 16,
        log_scale_init: float = 4.0,
        log_scale_clamp: tuple = (-2.0, 10.0),
        scales: list = [1, 2, 4]
    ):
        super().__init__(
            in_channel=in_channel,
            fea_channel=fea_channel,
            out_stride=out_stride,
            log_scale_init=log_scale_init,
            log_scale_clamp=log_scale_clamp
        )
        self.scales = scales
    
    def forward_multiscale(self, features: torch.Tensor) -> dict:
        """
        Forward pass con output multi-scala.
        
        Returns:
            dict con 'density' e 'multiscale_densities'
        """
        # Density principale
        density = self.forward(features)
        
        # Versioni a diverse scale
        multiscale = {}
        for s in self.scales:
            if s == 1:
                multiscale[s] = density
            else:
                # Average pooling preservando il count totale
                pooled = F.avg_pool2d(density, kernel_size=s, stride=s)
                # Moltiplica per s^2 per preservare il count
                multiscale[s] = pooled * (s ** 2)
        
        return {
            'density': density,
            'multiscale_densities': multiscale
        }


# Test del modulo
if __name__ == "__main__":
    # Test base
    head = P2RHead(in_channel=512, fea_channel=256, log_scale_init=4.0)
    x = torch.randn(2, 512, 24, 32)
    out = head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"log_scale: {head.get_log_scale():.4f}")
    print(f"scale: {head.get_scale():.4f}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"Output sum (simulated count): {out.sum(dim=[2,3]).mean().item():.2f}")
    
    # Verifica che con log_scale=4.0 le predizioni siano ragionevoli
    assert head.get_scale() > 50, "Scale dovrebbe essere ~55 con log_scale=4.0"
    print("\n✅ P2RHead V15 test passed!")
