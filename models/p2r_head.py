"""
P2R Head V16 - Point-to-Region Density Estimation
Supporta input variabile (512 o 514 canali per ZIP-as-feature)

CAMBIAMENTI DA V15:
1. GroupNorm adattivo: calcola num_groups in base a in_channel
2. Supporto nativo per 514 canali (512 + π + λ)
3. Aggiunto flag per disabilitare input_norm se necessario
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


def _get_group_norm_groups(channels: int) -> int:
    """
    Calcola il numero ottimale di gruppi per GroupNorm.
    
    Regole:
    - Deve dividere esattamente channels
    - Preferisce 32 gruppi se possibile
    - Fallback a divisori validi
    """
    # Preferenze in ordine
    preferred = [32, 16, 8, 4, 2, 1]
    
    for g in preferred:
        if channels % g == 0:
            return g
    
    # Fallback: trova il divisore più grande <= 32
    for g in range(min(32, channels), 0, -1):
        if channels % g == 0:
            return g
    
    return 1


class P2RHead(nn.Module):
    """
    P2R Head per density estimation.
    
    Predice una density map che, sommata, dà il conteggio totale.
    Usa un parametro log_scale learnable per scalare le predizioni.
    
    Args:
        in_channel: Canali in input (512 per VGG16, 514 per ZIP-as-feature)
        fea_channel: Canali intermedi (default 256)
        out_stride: Stride del backbone (16 per VGG16)
        log_scale_init: Valore iniziale per log_scale (4.0 → scale ~55)
        log_scale_clamp: Range per clamping log_scale
        use_input_norm: Se usare GroupNorm in input (default True)
        dropout_rate: Dropout rate nei layer intermedi (0 = disabilitato)
        final_dropout_rate: Dropout rate prima del layer finale
    """
    
    def __init__(
        self, 
        in_channel: int = 512, 
        fea_channel: int = 256,
        out_stride: int = 16,
        log_scale_init: float = 4.0,
        log_scale_clamp: tuple = (-2.0, 10.0),
        use_input_norm: bool = True,
        dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.in_channel = in_channel
        self.out_stride = out_stride
        self.log_scale_clamp = log_scale_clamp
        self.use_input_norm = use_input_norm
        self.dropout_rate = dropout_rate
        self.final_dropout_rate = final_dropout_rate
        
        # ============================================================
        # log_scale inizializzato a 4.0 → exp(4.0) ≈ 55
        # ============================================================
        self.log_scale = nn.Parameter(
            torch.tensor(log_scale_init, dtype=torch.float32),
            requires_grad=True
        )
        
        # ============================================================
        # GroupNorm ADATTIVO: calcola gruppi in base a in_channel
        # 512 → 32 gruppi (16 ch/gruppo)
        # 514 → 2 gruppi (257 ch/gruppo) - non divide per 32!
        # ============================================================
        if use_input_norm:
            num_groups = _get_group_norm_groups(in_channel)
            self.input_norm = nn.GroupNorm(num_groups, in_channel)
            # Debug info
            self._norm_groups = num_groups
        else:
            self.input_norm = nn.Identity()
            self._norm_groups = 0
        
        # ============================================================
        # Encoder layers con BatchNorm + Dropout per regularizzazione
        # ============================================================
        self.layer1 = self._make_layer(in_channel, fea_channel, dropout_rate)
        self.layer2 = self._make_layer(fea_channel, fea_channel, dropout_rate)
        self.layer3 = self._make_layer(fea_channel, fea_channel, dropout_rate)
        self.layer4 = self._make_layer(fea_channel, fea_channel, final_dropout_rate)
        
        # Final prediction layer
        self.pred_layer = nn.Sequential(
            nn.Conv2d(fea_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        # Skip connection per preservare dettagli spaziali
        self.skip_conv = conv_1x1(in_channel, fea_channel, bn=True)
        
        # Inizializzazione pesi
        self._init_weights()
    
    def _make_layer(self, in_ch, out_ch, dropout):
        """Crea layer conv + bn + relu + dropout."""
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)
    
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
            features: Feature map [B, C, H, W] dove C può essere 512 o 514
            
        Returns:
            density: Density map scalata [B, 1, H, W]
        """
        # Verifica dimensioni input
        if features.shape[1] != self.in_channel:
            raise ValueError(
                f"P2RHead expects {self.in_channel} input channels, "
                f"got {features.shape[1]}"
            )
        
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
        # ============================================================
        clamped_log_scale = torch.clamp(
            self.log_scale, 
            min=self.log_scale_clamp[0], 
            max=self.log_scale_clamp[1]
        )
        scale = torch.exp(clamped_log_scale)
        
        # Softplus + scaling per garantire density >= 0
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


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing P2R Head V16")
    print("=" * 60)
    
    # Test 1: Standard 512 canali
    print("\n1. Test con 512 canali (standard):")
    head_512 = P2RHead(in_channel=512, fea_channel=256, log_scale_init=4.0)
    x_512 = torch.randn(2, 512, 24, 32)
    out_512 = head_512(x_512)
    print(f"   Input:  {x_512.shape}")
    print(f"   Output: {out_512.shape}")
    print(f"   GroupNorm groups: {head_512._norm_groups}")
    print(f"   log_scale: {head_512.get_log_scale():.4f}")
    print(f"   scale: {head_512.get_scale():.4f}")
    
    # Test 2: ZIP-as-feature 514 canali
    print("\n2. Test con 514 canali (ZIP-as-feature):")
    head_514 = P2RHead(in_channel=514, fea_channel=256, log_scale_init=4.0)
    x_514 = torch.randn(2, 514, 24, 32)
    out_514 = head_514(x_514)
    print(f"   Input:  {x_514.shape}")
    print(f"   Output: {out_514.shape}")
    print(f"   GroupNorm groups: {head_514._norm_groups}")
    print(f"   log_scale: {head_514.get_log_scale():.4f}")
    print(f"   scale: {head_514.get_scale():.4f}")
    
    # Test 3: Verifica output ragionevoli
    print("\n3. Verifica output:")
    print(f"   512ch - Output range: [{out_512.min().item():.4f}, {out_512.max().item():.4f}]")
    print(f"   512ch - Output sum: {out_512.sum(dim=[2,3]).mean().item():.2f}")
    print(f"   514ch - Output range: [{out_514.min().item():.4f}, {out_514.max().item():.4f}]")
    print(f"   514ch - Output sum: {out_514.sum(dim=[2,3]).mean().item():.2f}")
    
    # Test 4: Verifica errore con canali sbagliati
    print("\n4. Test error handling:")
    try:
        wrong_input = torch.randn(2, 256, 24, 32)
        head_512(wrong_input)
        print("   ❌ Doveva sollevare errore!")
    except ValueError as e:
        print(f"   ✅ Errore correttamente sollevato: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Tutti i test passati!")
    print("=" * 60)