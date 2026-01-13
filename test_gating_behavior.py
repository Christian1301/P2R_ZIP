#!/usr/bin/env python3
"""
Test per verificare il comportamento reale del Feature Gating:
Dimostra che anche con feature mascherate (input=0), i bias delle convoluzioni
producono output NON-ZERO, richiedendo quindi l'output masking in eval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("🧪 TEST: Feature Gating vs Output Masking\n")
print("="*70)

# 1. Crea un semplice P2R-like head (2 convoluzioni)
class SimpleP2RHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

p2r_head = SimpleP2RHead()

# 2. Simula feature di backbone (4x4 spatial, 512 canali)
feat = torch.randn(1, 512, 4, 4)

# 3. Simula maschera ZIP: metà blocchi attivi, metà no
mask = torch.zeros(1, 1, 4, 4)
mask[0, 0, :2, :] = 1.0  # Solo metà superiore attiva
print(f"📊 Maschera ZIP:")
print(mask[0, 0])
print()

# 4. FEATURE GATING (come nel codice attuale)
gated_feat = feat * mask
print(f"📊 Feature dopo gating (dovrebbero essere 0 nella metà inferiore):")
print(f"   Metà superiore (attiva):  mean={gated_feat[0, :, 0, 0].mean():.4f}")
print(f"   Metà inferiore (mascherata): mean={gated_feat[0, :, 2, 0].mean():.4f} ✅ ~0")
print()

# 5. P2R su feature mascherate
with torch.no_grad():
    dens_gated = p2r_head(gated_feat)

print(f"📊 Densità P2R dopo Feature Gating:")
print(dens_gated[0, 0])
print(f"   Metà superiore (dovrebbe avere valori): mean={dens_gated[0, 0, :2, :].mean():.4f}")
print(f"   Metà inferiore (dovrebbe essere ~0):   mean={dens_gated[0, 0, 2:, :].mean():.4f}")
print()

# ⚠️ PROBLEMA: Anche con input=0, i BIAS producono output!
print("⚠️  PROBLEMA CRITICO:")
if abs(dens_gated[0, 0, 2:, :].mean().item()) > 1e-6:
    print(f"   ❌ La metà mascherata NON è zero! (bias effect)")
    print(f"   I bias delle convoluzioni producono: {dens_gated[0, 0, 2, 0].item():.6f}")
else:
    print(f"   ✅ La metà mascherata è effettivamente zero")
print()

# 6. OUTPUT MASKING (soluzione)
dens_masked = dens_gated * mask
print(f"📊 Densità dopo OUTPUT MASKING:")
print(dens_masked[0, 0])
print(f"   Metà inferiore dopo output mask: mean={dens_masked[0, 0, 2:, :].mean():.4f} ✅ Perfetto!")
print()

# 7. Confronto counts
count_gated = dens_gated.sum().item()
count_masked = dens_masked.sum().item()
print(f"📊 Count totale:")
print(f"   SOLO Feature Gating:  {count_gated:.4f}")
print(f"   Feature + Output Mask: {count_masked:.4f}")
print(f"   Differenza (bias leak): {count_gated - count_masked:.4f}")
print()

print("="*70)
print("🎯 CONCLUSIONE:")
print("   1. Feature Gating moltiplica input per 0 nelle zone mascherate")
print("   2. Ma Conv2d(0*x, w, b) = 0 + b = b (BIAS LEAK!)")
print("   3. Serve OUTPUT MASKING per azzerare completamente")
print("   4. In TRAINING: solo feature gating (permette gradiente)")
print("   5. In EVAL: feature gating + output masking (pulisce bias)")
print("="*70)
