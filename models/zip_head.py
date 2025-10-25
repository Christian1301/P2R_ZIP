# P2R_ZIP/models/zip_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ZIPHead(nn.Module):
    """
    ZIP Head migliorata:
    - λ (lambda) con range espanso e softplus
    - probabilità π bilanciate
    - regolarizzazione leggera su λ per maggiore variabilità
    """
    def __init__(
        self,
        in_ch: int,
        bins: List[Tuple[float, float]],
        lambda_scale: float = 1.0,       # ✅ scala iniziale aumentata
        lambda_max: float = 10.0,        # ✅ range massimo realistico per SHHA
        use_softplus: bool = True,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if not all(len(b) == 2 for b in bins):
            raise ValueError("I bin devono essere tuple di lunghezza 2 (es. [(0,0),(1,5),...])")
        if bins[0][0] != 0 or bins[0][1] != 0:
            raise ValueError("Il primo bin deve essere [0, 0]")

        self.bins = bins
        self.lambda_scale = lambda_scale
        self.lambda_max = lambda_max
        self.use_softplus = use_softplus
        self.epsilon = epsilon

        inter = max(64, in_ch // 4)

        # Blocchi convoluzionali condivisi
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )

        # Testa per π (probabilità blocco vuoto vs pieno)
        self.pi_head = nn.Conv2d(inter, 2, 1)

        # Testa per bin > 0 (distribuzione del conteggio)
        self.bin_head = nn.Conv2d(inter, len(bins) - 1, 1)

        # Parametro di bilanciamento (usato opzionalmente in loss)
        self.class_weights = torch.tensor([1.0, 3.0])  # ⬆️ peso maggiore ai blocchi vuoti

    def forward(self, feat: torch.Tensor, bin_centers: torch.Tensor):
        # Feature estratte dal backbone
        h = self.shared(feat)

        # Logits per π e per i bin
        logit_pi_maps = self.pi_head(h)     # [B, 2, Hb, Wb]
        logit_bin_maps = self.bin_head(h)   # [B, N-1, Hb, Wb]

        # --- λ (lambda) computation --------------------------------------------------
        centers = bin_centers
        if centers.dim() == 1:
            centers = centers.view(1, -1, 1, 1)

        # Escludi lo 0 (solo bin > 0)
        if centers.shape[1] == logit_bin_maps.shape[1] + 1:
            centers_positive = centers[:, 1:, :, :]
        else:
            centers_positive = centers

        assert centers_positive.shape[1] == logit_bin_maps.shape[1], \
            f"Dimensioni non corrispondenti: {centers_positive.shape[1]} vs {logit_bin_maps.shape[1]}"

        # Distribuzione dei bin
        p_bins = F.softmax(logit_bin_maps, dim=1)

        # Somma pesata dei centri dei bin
        lambda_raw = (p_bins * centers_positive).sum(dim=1, keepdim=True)

        # ✅ nuova formula per lambda
        if self.use_softplus:
            lambda_maps = F.softplus(lambda_raw * self.lambda_scale)
        else:
            lambda_maps = F.relu(lambda_raw * self.lambda_scale)

        # Clamp finale per sicurezza
        lambda_maps = torch.clamp(lambda_maps, min=self.epsilon, max=self.lambda_max)

        # ✅ Aggiunta leggera regolarizzazione stocastica
        if self.training:
            noise = torch.randn_like(lambda_maps) * 0.01
            lambda_maps = torch.clamp(lambda_maps + noise, min=self.epsilon, max=self.lambda_max)

        # Diagnostica (solo se necessario)
        if hasattr(self, "debug") and self.debug:
            print(f"[ZIPHead] λ range: [{lambda_maps.min().item():.3f}, {lambda_maps.max().item():.3f}], "
                  f"π logits mean: {logit_pi_maps.mean().item():.3f}")

        return {
            "logit_pi_maps": logit_pi_maps,   # [B, 2, Hb, Wb]
            "logit_bin_maps": logit_bin_maps, # [B, N-1, Hb, Wb]
            "lambda_maps": lambda_maps        # [B, 1, Hb, Wb]
        }
