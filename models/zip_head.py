# P2R_ZIP/models/zip_head.py
import torch
import torch.nn as nn

class ZIPHead(nn.Module):
    """
    Testa ZIP semplificata: due rami 1x1 che stimano:
      - pi: prob. blocco occupato (sigmoid)
      - lam: intensità Poisson (relu)
    """
    def __init__(self, in_ch: int):
        super().__init__()
        inter = max(64, in_ch // 4)
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        self.pi_head = nn.Conv2d(inter, 1, 1)
        self.lam_head = nn.Conv2d(inter, 1, 1)

    def forward(self, feat):
        h = self.shared(feat)
        pi = torch.sigmoid(self.pi_head(h))
        lam = torch.relu(self.lam_head(h))
        return pi, lam
