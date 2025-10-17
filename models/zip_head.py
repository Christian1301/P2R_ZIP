# P2R_ZIP/models/zip_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Aggiunto per F.softplus
from typing import List, Tuple

class ZIPHead(nn.Module):
    def __init__(self, in_ch: int, bins: List[Tuple[float, float]], lambda_scale: float = 0.01, lambda_max: float = 80.0): # Aggiunti parametri per controllo lambda
        super().__init__()
        if not all(len(b) == 2 for b in bins):
            raise ValueError(f"I bin devono essere una lista di tuple di lunghezza 2, ma abbiamo ricevuto {bins}")
        if bins[0][0] != 0 or bins[0][1] != 0:
             raise ValueError("Il primo bin deve essere [0, 0]")
        self.bins = bins
        self.lambda_scale = lambda_scale
        self.lambda_max = lambda_max

        inter = max(64, in_ch // 4)
        # Strato condiviso iniziale
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )

        # Testa per Pi (probabilità zero-inflation) -> 2 classi: Zero vs Non-Zero
        self.pi_head = nn.Conv2d(inter, 2, 1)
        # Testa per i Bin (classificazione del conteggio > 0) -> N-1 classi
        self.bin_head = nn.Conv2d(inter, len(self.bins) - 1, 1)

        # --- Aggiunta Opzionale: Testa diretta per Lambda ---
        # A volte predire lambda direttamente (con Softplus) è più stabile
        # che calcolarlo dai bin. Puoi sperimentare attivando questa.
        # self.lambda_head = nn.Conv2d(inter, 1, 1)
        # --- Fine Aggiunta Opzionale ---


    def forward(self, feat, bin_centers: torch.Tensor):
        h = self.shared(feat)
        logit_pi_maps = self.pi_head(h)     # [B, 2, Hb, Wb]
        logit_bin_maps = self.bin_head(h)   # [B, N-1, Hb, Wb]

        # --- Calcolo Lambda dai Bin (Metodo Originale + Scaling/Clamp) ---
        centers = bin_centers
        # Assicura che i centers abbiano la forma corretta [1, N_centers, 1, 1]
        if centers.dim() == 1:
            centers = centers.view(1, -1, 1, 1)
        # Se i centers includono lo 0 (hanno N elementi totali), rimuovi lo 0
        # perché logit_bin_maps predice solo per i bin > 0 (N-1 elementi)
        if centers.shape[1] == logit_bin_maps.shape[1] + 1:
            centers_positive = centers[:, 1:, :, :] # Prende da indice 1 in poi
        elif centers.shape[1] == logit_bin_maps.shape[1]:
             centers_positive = centers # Assume che i centers passati siano già solo quelli positivi
        else:
            raise ValueError(f"Incongruenza dimensioni bin_centers ({centers.shape[1]}) vs logit_bin_maps ({logit_bin_maps.shape[1]})")

        # Verifica finale dimensioni
        assert centers_positive.shape[1] == logit_bin_maps.shape[1], \
            f"Dimensioni non corrispondenti: centers_positive {centers_positive.shape[1]} vs logits {logit_bin_maps.shape[1]}"

        # Calcola le probabilità per i bin > 0
        p_bins = logit_bin_maps.softmax(dim=1) # [B, N-1, Hb, Wb]

        # Calcola lambda come somma pesata dei centri dei bin > 0
        lambda_maps_raw = (p_bins * centers_positive).sum(dim=1, keepdim=True) # [B, 1, Hb, Wb]

        # --- PATCH: Applica Scaling e Clamp a Lambda ---
        # 1. (Opzionale ma consigliato) Usa Softplus per garantire positività in modo smooth
        #    Alternativa a usare direttamente lambda_maps_raw se i centers sono già > 0
        # lambda_maps = F.softplus(lambda_maps_raw) # Applica Softplus

        # 2. Scala per mantenere i valori iniziali bassi
        lambda_maps = lambda_maps_raw * self.lambda_scale # Usa lambda_maps_raw se non usi Softplus sopra

        # 3. Clampa al valore massimo specifico per il dataset (es. 80 per SHHA)
        lambda_maps = torch.clamp(lambda_maps, min=1e-6, max=self.lambda_max) # Aggiunto min per evitare log(0)
        # --- FINE PATCH ---


        # --- Alternativa: Testa Lambda Diretta (se attivata nell'init) ---
        # if hasattr(self, 'lambda_head'):
        #     lambda_out = self.lambda_head(h)
        #     lambda_maps = F.softplus(lambda_out) # Softplus per garantire positività
        #     lambda_maps = lambda_maps * self.lambda_scale # Scaling
        #     lambda_maps = torch.clamp(lambda_maps, min=1e-6, max=self.lambda_max) # Clamping
        # --- Fine Alternativa ---


        return {
            "logit_pi_maps": logit_pi_maps,         # Logits per [Zero, NonZero]
            "logit_bin_maps": logit_bin_maps,       # Logits per Bin [1...N-1]
            "lambda_maps": lambda_maps              # Lambda finale (scalato e clampato)
        }