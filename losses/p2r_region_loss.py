# p2r_region_loss.py
# -*- coding: utf-8 -*-

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Opzioni di debug/sicurezza
# ============================================================
# Metti a True se vuoi forzare la sincronizzazione per ottenere stacktrace accurati
ENABLE_CUDA_LAUNCH_BLOCKING = True
# Metti a True per attivare l'anomaly detection (lento ma utile per localizzare NaN/Inf/backward bug)
ENABLE_AUTOGRAD_ANOMALY_DETECT = False
# Log di debug lightweight (dimensioni, num punti, range densità, ecc.)
ENABLE_LIGHT_DEBUG_LOG = True

if ENABLE_CUDA_LAUNCH_BLOCKING:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if ENABLE_AUTOGRAD_ANOMALY_DETECT:
    torch.autograd.set_detect_anomaly(True)


# ============================================================
# Utility
# ============================================================

def _safe_normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizza in [0,1] una mappa non negativa.
    Se max==0 restituisce lo zero tensor.
    """
    x = torch.clamp(x, min=0.0)
    xmax = torch.amax(x)
    if torch.isfinite(xmax) and xmax > 0:
        return x / (xmax + eps)
    return torch.zeros_like(x)


def _check_no_nan_inf(t: torch.Tensor, name: str = "tensor"):
    if torch.isnan(t).any():
        raise ValueError(f"{name} contiene NaN")
    if torch.isinf(t).any():
        raise ValueError(f"{name} contiene Inf")


# ============================================================
# L2 Distance (streaming)
# ============================================================

class L2DIS:
    """
    Calcola distanze L2 normalizzate tra insiemi di punti:
      - X: [B, NX, 2]
      - Y: [B, NY, 2]
    Ritorna: [B, NX, NY]
    """
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: [B,NX,2] -> [...,1,2]
        # Y: [B,NY,2] -> [...,2]
        # broadcasting: (B, NX, 1, 2) - (B, 1, NY, 2) -> (B, NX, NY, 2)
        x_col = X.unsqueeze(-2)   # [B, NX, 1, 2]
        y_row = Y.unsqueeze(-3)   # [B, 1, NY, 2]
        return torch.norm(x_col - y_row, dim=-1) / max(self.factor, 1e-6)


# ============================================================
# P2R Loss (robusta)
# ============================================================

class P2RLoss(nn.Module):
    """
    Point-to-Region Loss (versione robusta e paper-consistent).

    - BCE su mappa di densità continua (normalizzata a [0,1]).
    - Penalità di scala sul conteggio globale (MAE tra pred e GT).
    - Gestione esplicita batch con zero punti.
    - Controlli su NaN/Inf, forme, e clamp dei punti dentro immagine.
    - Calcolo delle min-distanze per chunk per ridurre memoria.

    Args:
        reduction: "mean" o "sum" (default "mean")
        chunk_size: numero di pixel per chunk nel matching punto-pixel
        scale_weight: peso della penalità di scala sui conteggi globali
        min_radius: soglia in pixel (nello spazio input) per classificare un pixel come positivo (vicino a un punto)
        max_radius: clamp superiore per la scala distanze (utile a stabilizzare i pesi)
        cost_point: peso del costo punto
        cost_class: peso del costo class (se in futuro si estende alla formulazione con costi)
    """
    def __init__(
        self,
        reduction: str = "mean",
        chunk_size: int = 4096,
        scale_weight: float = 0.02,
        min_radius: float = 8.0,
        max_radius: float = 96.0,
        cost_point: float = 8.0,
        cost_class: float = 1.0,
    ):
        super().__init__()
        self.cost = L2DIS(1.0)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.reduction = reduction
        self.chunk_size = int(chunk_size)
        self.scale_weight = float(scale_weight)

    @torch.no_grad()
    def _min_distances_streaming(
        self,
        A_coord: torch.Tensor,  # [1, HW, 2]
        B_coord: torch.Tensor,  # [1, NP, 2]
        chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola, per ogni pixel (A_coord), la minima distanza verso i punti (B_coord)
        in modalità streaming per ridurre i picchi di memoria.
        Ritorna:
            minC:  [1, HW, 1]
            mcidx: [1, HW, 1]  (indici del punto più vicino)
        """
        device = A_coord.device
        bsz, n_pix, _ = A_coord.shape
        _, n_pts, _ = B_coord.shape
        assert bsz == 1, "Questa implementazione streaming assume batch=1 per la parte di matching fine."

        minC_list = []
        mcidx_list = []

        # Caso particolare: nessun punto -> ritorna distanza molto alta e indice 0
        if n_pts == 0:
            very_large = torch.full((1, n_pix, 1), float('inf'), device=device)
            zero_idx = torch.zeros((1, n_pix, 1), dtype=torch.long, device=device)
            return very_large, zero_idx

        for start in range(0, n_pix, chunk_size):
            end = min(start + chunk_size, n_pix)
            # [1, chunk, 2] vs [1, n_pts, 2] -> [1, chunk, n_pts]
            C_chunk = self.cost(A_coord[:, start:end, :], B_coord)
            minC, mcidx = torch.min(C_chunk, dim=-1, keepdim=True)  # [1, chunk, 1]
            minC_list.append(minC)
            mcidx_list.append(mcidx)
            # Rilascia subito il chunk
            del C_chunk, minC, mcidx

        return torch.cat(minC_list, dim=1), torch.cat(mcidx_list, dim=1)

    def forward(
        self,
        dens: torch.Tensor,                  # [B, 1, H_out, W_out] (ReLU/(upscale^2))
        points,                              # lista di B tensori [Ni, >=2] (x,y,[...]) in coord spazio input
        down = 16,                           # fattore di downsampling: input/H_out (scalar o tuple)
        masks: Optional[torch.Tensor] = None,
        crop_den_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola la P2R loss.

                Note importanti:
                - Le coordinate dei punti devono essere nello spazio input (H_in x W_in).
                - Le mappe di densità sono nello spazio ridotto (H_out x W_out).
                - Per allineare i due spazi, generiamo i centri pixel nello spazio input con:
                            rr_in = rr * down_h + (down_h - 1) / 2
                            cc_in = cc * down_w + (down_w - 1) / 2
                - T (target binario) si ottiene confrontando la distanza minima con min_radius (in pixel input).
        """
        device = dens.device
        _check_no_nan_inf(dens, "dens")

        if isinstance(down, (list, tuple)):
            if len(down) != 2:
                raise ValueError(f"down come sequenza deve avere due elementi, trovato {down}")
            down_h, down_w = float(down[0]), float(down[1])
        else:
            down_h = down_w = float(down)

        if down_h <= 0 or down_w <= 0:
            raise ValueError(f"down deve essere positivo, trovato down_h={down_h}, down_w={down_w}")

        B = len(points)
        assert dens.ndim == 4 and dens.shape[1] == 1, f"dens deve essere [B,1,H,W], trovato {tuple(dens.shape)}"
        assert B == dens.shape[0], f"batch mismatch: len(points)={B} vs dens.shape[0]={dens.shape[0]}"
        if crop_den_masks is not None:
            assert crop_den_masks.shape == dens.shape, \
                f"crop_den_masks deve avere shape {dens.shape}, trovato {tuple(crop_den_masks.shape)}"

        total_loss = dens.new_tensor(0.0)
        pred_counts = []
        gt_counts = []

        for i in range(B):
            # dens[i]: [1, H, W] -> den: [H, W, 1]
            den = dens[i].permute(1, 2, 0).contiguous()  # [H, W, 1]
            H, W = den.shape[:2]

            # Debug lightweight
            if ENABLE_LIGHT_DEBUG_LOG and i == 0:
                with torch.no_grad():
                    print(f"[P2R DEBUG] B={B}, HxW={H}x{W}, down=({down_h:.3f},{down_w:.3f}), "
                          f"den_range=[{den.min().item():.4e},{den.max().item():.4e}], "
                          f"den_mean={den.mean().item():.6f}")

            seq: torch.Tensor = points[i]
            if not torch.is_tensor(seq):
                raise TypeError(f"points[{i}] deve essere torch.Tensor, ottenuto {type(seq)}")

            # Normalizziamo forma e tipo
            if seq.ndim == 1:
                # es: shape [2] -> reshape a [1,2]
                if seq.numel() == 0:
                    seq = seq.reshape(0, 2)
                else:
                    seq = seq.reshape(1, -1)
            if seq.numel() > 0 and seq.shape[-1] < 2:
                raise ValueError(f"points[{i}] deve avere almeno 2 colonne (x,y). Shape: {tuple(seq.shape)}")

            seq = seq.to(device, dtype=torch.float32)

            # Gestione batch con 0 punti
            if seq.numel() == 0 or seq.shape[0] == 0:
                # BCE contro tutto-zero (peso 0.5 come nel tuo codice)
                loss_empty = F.binary_cross_entropy(
                    _safe_normalize_01(den),
                    torch.zeros_like(den),
                    weight=torch.ones_like(den) * 0.5,
                    reduction="mean",
                )
                total_loss = total_loss + loss_empty
                pred_counts.append(dens[i].sum() / (down_h * down_w))
                gt_counts.append(dens.new_tensor(0.0))
                if ENABLE_LIGHT_DEBUG_LOG:
                    print(f"[P2R DEBUG] i={i} senza punti → loss_empty={loss_empty.item():.6f}")
                continue

            # Check NaN/Inf sui punti
            _check_no_nan_inf(seq, f"points[{i}]")

            # Clamp dei punti nel dominio input (HxW nello spazio input = H*down, W*down)
            H_in = H * down_h
            W_in = W * down_w
            # seq: [N, >=2], coordinate [x, y] in seq[:, :2]
            # Convenzione comune: seq[:,0]=x (colonna), seq[:,1]=y (riga). Qui NON le scambiamo,
            # usiamo coerenza interna: A_coord = (row, col) nello spazio input, B_coord = (y, x).
            seq[:, 0] = torch.clamp(seq[:, 0], 0, W_in - 1)  # x in [0, W_in-1]
            seq[:, 1] = torch.clamp(seq[:, 1], 0, H_in - 1)  # y in [0, H_in-1]

            # Costruzione A (feature su pixel), shape [1, HW, 1]
            A = den.view(1, -1, 1)
            _check_no_nan_inf(A, "A(den_flat)")

            # Coordinate dei pixel (centri) nello spazio input
            # meshgrid in ordine (row=i, col=j)
            rows = torch.arange(H, device=device, dtype=torch.float32)
            cols = torch.arange(W, device=device, dtype=torch.float32)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")  # [H,W], [H,W]
            # Centro del pixel in input-space:
            #   rr_in = rr*down + (down-1)/2,  cc_in = cc*down + (down-1)/2
            center_offset_h = (down_h - 1.0) / 2.0
            center_offset_w = (down_w - 1.0) / 2.0
            rr_in = rr * down_h + center_offset_h
            cc_in = cc * down_w + center_offset_w
            A_coord = torch.stack([rr_in, cc_in], dim=-1).view(1, -1, 2)  # [1, HW, 2]

            # Coordinate dei punti nello spazio input (B_coord usa [y, x] coerente con rr/cc)
            B_coord = torch.stack([seq[:, 1], seq[:, 0]], dim=-1).unsqueeze(0)  # [1, N, 2]

            # Calcolo minima distanza punto-pixel in streaming
            minC, mcidx = self._min_distances_streaming(A_coord, B_coord, self.chunk_size)
            # Stabilizza
            maxC = torch.clamp(torch.amax(minC, dim=1, keepdim=True), self.min_radius, self.max_radius)

            # Target binario T: pixel positivo se vicino a un punto (< min_radius nel dominio input)
            T = (minC < self.min_radius).float().view_as(A)  # [1, HW, 1]
            # Pesi (semplici): 1 su positivi, 1 su negativi (o 2 su pos se vuoi)
            Wt = T + 1.0  # enfatizza i positivi (pos=2, neg=1)
            if crop_den_masks is not None:
                Wt = Wt * crop_den_masks[i].view_as(Wt)

            # Normalizzazione della densità per la BCE (range [0,1])
            A_norm = _safe_normalize_01(A)

            # BCE
            loss_i = F.binary_cross_entropy(
                A_norm, T, weight=Wt, reduction="mean"
            )
            total_loss = total_loss + loss_i

            # Penalità di scala sui conteggi (entrambe le quantità in "count di persone")
            # dens è nello spazio ridotto: somma(dens) ≈ count*(1/(down_h*down_w)) → riportiamo a count dividendo di (down_h*down_w)
            pred_counts.append(dens[i].sum() / (down_h * down_w))
            gt_counts.append(dens.new_tensor(float(seq.shape[0])))

            if ENABLE_LIGHT_DEBUG_LOG and i == 0:
                with torch.no_grad():
                    print(f"[P2R DEBUG] i={i} points={seq.shape[0]}, "
                          f"loss_bce={loss_i.item():.6f}, "
                          f"A_norm_range=[{A_norm.min().item():.4e},{A_norm.max().item():.4e}]")

            # cleanup chunk
            del A_norm, A_coord, B_coord, minC, mcidx, T, Wt, rr, cc, rr_in, cc_in, center_offset_h, center_offset_w

        # Penalità globale sui conteggi
        if len(pred_counts) > 0:
            pred_counts_t = torch.stack(pred_counts)  # [B]
            gt_counts_t = torch.stack(gt_counts)      # [B]
            _check_no_nan_inf(pred_counts_t, "pred_counts")
            _check_no_nan_inf(gt_counts_t, "gt_counts")

            scale_penalty = self.scale_weight * torch.abs(pred_counts_t - gt_counts_t).mean()
            total_loss = total_loss + 1.0 * scale_penalty
        else:
            scale_penalty = dens.new_tensor(0.0)

        # Riduzione per batch
        if self.reduction == "mean":
            total_loss = total_loss / max(B, 1)

        # Debug finale
        if ENABLE_LIGHT_DEBUG_LOG:
            with torch.no_grad():
                pc = torch.stack(pred_counts).mean().item() if pred_counts else 0.0
                gc = torch.stack(gt_counts).mean().item() if gt_counts else 0.0
                print(f"[P2R DEBUG] total_loss={total_loss.item():.6f}, "
                      f"scale_penalty={scale_penalty.item():.6f}, "
                      f"pred_count(avg)={pc:.3f}, gt_count(avg)={gc:.3f}")

        return total_loss
