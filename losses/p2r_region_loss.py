import torch
import torch.nn as nn
import torch.nn.functional as F

class L2DIS:
    def __init__(self, factor=512):
        self.factor = factor

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        return torch.norm(x_col - y_row, dim=-1) / self.factor


class P2RLoss(nn.Module):
    """
    Versione ottimizzata della Point-to-Region Loss:
    - Calcolo chunked e streaming
    - Memoria costante anche con immagini grandi
    - Stessa formulazione logica del paper
    """
    def __init__(self, reduction="mean", chunk_size=4096):
        super().__init__()
        self.cost = L2DIS(1)
        self.min_radius = 8
        self.max_radius = 96
        self.cost_class = 1
        self.cost_point = 8
        self.reduction = reduction
        self.chunk_size = chunk_size

    @torch.no_grad()
    def _min_distances(self, A_coord, B_coord):
        """Calcola distanza minima punto-pixel in streaming."""
        Npix = A_coord.shape[1]
        minC_list = []
        mcidx_list = []
        for start in range(0, Npix, self.chunk_size):
            end = min(start + self.chunk_size, Npix)
            C_chunk = self.cost(A_coord[:, start:end, :], B_coord)
            minC, mcidx = C_chunk.min(dim=-1, keepdim=True)
            minC_list.append(minC)
            mcidx_list.append(mcidx)
            del C_chunk
        return torch.cat(minC_list, dim=1), torch.cat(mcidx_list, dim=1)

    def forward(self, dens, points, down=16, masks=None, crop_den_masks=None):
        B = len(points)
        total_loss = 0.0
        device = dens.device

        for i in range(B):
            den = dens[i].permute(1, 2, 0)  # [H,W,1]
            seq = points[i]
            if seq.numel() == 0:
                total_loss += F.binary_cross_entropy_with_logits(
                    den, torch.zeros_like(den), weight=torch.ones_like(den) * 0.5
                )
                continue

            H, W = den.shape[:2]
            A = den.view(1, -1, 1)
            A_coord = (
                torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)
                .view(1, -1, 2)
                .float()
                .to(device)
            ) * down + (down - 1) / 2
            B_coord = seq[None, :, :2].float().to(device)

            # --- Calcolo minimo a blocchi ---
            minC, mcidx = self._min_distances(A_coord, B_coord)
            maxC = torch.clamp(minC.amax(dim=1, keepdim=True), self.min_radius, self.max_radius)

            # --- Maschere e normalizzazione ---
            M = (minC < self.max_radius).float()
            Cnorm = (minC / maxC) * self.cost_point - A * self.cost_class

            # --- Bersaglio T e pesi Wt (approssimazione paper) ---
            T = (minC < self.min_radius).float().view_as(A)
            Wt = T + 1.0
            if crop_den_masks is not None:
                Wt = Wt * crop_den_masks[i].view_as(Wt)

            # --- BCE su mappa flattenata ---
            loss_i = F.binary_cross_entropy_with_logits(A, T, weight=Wt)
            total_loss += loss_i

            del A, A_coord, B_coord, minC, mcidx, M, Cnorm, T, Wt
            torch.cuda.empty_cache()

        if self.reduction == "mean":
            total_loss = total_loss / B
        return total_loss
