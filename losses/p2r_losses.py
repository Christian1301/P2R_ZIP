# P2R_ZIP/losses/p2r_losses.py
import torch
import torch.nn.functional as F

def gaussian_kernel_density(points, H, W, sigma=4.0, device="cpu"):
    """
    Crea una mappa densità da un set di punti (x,y) in pixel.
    points: lista di tensori di shape [Ni, 2] (per batch) oppure un tensore [N,2].
    Restituisce: D [1,H,W] per immagine.
    """
    yy, xx = torch.meshgrid(
        torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing="ij"
    )
    grid = torch.stack([xx, yy], dim=-1).float()  # [H,W,2]
    D = torch.zeros(H, W, device=device)
    if points is None or (isinstance(points, torch.Tensor) and points.numel() == 0):
        return D.unsqueeze(0)

    if isinstance(points, list):
        pts = points
    else:
        pts = [points]

    for p in pts:
        if p.numel() == 0:
            continue
        # p: [K,2] in (x,y)
        px = p[:, 0].view(-1, 1, 1)
        py = p[:, 1].view(-1, 1, 1)
        dist2 = (grid[..., 0] - px) ** 2 + (grid[..., 1] - py) ** 2  # [K,H,W]
        k = torch.exp(-0.5 * dist2 / (sigma ** 2)) / (2 * 3.14159265 * sigma ** 2)
        D = D + k.sum(dim=0)
    return D.unsqueeze(0)  # [1,H,W]

def p2r_density_mse(pred_density, points, sigma=4.0, count_l1_w=0.0):
    """
    Loss P2R semplificata: MSE tra densità e KDE dei punti + opzionale L1 sul conteggio.
    pred_density: [B,1,H,W]
    points: lista di tensori [Ni,2] per immagine
    """
    B, _, H, W = pred_density.shape
    device = pred_density.device
    dens_gt = []
    for i in range(B):
        pts = points[i] if isinstance(points, list) else points
        dens_gt.append(gaussian_kernel_density(pts, H, W, sigma, device))
    dens_gt = torch.stack(dens_gt, dim=0)  # [B,1,H,W]

    mse = F.mse_loss(pred_density, dens_gt)

    loss = mse
    if count_l1_w > 0:
        pred_c = pred_density.sum(dim=[1,2,3])
        gt_c = dens_gt.sum(dim=[1,2,3])
        loss = loss + count_l1_w * torch.mean(torch.abs(pred_c - gt_c))
    return loss
