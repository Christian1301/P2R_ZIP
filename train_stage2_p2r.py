#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V13 - P2R Training con Advanced Losses

STRATEGIA:
- Architettura P2RHead ORIGINALE (che funziona, MAE ~72)
- Advanced Losses: Distribution Matching + Optimal Transport
- Training più lungo con cosine restart
- Nessuna modifica all'architettura

TARGET: MAE < 60 partendo da MAE 72
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Tuple

# Imports locali
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """Smooth L1 per count - più robusto di L1 per outlier."""
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss (DM-Count style).
    
    Confronta la distribuzione spaziale della density predetta con GT.
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(6 * sigma) | 1
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # L2 loss sulla distribuzione normalizzata
        loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        return loss.mean()


class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con Sinkhorn.
    
    Calcola distanza Wasserstein tra pred e GT.
    """
    
    def __init__(self, reg: float = 0.1, num_iters: int = 50, downsample: int = 8):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
        self._cost_cache = {}
    
    def _get_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cost_cache:
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            self._cost_cache[key] = (diff ** 2).sum(dim=-1)
        return self._cost_cache[key]
    
    def _sinkhorn(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        
        for _ in range(self.num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        P = u.view(-1, 1) * K * v.view(1, -1)
        return (P * C).sum()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        B, _, H, W = pred.shape
        C = self._get_cost_matrix(H, W, pred.device)
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(B):
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a / a_sum, b / b_sum, C)
                total_loss = total_loss + cost
                valid_count += 1
        
        return total_loss / max(valid_count, 1)


class CombinedP2RLoss(nn.Module):
    """
    Loss combinata per Stage 2.
    
    Componenti:
    1. Count Loss (Smooth L1) - peso principale
    2. Distribution Matching - per localizzazione
    3. Optimal Transport (opzionale) - per matching preciso
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        dm_weight: float = 0.5,
        ot_weight: float = 0.0,  # Disabilitato di default (lento)
        dm_sigma: float = 8.0,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        
        self.count_loss = SmoothL1CountLoss(beta=10.0)
        
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma, downsample=4)
        
        if ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(reg=0.1, num_iters=30, downsample=8)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        
        B = pred_density.shape[0]
        device = pred_density.device
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        loss_count = self.count_loss(pred_t, gt_t)
        total_loss = self.count_weight * loss_count
        
        metrics = {
            'count': loss_count.item(),
        }
        
        # 2. Distribution Matching
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_dm = self.dm_loss(pred_density, gt_resized)
            total_loss = total_loss + self.dm_weight * loss_dm
            metrics['dm'] = loss_dm.item()
        
        # 3. Optimal Transport (se abilitato)
        if self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_ot = self.ot_loss(pred_density, gt_resized)
            total_loss = total_loss + self.ot_weight * loss_ot
            metrics['ot'] = loss_ot.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_t - gt_t).mean().item()
            bias = pred_t.sum().item() / gt_t.sum().clamp(min=1).item()
        
        metrics['total'] = total_loss.item()
        metrics['mae'] = mae
        metrics['bias'] = bias
        
        return total_loss, metrics


# =============================================================================
# SCHEDULER
# =============================================================================

class CosineRestartScheduler:
    """Cosine annealing con warmup e restart."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup=100):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.epoch <= self.warmup:
                lr = base_lr * self.epoch / self.warmup
            else:
                t = self.epoch - self.warmup
                T_cur = self.T_0
                while t >= T_cur:
                    t -= T_cur
                    T_cur = int(T_cur * self.T_mult)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T_cur)) / 2
            pg['lr'] = lr
    
    def state_dict(self):
        return {'epoch': self.epoch}
    
    def load_state_dict(self, d):
        self.epoch = d['epoch']


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        loss, metrics = criterion(pred, gt_density, points_list, cell_area)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_m.update(loss.item())
        mae_m.update(metrics['mae'])
        
        pbar.set_postfix({
            'L': f"{loss_m.avg:.3f}",
            'MAE': f"{mae_m.avg:.1f}",
            'DM': f"{metrics.get('dm', 0):.3f}",
        })
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae, all_mse = [], []
    total_pred, total_gt = 0.0, 0.0
    
    for images, _, points in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(p - gt))
            all_mse.append((p - gt) ** 2)
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


@torch.no_grad()
def calibrate(model, loader, device, down, batches=15):
    model.eval()
    
    preds, gts = [], []
    
    for i, (images, _, points) in enumerate(loader):
        if i >= batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        outputs = model(images)
        pred = outputs['p2r_density']
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for j, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                preds.append((pred[j].sum() / cell_area).item())
                gts.append(gt)
    
    if not gts:
        return
    
    bias = sum(preds) / sum(gts)
    
    if abs(bias - 1.0) > 0.05:
        adj = np.clip(np.log(bias), -1.0, 1.0)
        old = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adj, device=device)
        new = model.p2r_head.log_scale.item()
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old:.3f}→{new:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dm-weight', type=float, default=0.5)
    parser.add_argument('--ot-weight', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--load-from', type=str, default=None, 
                        help='Directory da cui caricare Stage 1 (es: exp/shha_v10)')
    parser.add_argument('--continue-stage2', type=str, default=None,
                        help='Checkpoint Stage 2 da cui continuare (es: exp/shha_v10/stage2_best.pth)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    
    epochs = args.epochs
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 500)
    warmup = p2r_cfg.get('WARMUP_EPOCHS', 100)
    restart_T = 1000
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_bb = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v13')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V13 - Advanced Losses (Architettura Originale)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"DM Weight: {args.dm_weight}")
    print(f"OT Weight: {args.ot_weight}")
    print(f"LR: {lr}, LR_backbone: {lr_bb}")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DS = get_dataset(cfg['DATASET'])
    train_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_tf
    )
    val_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True, num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_cfg = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=cfg['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_cfg['bins'],
        bin_centers=bin_cfg['bin_centers'],
        use_ste_mask=cfg['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load Stage 1 o continua da Stage 2
    load_dir = args.load_from if args.load_from else out_dir
    
    if args.continue_stage2:
        # Continua da un checkpoint Stage 2 esistente
        print(f"✅ Continuando da Stage 2: {args.continue_stage2}")
        state = torch.load(args.continue_stage2, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        # Carica Stage 1
        s1_path = os.path.join(load_dir, 'best_model.pth')
        if os.path.isfile(s1_path):
            state = torch.load(s1_path, map_location=device)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded Stage 1: {s1_path}")
        else:
            print(f"⚠️ Stage 1 non trovato in {s1_path}, training da zero")
    
    # Freeze ZIP head
    for p in model.zip_head.parameters():
        p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    groups = []
    if lr_bb > 0:
        groups.append({'params': model.backbone.parameters(), 'lr': lr_bb})
        print(f"   Backbone: LR={lr_bb}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        groups.append({'params': p2r_params, 'lr': lr})
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    scheduler = CosineRestartScheduler(optimizer, T_0=restart_T, warmup=warmup)
    
    # Loss con DM
    criterion = CombinedP2RLoss(
        count_weight=3.0,
        dm_weight=args.dm_weight,
        ot_weight=args.ot_weight,
        dm_sigma=8.0,
    )
    print(f"   Loss: Count(3.0) + DM({args.dm_weight}) + OT({args.ot_weight})")
    
    # Resume
    start = 1
    best_mae = float('inf')
    no_improve = 0
    
    ckpt = os.path.join(out_dir, 'stage2_last.pth')
    if args.resume and os.path.isfile(ckpt):
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        start = c['epoch'] + 1
        best_mae = c.get('best_mae', float('inf'))
        no_improve = c.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Initial validation:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    if val['mae'] < best_mae:
        best_mae = val['mae']
    
    # Training
    print(f"\n🚀 Training: {start} → {epochs}")
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, down, epoch, grad_clip)
        scheduler.step()
        
        if epoch % val_interval == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_best.pth'))
                
                if mae < 60:
                    print(f"   🎯 TARGET RAGGIUNTO! MAE={mae:.2f}")
            else:
                no_improve += val_interval
            
            if epoch % 100 == 0:
                calibrate(model, val_loader, device, down)
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(out_dir, 'stage2_last.pth'))
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 65:
        print("   ✅ Vicino al target")
    
    print("=" * 60)


if __name__ == '__main__':
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V13 - P2R Training con Advanced Losses

STRATEGIA:
- Architettura P2RHead ORIGINALE (che funziona, MAE ~72)
- Advanced Losses: Distribution Matching + Optimal Transport
- Training più lungo con cosine restart
- Nessuna modifica all'architettura

TARGET: MAE < 60 partendo da MAE 72
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Tuple

# Imports locali
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """Smooth L1 per count - più robusto di L1 per outlier."""
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss (DM-Count style).
    
    Confronta la distribuzione spaziale della density predetta con GT.
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(6 * sigma) | 1
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # L2 loss sulla distribuzione normalizzata
        loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        return loss.mean()


class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con Sinkhorn.
    
    Calcola distanza Wasserstein tra pred e GT.
    """
    
    def __init__(self, reg: float = 0.1, num_iters: int = 50, downsample: int = 8):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
        self._cost_cache = {}
    
    def _get_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cost_cache:
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            self._cost_cache[key] = (diff ** 2).sum(dim=-1)
        return self._cost_cache[key]
    
    def _sinkhorn(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        
        for _ in range(self.num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        P = u.view(-1, 1) * K * v.view(1, -1)
        return (P * C).sum()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        B, _, H, W = pred.shape
        C = self._get_cost_matrix(H, W, pred.device)
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(B):
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a / a_sum, b / b_sum, C)
                total_loss = total_loss + cost
                valid_count += 1
        
        return total_loss / max(valid_count, 1)


class CombinedP2RLoss(nn.Module):
    """
    Loss combinata per Stage 2.
    
    Componenti:
    1. Count Loss (Smooth L1) - peso principale
    2. Distribution Matching - per localizzazione
    3. Optimal Transport (opzionale) - per matching preciso
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        dm_weight: float = 0.5,
        ot_weight: float = 0.0,  # Disabilitato di default (lento)
        dm_sigma: float = 8.0,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        
        self.count_loss = SmoothL1CountLoss(beta=10.0)
        
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma, downsample=4)
        
        if ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(reg=0.1, num_iters=30, downsample=8)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        
        B = pred_density.shape[0]
        device = pred_density.device
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        loss_count = self.count_loss(pred_t, gt_t)
        total_loss = self.count_weight * loss_count
        
        metrics = {
            'count': loss_count.item(),
        }
        
        # 2. Distribution Matching
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_dm = self.dm_loss(pred_density, gt_resized)
            total_loss = total_loss + self.dm_weight * loss_dm
            metrics['dm'] = loss_dm.item()
        
        # 3. Optimal Transport (se abilitato)
        if self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_ot = self.ot_loss(pred_density, gt_resized)
            total_loss = total_loss + self.ot_weight * loss_ot
            metrics['ot'] = loss_ot.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_t - gt_t).mean().item()
            bias = pred_t.sum().item() / gt_t.sum().clamp(min=1).item()
        
        metrics['total'] = total_loss.item()
        metrics['mae'] = mae
        metrics['bias'] = bias
        
        return total_loss, metrics


# =============================================================================
# SCHEDULER
# =============================================================================

class CosineRestartScheduler:
    """Cosine annealing con warmup e restart."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup=100):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.epoch <= self.warmup:
                lr = base_lr * self.epoch / self.warmup
            else:
                t = self.epoch - self.warmup
                T_cur = self.T_0
                while t >= T_cur:
                    t -= T_cur
                    T_cur = int(T_cur * self.T_mult)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T_cur)) / 2
            pg['lr'] = lr
    
    def state_dict(self):
        return {'epoch': self.epoch}
    
    def load_state_dict(self, d):
        self.epoch = d['epoch']


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        loss, metrics = criterion(pred, gt_density, points_list, cell_area)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_m.update(loss.item())
        mae_m.update(metrics['mae'])
        
        pbar.set_postfix({
            'L': f"{loss_m.avg:.3f}",
            'MAE': f"{mae_m.avg:.1f}",
            'DM': f"{metrics.get('dm', 0):.3f}",
        })
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae, all_mse = [], []
    total_pred, total_gt = 0.0, 0.0
    
    for images, _, points in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(p - gt))
            all_mse.append((p - gt) ** 2)
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


@torch.no_grad()
def calibrate(model, loader, device, down, batches=15):
    model.eval()
    
    preds, gts = [], []
    
    for i, (images, _, points) in enumerate(loader):
        if i >= batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        outputs = model(images)
        pred = outputs['p2r_density']
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for j, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                preds.append((pred[j].sum() / cell_area).item())
                gts.append(gt)
    
    if not gts:
        return
    
    bias = sum(preds) / sum(gts)
    
    if abs(bias - 1.0) > 0.05:
        adj = np.clip(np.log(bias), -1.0, 1.0)
        old = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adj, device=device)
        new = model.p2r_head.log_scale.item()
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old:.3f}→{new:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dm-weight', type=float, default=0.5)
    parser.add_argument('--ot-weight', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--load-from', type=str, default=None, 
                        help='Directory da cui caricare Stage 1 (es: exp/shha_v10)')
    parser.add_argument('--continue-stage2', type=str, default=None,
                        help='Checkpoint Stage 2 da cui continuare (es: exp/shha_v10/stage2_best.pth)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    
    epochs = args.epochs
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 500)
    warmup = p2r_cfg.get('WARMUP_EPOCHS', 100)
    restart_T = 1000
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_bb = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v13')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V13 - Advanced Losses (Architettura Originale)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"DM Weight: {args.dm_weight}")
    print(f"OT Weight: {args.ot_weight}")
    print(f"LR: {lr}, LR_backbone: {lr_bb}")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DS = get_dataset(cfg['DATASET'])
    train_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_tf
    )
    val_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True, num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_cfg = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=cfg['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_cfg['bins'],
        bin_centers=bin_cfg['bin_centers'],
        use_ste_mask=cfg['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load Stage 1 o continua da Stage 2
    load_dir = args.load_from if args.load_from else out_dir
    
    if args.continue_stage2:
        # Continua da un checkpoint Stage 2 esistente
        print(f"✅ Continuando da Stage 2: {args.continue_stage2}")
        state = torch.load(args.continue_stage2, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        # Carica Stage 1
        s1_path = os.path.join(load_dir, 'best_model.pth')
        if os.path.isfile(s1_path):
            state = torch.load(s1_path, map_location=device)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded Stage 1: {s1_path}")
        else:
            print(f"⚠️ Stage 1 non trovato in {s1_path}, training da zero")
    
    # Freeze ZIP head
    for p in model.zip_head.parameters():
        p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    groups = []
    if lr_bb > 0:
        groups.append({'params': model.backbone.parameters(), 'lr': lr_bb})
        print(f"   Backbone: LR={lr_bb}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        groups.append({'params': p2r_params, 'lr': lr})
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    scheduler = CosineRestartScheduler(optimizer, T_0=restart_T, warmup=warmup)
    
    # Loss con DM
    criterion = CombinedP2RLoss(
        count_weight=3.0,
        dm_weight=args.dm_weight,
        ot_weight=args.ot_weight,
        dm_sigma=8.0,
    )
    print(f"   Loss: Count(3.0) + DM({args.dm_weight}) + OT({args.ot_weight})")
    
    # Resume
    start = 1
    best_mae = float('inf')
    no_improve = 0
    
    ckpt = os.path.join(out_dir, 'stage2_last.pth')
    if args.resume and os.path.isfile(ckpt):
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        start = c['epoch'] + 1
        best_mae = c.get('best_mae', float('inf'))
        no_improve = c.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Initial validation:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    if val['mae'] < best_mae:
        best_mae = val['mae']
    
    # Training
    print(f"\n🚀 Training: {start} → {epochs}")
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, down, epoch, grad_clip)
        scheduler.step()
        
        if epoch % val_interval == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_best.pth'))
                
                if mae < 60:
                    print(f"   🎯 TARGET RAGGIUNTO! MAE={mae:.2f}")
            else:
                no_improve += val_interval
            
            if epoch % 100 == 0:
                calibrate(model, val_loader, device, down)
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(out_dir, 'stage2_last.pth'))
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 65:
        print("   ✅ Vicino al target")
    
    print("=" * 60)


if __name__ == '__main__':
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V13 - P2R Training con Advanced Losses

STRATEGIA:
- Architettura P2RHead ORIGINALE (che funziona, MAE ~72)
- Advanced Losses: Distribution Matching + Optimal Transport
- Training più lungo con cosine restart
- Nessuna modifica all'architettura

TARGET: MAE < 60 partendo da MAE 72
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Tuple

# Imports locali
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """Smooth L1 per count - più robusto di L1 per outlier."""
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss (DM-Count style).
    
    Confronta la distribuzione spaziale della density predetta con GT.
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(6 * sigma) | 1
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # L2 loss sulla distribuzione normalizzata
        loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        return loss.mean()


class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con Sinkhorn.
    
    Calcola distanza Wasserstein tra pred e GT.
    """
    
    def __init__(self, reg: float = 0.1, num_iters: int = 50, downsample: int = 8):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
        self._cost_cache = {}
    
    def _get_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cost_cache:
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            self._cost_cache[key] = (diff ** 2).sum(dim=-1)
        return self._cost_cache[key]
    
    def _sinkhorn(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        
        for _ in range(self.num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        P = u.view(-1, 1) * K * v.view(1, -1)
        return (P * C).sum()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        B, _, H, W = pred.shape
        C = self._get_cost_matrix(H, W, pred.device)
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(B):
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a / a_sum, b / b_sum, C)
                total_loss = total_loss + cost
                valid_count += 1
        
        return total_loss / max(valid_count, 1)


class CombinedP2RLoss(nn.Module):
    """
    Loss combinata per Stage 2.
    
    Componenti:
    1. Count Loss (Smooth L1) - peso principale
    2. Distribution Matching - per localizzazione
    3. Optimal Transport (opzionale) - per matching preciso
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        dm_weight: float = 0.5,
        ot_weight: float = 0.0,  # Disabilitato di default (lento)
        dm_sigma: float = 8.0,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        
        self.count_loss = SmoothL1CountLoss(beta=10.0)
        
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma, downsample=4)
        
        if ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(reg=0.1, num_iters=30, downsample=8)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        
        B = pred_density.shape[0]
        device = pred_density.device
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        loss_count = self.count_loss(pred_t, gt_t)
        total_loss = self.count_weight * loss_count
        
        metrics = {
            'count': loss_count.item(),
        }
        
        # 2. Distribution Matching
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_dm = self.dm_loss(pred_density, gt_resized)
            total_loss = total_loss + self.dm_weight * loss_dm
            metrics['dm'] = loss_dm.item()
        
        # 3. Optimal Transport (se abilitato)
        if self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_ot = self.ot_loss(pred_density, gt_resized)
            total_loss = total_loss + self.ot_weight * loss_ot
            metrics['ot'] = loss_ot.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_t - gt_t).mean().item()
            bias = pred_t.sum().item() / gt_t.sum().clamp(min=1).item()
        
        metrics['total'] = total_loss.item()
        metrics['mae'] = mae
        metrics['bias'] = bias
        
        return total_loss, metrics


# =============================================================================
# SCHEDULER
# =============================================================================

class CosineRestartScheduler:
    """Cosine annealing con warmup e restart."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup=100):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.epoch <= self.warmup:
                lr = base_lr * self.epoch / self.warmup
            else:
                t = self.epoch - self.warmup
                T_cur = self.T_0
                while t >= T_cur:
                    t -= T_cur
                    T_cur = int(T_cur * self.T_mult)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T_cur)) / 2
            pg['lr'] = lr
    
    def state_dict(self):
        return {'epoch': self.epoch}
    
    def load_state_dict(self, d):
        self.epoch = d['epoch']


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        loss, metrics = criterion(pred, gt_density, points_list, cell_area)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_m.update(loss.item())
        mae_m.update(metrics['mae'])
        
        pbar.set_postfix({
            'L': f"{loss_m.avg:.3f}",
            'MAE': f"{mae_m.avg:.1f}",
            'DM': f"{metrics.get('dm', 0):.3f}",
        })
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae, all_mse = [], []
    total_pred, total_gt = 0.0, 0.0
    
    for images, _, points in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(p - gt))
            all_mse.append((p - gt) ** 2)
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


@torch.no_grad()
def calibrate(model, loader, device, down, batches=15):
    model.eval()
    
    preds, gts = [], []
    
    for i, (images, _, points) in enumerate(loader):
        if i >= batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        outputs = model(images)
        pred = outputs['p2r_density']
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for j, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                preds.append((pred[j].sum() / cell_area).item())
                gts.append(gt)
    
    if not gts:
        return
    
    bias = sum(preds) / sum(gts)
    
    if abs(bias - 1.0) > 0.05:
        adj = np.clip(np.log(bias), -1.0, 1.0)
        old = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adj, device=device)
        new = model.p2r_head.log_scale.item()
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old:.3f}→{new:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dm-weight', type=float, default=0.5)
    parser.add_argument('--ot-weight', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--load-from', type=str, default=None, 
                        help='Directory da cui caricare Stage 1 (es: exp/shha_v10)')
    parser.add_argument('--continue-stage2', type=str, default=None,
                        help='Checkpoint Stage 2 da cui continuare (es: exp/shha_v10/stage2_best.pth)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    
    epochs = args.epochs
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 500)
    warmup = p2r_cfg.get('WARMUP_EPOCHS', 100)
    restart_T = 1000
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_bb = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v13')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V13 - Advanced Losses (Architettura Originale)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"DM Weight: {args.dm_weight}")
    print(f"OT Weight: {args.ot_weight}")
    print(f"LR: {lr}, LR_backbone: {lr_bb}")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DS = get_dataset(cfg['DATASET'])
    train_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_tf
    )
    val_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True, num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_cfg = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=cfg['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_cfg['bins'],
        bin_centers=bin_cfg['bin_centers'],
        use_ste_mask=cfg['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load Stage 1 o continua da Stage 2
    load_dir = args.load_from if args.load_from else out_dir
    
    if args.continue_stage2:
        # Continua da un checkpoint Stage 2 esistente
        print(f"✅ Continuando da Stage 2: {args.continue_stage2}")
        state = torch.load(args.continue_stage2, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        # Carica Stage 1
        s1_path = os.path.join(load_dir, 'best_model.pth')
        if os.path.isfile(s1_path):
            state = torch.load(s1_path, map_location=device)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded Stage 1: {s1_path}")
        else:
            print(f"⚠️ Stage 1 non trovato in {s1_path}, training da zero")
    
    # Freeze ZIP head
    for p in model.zip_head.parameters():
        p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    groups = []
    if lr_bb > 0:
        groups.append({'params': model.backbone.parameters(), 'lr': lr_bb})
        print(f"   Backbone: LR={lr_bb}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        groups.append({'params': p2r_params, 'lr': lr})
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    scheduler = CosineRestartScheduler(optimizer, T_0=restart_T, warmup=warmup)
    
    # Loss con DM
    criterion = CombinedP2RLoss(
        count_weight=3.0,
        dm_weight=args.dm_weight,
        ot_weight=args.ot_weight,
        dm_sigma=8.0,
    )
    print(f"   Loss: Count(3.0) + DM({args.dm_weight}) + OT({args.ot_weight})")
    
    # Resume
    start = 1
    best_mae = float('inf')
    no_improve = 0
    
    ckpt = os.path.join(out_dir, 'stage2_last.pth')
    if args.resume and os.path.isfile(ckpt):
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        start = c['epoch'] + 1
        best_mae = c.get('best_mae', float('inf'))
        no_improve = c.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Initial validation:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    if val['mae'] < best_mae:
        best_mae = val['mae']
    
    # Training
    print(f"\n🚀 Training: {start} → {epochs}")
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, down, epoch, grad_clip)
        scheduler.step()
        
        if epoch % val_interval == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_best.pth'))
                
                if mae < 60:
                    print(f"   🎯 TARGET RAGGIUNTO! MAE={mae:.2f}")
            else:
                no_improve += val_interval
            
            if epoch % 100 == 0:
                calibrate(model, val_loader, device, down)
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(out_dir, 'stage2_last.pth'))
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 65:
        print("   ✅ Vicino al target")
    
    print("=" * 60)


if __name__ == '__main__':
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V13 - P2R Training con Advanced Losses

STRATEGIA:
- Architettura P2RHead ORIGINALE (che funziona, MAE ~72)
- Advanced Losses: Distribution Matching + Optimal Transport
- Training più lungo con cosine restart
- Nessuna modifica all'architettura

TARGET: MAE < 60 partendo da MAE 72
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Tuple

# Imports locali
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """Smooth L1 per count - più robusto di L1 per outlier."""
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss (DM-Count style).
    
    Confronta la distribuzione spaziale della density predetta con GT.
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(6 * sigma) | 1
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # L2 loss sulla distribuzione normalizzata
        loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        return loss.mean()


class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con Sinkhorn.
    
    Calcola distanza Wasserstein tra pred e GT.
    """
    
    def __init__(self, reg: float = 0.1, num_iters: int = 50, downsample: int = 8):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
        self._cost_cache = {}
    
    def _get_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cost_cache:
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            self._cost_cache[key] = (diff ** 2).sum(dim=-1)
        return self._cost_cache[key]
    
    def _sinkhorn(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        
        for _ in range(self.num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        P = u.view(-1, 1) * K * v.view(1, -1)
        return (P * C).sum()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        B, _, H, W = pred.shape
        C = self._get_cost_matrix(H, W, pred.device)
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(B):
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a / a_sum, b / b_sum, C)
                total_loss = total_loss + cost
                valid_count += 1
        
        return total_loss / max(valid_count, 1)


class CombinedP2RLoss(nn.Module):
    """
    Loss combinata per Stage 2.
    
    Componenti:
    1. Count Loss (Smooth L1) - peso principale
    2. Distribution Matching - per localizzazione
    3. Optimal Transport (opzionale) - per matching preciso
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        dm_weight: float = 0.5,
        ot_weight: float = 0.0,  # Disabilitato di default (lento)
        dm_sigma: float = 8.0,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        
        self.count_loss = SmoothL1CountLoss(beta=10.0)
        
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma, downsample=4)
        
        if ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(reg=0.1, num_iters=30, downsample=8)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        
        B = pred_density.shape[0]
        device = pred_density.device
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        loss_count = self.count_loss(pred_t, gt_t)
        total_loss = self.count_weight * loss_count
        
        metrics = {
            'count': loss_count.item(),
        }
        
        # 2. Distribution Matching
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_dm = self.dm_loss(pred_density, gt_resized)
            total_loss = total_loss + self.dm_weight * loss_dm
            metrics['dm'] = loss_dm.item()
        
        # 3. Optimal Transport (se abilitato)
        if self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_ot = self.ot_loss(pred_density, gt_resized)
            total_loss = total_loss + self.ot_weight * loss_ot
            metrics['ot'] = loss_ot.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_t - gt_t).mean().item()
            bias = pred_t.sum().item() / gt_t.sum().clamp(min=1).item()
        
        metrics['total'] = total_loss.item()
        metrics['mae'] = mae
        metrics['bias'] = bias
        
        return total_loss, metrics


# =============================================================================
# SCHEDULER
# =============================================================================

class CosineRestartScheduler:
    """Cosine annealing con warmup e restart."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup=100):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.epoch <= self.warmup:
                lr = base_lr * self.epoch / self.warmup
            else:
                t = self.epoch - self.warmup
                T_cur = self.T_0
                while t >= T_cur:
                    t -= T_cur
                    T_cur = int(T_cur * self.T_mult)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T_cur)) / 2
            pg['lr'] = lr
    
    def state_dict(self):
        return {'epoch': self.epoch}
    
    def load_state_dict(self, d):
        self.epoch = d['epoch']


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        loss, metrics = criterion(pred, gt_density, points_list, cell_area)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_m.update(loss.item())
        mae_m.update(metrics['mae'])
        
        pbar.set_postfix({
            'L': f"{loss_m.avg:.3f}",
            'MAE': f"{mae_m.avg:.1f}",
            'DM': f"{metrics.get('dm', 0):.3f}",
        })
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae, all_mse = [], []
    total_pred, total_gt = 0.0, 0.0
    
    for images, _, points in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(p - gt))
            all_mse.append((p - gt) ** 2)
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


@torch.no_grad()
def calibrate(model, loader, device, down, batches=15):
    model.eval()
    
    preds, gts = [], []
    
    for i, (images, _, points) in enumerate(loader):
        if i >= batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        outputs = model(images)
        pred = outputs['p2r_density']
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for j, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                preds.append((pred[j].sum() / cell_area).item())
                gts.append(gt)
    
    if not gts:
        return
    
    bias = sum(preds) / sum(gts)
    
    if abs(bias - 1.0) > 0.05:
        adj = np.clip(np.log(bias), -1.0, 1.0)
        old = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adj, device=device)
        new = model.p2r_head.log_scale.item()
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old:.3f}→{new:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dm-weight', type=float, default=0.5)
    parser.add_argument('--ot-weight', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--load-from', type=str, default=None, 
                        help='Directory da cui caricare Stage 1 (es: exp/shha_v10)')
    parser.add_argument('--continue-stage2', type=str, default=None,
                        help='Checkpoint Stage 2 da cui continuare (es: exp/shha_v10/stage2_best.pth)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    
    epochs = args.epochs
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 500)
    warmup = p2r_cfg.get('WARMUP_EPOCHS', 100)
    restart_T = 1000
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_bb = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v13')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V13 - Advanced Losses (Architettura Originale)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"DM Weight: {args.dm_weight}")
    print(f"OT Weight: {args.ot_weight}")
    print(f"LR: {lr}, LR_backbone: {lr_bb}")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DS = get_dataset(cfg['DATASET'])
    train_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_tf
    )
    val_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True, num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_cfg = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=cfg['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_cfg['bins'],
        bin_centers=bin_cfg['bin_centers'],
        use_ste_mask=cfg['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load Stage 1 o continua da Stage 2
    load_dir = args.load_from if args.load_from else out_dir
    
    if args.continue_stage2:
        # Continua da un checkpoint Stage 2 esistente
        print(f"✅ Continuando da Stage 2: {args.continue_stage2}")
        state = torch.load(args.continue_stage2, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        # Carica Stage 1
        s1_path = os.path.join(load_dir, 'best_model.pth')
        if os.path.isfile(s1_path):
            state = torch.load(s1_path, map_location=device)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded Stage 1: {s1_path}")
        else:
            print(f"⚠️ Stage 1 non trovato in {s1_path}, training da zero")
    
    # Freeze ZIP head
    for p in model.zip_head.parameters():
        p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    groups = []
    if lr_bb > 0:
        groups.append({'params': model.backbone.parameters(), 'lr': lr_bb})
        print(f"   Backbone: LR={lr_bb}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        groups.append({'params': p2r_params, 'lr': lr})
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    scheduler = CosineRestartScheduler(optimizer, T_0=restart_T, warmup=warmup)
    
    # Loss con DM
    criterion = CombinedP2RLoss(
        count_weight=3.0,
        dm_weight=args.dm_weight,
        ot_weight=args.ot_weight,
        dm_sigma=8.0,
    )
    print(f"   Loss: Count(3.0) + DM({args.dm_weight}) + OT({args.ot_weight})")
    
    # Resume
    start = 1
    best_mae = float('inf')
    no_improve = 0
    
    ckpt = os.path.join(out_dir, 'stage2_last.pth')
    if args.resume and os.path.isfile(ckpt):
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        start = c['epoch'] + 1
        best_mae = c.get('best_mae', float('inf'))
        no_improve = c.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Initial validation:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    if val['mae'] < best_mae:
        best_mae = val['mae']
    
    # Training
    print(f"\n🚀 Training: {start} → {epochs}")
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, down, epoch, grad_clip)
        scheduler.step()
        
        if epoch % val_interval == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_best.pth'))
                
                if mae < 60:
                    print(f"   🎯 TARGET RAGGIUNTO! MAE={mae:.2f}")
            else:
                no_improve += val_interval
            
            if epoch % 100 == 0:
                calibrate(model, val_loader, device, down)
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(out_dir, 'stage2_last.pth'))
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 65:
        print("   ✅ Vicino al target")
    
    print("=" * 60)


if __name__ == '__main__':
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V13 - P2R Training con Advanced Losses

STRATEGIA:
- Architettura P2RHead ORIGINALE (che funziona, MAE ~72)
- Advanced Losses: Distribution Matching + Optimal Transport
- Training più lungo con cosine restart
- Nessuna modifica all'architettura

TARGET: MAE < 60 partendo da MAE 72
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Tuple

# Imports locali
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SmoothL1CountLoss(nn.Module):
    """Smooth L1 per count - più robusto di L1 per outlier."""
    
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss (DM-Count style).
    
    Confronta la distribuzione spaziale della density predetta con GT.
    """
    
    def __init__(self, sigma: float = 8.0, downsample: int = 4):
        super().__init__()
        self.sigma = sigma
        self.downsample = downsample
        self.register_buffer('kernel', self._make_gaussian_kernel(sigma))
    
    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(6 * sigma) | 1
        x = torch.arange(size).float() - size // 2
        x = x.view(1, -1).expand(size, -1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x.device)
        padding = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        # Smooth
        pred_smooth = self._smooth(pred)
        gt_smooth = self._smooth(gt)
        
        # Normalize
        pred_sum = pred_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        gt_sum = gt_smooth.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        
        pred_norm = pred_smooth / pred_sum
        gt_norm = gt_smooth / gt_sum
        
        # L2 loss sulla distribuzione normalizzata
        loss = F.mse_loss(pred_norm, gt_norm, reduction='none').sum(dim=[2, 3])
        
        return loss.mean()


class OptimalTransportLoss(nn.Module):
    """
    Optimal Transport Loss con Sinkhorn.
    
    Calcola distanza Wasserstein tra pred e GT.
    """
    
    def __init__(self, reg: float = 0.1, num_iters: int = 50, downsample: int = 8):
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.downsample = downsample
        self._cost_cache = {}
    
    def _get_cost_matrix(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cost_cache:
            y = torch.arange(H, device=device).float()
            x = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            self._cost_cache[key] = (diff ** 2).sum(dim=-1)
        return self._cost_cache[key]
    
    def _sinkhorn(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        
        for _ in range(self.num_iters):
            v = b / (K.t() @ u + 1e-8)
            u = a / (K @ v + 1e-8)
        
        P = u.view(-1, 1) * K * v.view(1, -1)
        return (P * C).sum()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Downsample
        if self.downsample > 1:
            pred = F.avg_pool2d(pred, self.downsample) * (self.downsample ** 2)
            gt = F.avg_pool2d(gt, self.downsample) * (self.downsample ** 2)
        
        B, _, H, W = pred.shape
        C = self._get_cost_matrix(H, W, pred.device)
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(B):
            a = pred[i].flatten()
            b = gt[i].flatten()
            
            a_sum = a.sum().clamp(min=1e-8)
            b_sum = b.sum().clamp(min=1e-8)
            
            if a_sum > 0.5 and b_sum > 0.5:
                cost = self._sinkhorn(a / a_sum, b / b_sum, C)
                total_loss = total_loss + cost
                valid_count += 1
        
        return total_loss / max(valid_count, 1)


class CombinedP2RLoss(nn.Module):
    """
    Loss combinata per Stage 2.
    
    Componenti:
    1. Count Loss (Smooth L1) - peso principale
    2. Distribution Matching - per localizzazione
    3. Optimal Transport (opzionale) - per matching preciso
    """
    
    def __init__(
        self,
        count_weight: float = 3.0,
        dm_weight: float = 0.5,
        ot_weight: float = 0.0,  # Disabilitato di default (lento)
        dm_sigma: float = 8.0,
    ):
        super().__init__()
        
        self.count_weight = count_weight
        self.dm_weight = dm_weight
        self.ot_weight = ot_weight
        
        self.count_loss = SmoothL1CountLoss(beta=10.0)
        
        if dm_weight > 0:
            self.dm_loss = DistributionMatchingLoss(sigma=dm_sigma, downsample=4)
        
        if ot_weight > 0:
            self.ot_loss = OptimalTransportLoss(reg=0.1, num_iters=30, downsample=8)
    
    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        points_list: List[torch.Tensor],
        cell_area: float,
    ) -> Tuple[torch.Tensor, Dict]:
        
        B = pred_density.shape[0]
        device = pred_density.device
        
        # 1. Count Loss
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred = (pred_density[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(pred)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        loss_count = self.count_loss(pred_t, gt_t)
        total_loss = self.count_weight * loss_count
        
        metrics = {
            'count': loss_count.item(),
        }
        
        # 2. Distribution Matching
        if self.dm_weight > 0:
            # Resize GT se necessario
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_dm = self.dm_loss(pred_density, gt_resized)
            total_loss = total_loss + self.dm_weight * loss_dm
            metrics['dm'] = loss_dm.item()
        
        # 3. Optimal Transport (se abilitato)
        if self.ot_weight > 0:
            if gt_density.shape[-2:] != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density, size=pred_density.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                gt_resized = gt_density
            
            loss_ot = self.ot_loss(pred_density, gt_resized)
            total_loss = total_loss + self.ot_weight * loss_ot
            metrics['ot'] = loss_ot.item()
        
        # Metriche
        with torch.no_grad():
            mae = torch.abs(pred_t - gt_t).mean().item()
            bias = pred_t.sum().item() / gt_t.sum().clamp(min=1).item()
        
        metrics['total'] = total_loss.item()
        metrics['mae'] = mae
        metrics['bias'] = bias
        
        return total_loss, metrics


# =============================================================================
# SCHEDULER
# =============================================================================

class CosineRestartScheduler:
    """Cosine annealing con warmup e restart."""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, warmup=100):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.epoch <= self.warmup:
                lr = base_lr * self.epoch / self.warmup
            else:
                t = self.epoch - self.warmup
                T_cur = self.T_0
                while t >= T_cur:
                    t -= T_cur
                    T_cur = int(T_cur * self.T_mult)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T_cur)) / 2
            pg['lr'] = lr
    
    def state_dict(self):
        return {'epoch': self.epoch}
    
    def load_state_dict(self, d):
        self.epoch = d['epoch']


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        loss, metrics = criterion(pred, gt_density, points_list, cell_area)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_m.update(loss.item())
        mae_m.update(metrics['mae'])
        
        pbar.set_postfix({
            'L': f"{loss_m.avg:.3f}",
            'MAE': f"{mae_m.avg:.1f}",
            'DM': f"{metrics.get('dm', 0):.3f}",
        })
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae, all_mse = [], []
    total_pred, total_gt = 0.0, 0.0
    
    for images, _, points in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(p - gt))
            all_mse.append((p - gt) ** 2)
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'rmse': np.sqrt(np.mean(all_mse)),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


@torch.no_grad()
def calibrate(model, loader, device, down, batches=15):
    model.eval()
    
    preds, gts = [], []
    
    for i, (images, _, points) in enumerate(loader):
        if i >= batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        outputs = model(images)
        pred = outputs['p2r_density']
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        for j, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                preds.append((pred[j].sum() / cell_area).item())
                gts.append(gt)
    
    if not gts:
        return
    
    bias = sum(preds) / sum(gts)
    
    if abs(bias - 1.0) > 0.05:
        adj = np.clip(np.log(bias), -1.0, 1.0)
        old = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adj, device=device)
        new = model.p2r_head.log_scale.item()
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old:.3f}→{new:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dm-weight', type=float, default=0.5)
    parser.add_argument('--ot-weight', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--load-from', type=str, default=None, 
                        help='Directory da cui caricare Stage 1 (es: exp/shha_v10)')
    parser.add_argument('--continue-stage2', type=str, default=None,
                        help='Checkpoint Stage 2 da cui continuare (es: exp/shha_v10/stage2_best.pth)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    
    epochs = args.epochs
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 500)
    warmup = p2r_cfg.get('WARMUP_EPOCHS', 100)
    restart_T = 1000
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_bb = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v13')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V13 - Advanced Losses (Architettura Originale)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"DM Weight: {args.dm_weight}")
    print(f"OT Weight: {args.ot_weight}")
    print(f"LR: {lr}, LR_backbone: {lr_bb}")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DS = get_dataset(cfg['DATASET'])
    train_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_tf
    )
    val_ds = DS(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True, num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_cfg = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=cfg['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_cfg['bins'],
        bin_centers=bin_cfg['bin_centers'],
        use_ste_mask=cfg['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load Stage 1 o continua da Stage 2
    load_dir = args.load_from if args.load_from else out_dir
    
    if args.continue_stage2:
        # Continua da un checkpoint Stage 2 esistente
        print(f"✅ Continuando da Stage 2: {args.continue_stage2}")
        state = torch.load(args.continue_stage2, map_location=device)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        # Carica Stage 1
        s1_path = os.path.join(load_dir, 'best_model.pth')
        if os.path.isfile(s1_path):
            state = torch.load(s1_path, map_location=device)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            print(f"✅ Loaded Stage 1: {s1_path}")
        else:
            print(f"⚠️ Stage 1 non trovato in {s1_path}, training da zero")
    
    # Freeze ZIP head
    for p in model.zip_head.parameters():
        p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    groups = []
    if lr_bb > 0:
        groups.append({'params': model.backbone.parameters(), 'lr': lr_bb})
        print(f"   Backbone: LR={lr_bb}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        groups.append({'params': p2r_params, 'lr': lr})
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    scheduler = CosineRestartScheduler(optimizer, T_0=restart_T, warmup=warmup)
    
    # Loss con DM
    criterion = CombinedP2RLoss(
        count_weight=3.0,
        dm_weight=args.dm_weight,
        ot_weight=args.ot_weight,
        dm_sigma=8.0,
    )
    print(f"   Loss: Count(3.0) + DM({args.dm_weight}) + OT({args.ot_weight})")
    
    # Resume
    start = 1
    best_mae = float('inf')
    no_improve = 0
    
    ckpt = os.path.join(out_dir, 'stage2_last.pth')
    if args.resume and os.path.isfile(ckpt):
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        start = c['epoch'] + 1
        best_mae = c.get('best_mae', float('inf'))
        no_improve = c.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Initial validation:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    if val['mae'] < best_mae:
        best_mae = val['mae']
    
    # Training
    print(f"\n🚀 Training: {start} → {epochs}")
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start, epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, down, epoch, grad_clip)
        scheduler.step()
        
        if epoch % val_interval == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            lr_now = optimizer.param_groups[-1]['lr']
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_best.pth'))
                
                if mae < 60:
                    print(f"   🎯 TARGET RAGGIUNTO! MAE={mae:.2f}")
            else:
                no_improve += val_interval
            
            if epoch % 100 == 0:
                calibrate(model, val_loader, device, down)
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(out_dir, 'stage2_last.pth'))
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 65:
        print("   ✅ Vicino al target")
    
    print("=" * 60)


if __name__ == '__main__':
    main()