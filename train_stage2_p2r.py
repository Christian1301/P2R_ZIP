#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V9 - P2R Training SENZA Maschera π

CAMBIAMENTO CHIAVE:
Il P2R head impara su TUTTE le features del backbone, NON solo quelle mascherate.
Questo rende il P2R head indipendente dalla qualità del π-head.

MOTIVAZIONE:
- Stage 1 produce un π-head con recall ~80% che perde persone
- Se Stage 2 usa la maschera, eredita questo problema
- Soluzione: P2R impara senza maschera, poi Stage 3 combina con soft weighting

TARGET: MAE < 70 usando raw density (senza masking)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds,
    collate_fn,
    canonicalize_p2r_grid,
)


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# P2R LOSS - Focus su Count Accuracy
# =============================================================================

class P2RLossV9(nn.Module):
    """
    P2R Loss semplificata per Stage 2 V9.
    
    Focus su:
    1. Count Loss (MAE) - peso alto, con density weighting opzionale
    2. Spatial Loss - peso basso
    3. Scale regularization
    
    NO masking - lavora su raw density.
    
    DENSITY WEIGHTING (V10):
    - weight = 1 + alpha * (gt_count / reference)^beta
    - Pesa di più gli errori su scene dense
    - alpha=0 disabilita (default per compatibilità)
    """
    
    def __init__(
        self,
        count_weight: float = 2.5,
        spatial_weight: float = 0.1,
        scale_weight: float = 0.3,
        # Density weighting (default: disabilitato)
        density_alpha: float = 0.0,
        density_beta: float = 0.5,
        reference_count: float = 300.0,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.density_alpha = density_alpha
        self.density_beta = density_beta
        self.reference_count = reference_count
    
    def _density_weight(self, gt_count):
        """Calcola peso per density weighting."""
        if self.density_alpha == 0:
            return 1.0
        w = 1.0 + self.density_alpha * (gt_count / self.reference_count) ** self.density_beta
        return min(w, 3.0)  # Cap a 3x
    
    def forward(self, pred, points_list, cell_area, H_in, W_in):
        """
        Args:
            pred: [B, 1, H, W] raw density (NO masking)
            points_list: lista di tensori [N_i, 2]
            cell_area: area cella per scaling
            H_in, W_in: dimensioni input
        """
        B, _, H, W = pred.shape
        device = pred.device
        
        total_count_loss = torch.tensor(0.0, device=device)
        total_spatial_loss = torch.tensor(0.0, device=device)
        
        gt_counts = []
        pred_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = pred[i].sum() / cell_area
            
            # Density weight
            dw = self._density_weight(gt)
            
            # Count Loss (L1) pesata
            total_count_loss += dw * torch.abs(pred_count - gt)
            
            gt_counts.append(gt)
            pred_counts.append(pred_count.item())
            
            # Spatial Loss
            if gt > 0 and self.spatial_weight > 0:
                target = torch.zeros(H, W, device=device)
                scale_h = H / H_in
                scale_w = W / W_in
                
                for pt in pts:
                    x = int((pt[0] * scale_w).clamp(0, W-1).item())
                    y = int((pt[1] * scale_h).clamp(0, H-1).item())
                    
                    # 3x3 gaussian
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                dist = (dx*dx + dy*dy) ** 0.5
                                target[ny, nx] += np.exp(-dist / 2)
                
                if target.sum() > 0:
                    target = target / target.sum()
                    pred_norm = pred[i, 0] / (pred[i, 0].sum() + 1e-8)
                    total_spatial_loss += F.mse_loss(pred_norm, target)
        
        # Averages
        avg_count = total_count_loss / B
        avg_spatial = total_spatial_loss / B
        
        # Scale loss (regularization)
        scale_loss = torch.tensor(0.0, device=device)
        
        total = (
            self.count_weight * avg_count +
            self.spatial_weight * avg_spatial +
            self.scale_weight * scale_loss
        )
        
        return {
            'total': total,
            'count': avg_count,
            'spatial': avg_spatial,
            'gt_counts': gt_counts,
            'pred_counts': pred_counts,
        }


# =============================================================================
# MODIFIED FORWARD - No Masking
# =============================================================================

def forward_without_mask(model, images):
    """
    Forward pass che bypassa il masking.
    
    Invece di usare:
      features → mask → gated_features → P2R head
    
    Usa direttamente:
      features → P2R head
    """
    # Backbone features
    feat = model.backbone(images)
    
    # ZIP head (per metriche, non per masking)
    zip_outputs = model.zip_head(feat, model.bin_centers)
    
    # P2R head direttamente sulle features (NO masking)
    density = model.p2r_head(feat)
    
    # Upsample se necessario
    if model.upsample_to_input:
        _, _, H, W = images.shape
        density = F.interpolate(density, size=(H, W), mode='bilinear', align_corners=False)
    
    # π probs per monitoring
    logit_pi = zip_outputs['logit_pi_maps']
    pi_softmax = logit_pi.softmax(dim=1)
    pi_probs = pi_softmax[:, 1:]
    
    return {
        'p2r_density': density,
        'logit_pi_maps': logit_pi,
        'lambda_maps': zip_outputs['lambda_maps'],
        'pi_probs': pi_probs,
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, default_down, use_mask=False):
    """
    Validazione.
    
    Args:
        use_mask: Se True, usa il forward standard con masking.
                  Se False (default), usa forward senza masking.
    """
    model.eval()
    
    all_mae = []
    all_mse = []
    total_pred = 0.0
    total_gt = 0.0
    
    for images, densities, points in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        if use_mask:
            outputs = model(images)
        else:
            outputs = forward_without_mask(model, images)
        
        pred = outputs['p2r_density']
        
        _, _, H_in, W_in = images.shape
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            pred_count = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(pred_count - gt))
            all_mse.append((pred_count - gt) ** 2)
            
            total_pred += pred_count
            total_gt += gt
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, default_down, epoch, grad_clip=1.0):
    """Training senza masking."""
    model.train()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Stage2 V9 [Ep {epoch}]")
    
    for images, densities, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H_in, W_in = images.shape
        
        # Forward SENZA maschera
        outputs = forward_without_mask(model, images)
        pred = outputs['p2r_density']
        
        # Canonicalize
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Loss
        losses = criterion(pred, points_list, cell_area, H_in, W_in)
        loss = losses['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Metriche
        mae = np.mean([abs(p - g) for p, g in zip(losses['pred_counts'], losses['gt_counts'])])
        
        loss_meter.update(loss.item())
        mae_meter.update(mae)
        
        pbar.set_postfix({
            'L': f"{loss_meter.avg:.3f}",
            'MAE': f"{mae_meter.avg:.1f}",
            'C': f"{losses['count'].item():.2f}",
        })
    
    return {'loss': loss_meter.avg, 'mae': mae_meter.avg}


# =============================================================================
# CALIBRATE LOG_SCALE
# =============================================================================

@torch.no_grad()
def calibrate_log_scale(model, loader, device, default_down, max_batches=15):
    """Calibra log_scale per correggere bias sistematico."""
    model.eval()
    
    pred_counts = []
    gt_counts = []
    
    for batch_idx, (images, _, points) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = forward_without_mask(model, images)
        pred = outputs['p2r_density']
        
        _, _, H_in, W_in = images.shape
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                pred_count = (pred[i].sum() / cell_area).item()
                pred_counts.append(pred_count)
                gt_counts.append(gt)
    
    if len(gt_counts) == 0:
        return
    
    bias = sum(pred_counts) / sum(gt_counts)
    
    if abs(bias - 1.0) > 0.05:
        adjust = np.log(bias)
        adjust = np.clip(adjust, -1.0, 1.0)
        
        old_scale = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adjust, device=device)
        new_scale = model.p2r_head.log_scale.item()
        
        print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {old_scale:.4f}→{new_scale:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    p2r_cfg = cfg['OPTIM_P2R']
    loss_cfg = cfg['P2R_LOSS']
    
    epochs = p2r_cfg.get('EPOCHS', 5000)
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 300)
    warmup_epochs = p2r_cfg.get('WARMUP_EPOCHS', 50)
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_backbone = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    # NO masking in training
    use_mask_in_training = p2r_cfg.get('USE_MASK_IN_TRAINING', False)
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip_v9')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V9 - P2R Training SENZA Maschera π")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Use Mask in Training: {use_mask_in_training}")
    print(f"Target: MAE < 70 (raw density)")
    print("=" * 60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(cfg["DATASET"])
    train_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf
    )
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=p2r_cfg.get('BATCH_SIZE', 8),
        shuffle=True,
        num_workers=p2r_cfg.get('NUM_WORKERS', 4),
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bin_config = cfg["BINS_CONFIG"][cfg["DATASET"]]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=False,
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        use_ste_mask=False,  # Non usato
        zip_head_kwargs={
            "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
            "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
            "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        },
    ).to(device)
    
    # Carica Stage 1
    stage1_path = os.path.join(output_dir, "best_model.pth")
    if os.path.isfile(stage1_path):
        state = torch.load(stage1_path, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"✅ Caricato Stage 1: {stage1_path}")
    else:
        print("⚠️ Stage 1 non trovato, training da zero")
    
    # Freeze ZIP head
    for param in model.zip_head.parameters():
        param.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Optimizer
    param_groups = []
    
    if lr_backbone > 0:
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': lr_backbone,
        })
        print(f"   Backbone: LR={lr_backbone}")
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("   Backbone: FROZEN")
    
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    if p2r_params:
        param_groups.append({
            'params': p2r_params,
            'lr': lr,
        })
        print(f"   P2R head: LR={lr}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss
    criterion = P2RLossV9(
        count_weight=loss_cfg.get('COUNT_WEIGHT', 2.5),
        spatial_weight=loss_cfg.get('SPATIAL_WEIGHT', 0.1),
        scale_weight=loss_cfg.get('SCALE_WEIGHT', 0.3),
        density_alpha=loss_cfg.get('DENSITY_ALPHA', 0.0),
        density_beta=loss_cfg.get('DENSITY_BETA', 0.5),
        reference_count=loss_cfg.get('REFERENCE_COUNT', 300.0),
    )
    
    if loss_cfg.get('DENSITY_ALPHA', 0.0) > 0:
        print(f"   Density Weighting: α={loss_cfg['DENSITY_ALPHA']}, β={loss_cfg.get('DENSITY_BETA', 0.5)}")
    
    # Resume
    start_epoch = 1
    best_mae = float('inf')
    no_improve = 0
    
    checkpoint_path = os.path.join(output_dir, "stage2_last.pth")
    if args.resume and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        no_improve = ckpt.get('no_improve', 0)
        print(f"✅ Resumed from epoch {start_epoch-1}, best MAE: {best_mae:.2f}")
    
    # Initial validation
    print("\n📋 Valutazione iniziale:")
    val_results = validate(model, val_loader, device, default_down, use_mask=False)
    print(f"   MAE: {val_results['mae']:.2f}, Bias: {val_results['bias']:.3f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
    
    # Training
    print(f"\n🚀 Training: {epochs} epochs (senza masking)")
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion,
            device, default_down, epoch, grad_clip
        )
        
        scheduler.step()
        
        # Validate
        if epoch % p2r_cfg.get('VAL_INTERVAL', 5) == 0 or epoch == 1:
            val_results = validate(model, val_loader, device, default_down, use_mask=False)
            
            mae = val_results['mae']
            improved = mae < best_mae
            
            print(f"Epoch {epoch:4d} | Train MAE: {train_results['mae']:.1f} | Val MAE: {mae:.2f} | Best: {best_mae:.2f} {'✅' if improved else ''}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val_results['bias'],
                }, os.path.join(output_dir, "stage2_best.pth"))
            else:
                no_improve += p2r_cfg.get('VAL_INTERVAL', 5)
            
            # Calibrazione periodica
            if epoch % 50 == 0:
                calibrate_log_scale(model, val_loader, device, default_down)
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        # Checkpoint periodico
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve': no_improve,
            }, os.path.join(output_dir, "stage2_last.pth"))
    
    # Risultati
    print("\n" + "=" * 60)
    print("🏁 STAGE 2 V9 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE (raw, no mask): {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_best.pth")
    
    if best_mae < 70:
        print("   🎯 TARGET RAGGIUNTO! MAE < 70")
    
    print("=" * 60)
    print("\n📌 Prossimo: python train_stage3_joint.py")


if __name__ == "__main__":
    main()