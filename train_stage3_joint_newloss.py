#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V3 - Joint Training con P2RLoss Originale

Aggiornato per usare la stessa P2RLoss dello Stage 2, garantendo coerenza.

OBIETTIVO:
Fine-tuning collaborativo ZIP + P2R per migliorare la sinergia tra i due componenti.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn
from losses.zip_nll import zip_nll
from losses.p2rloss_original import P2RLossOriginal


# =============================================================================
# UTILITIES
# =============================================================================

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


def save_checkpoint(state, filepath, description="checkpoint"):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(state, filepath)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"💾 {description} salvato: {filepath} ({size/1024/1024:.1f} MB)")
            sys.stdout.flush()
            return True
        return False
    except Exception as e:
        print(f"❌ ERRORE salvataggio: {e}")
        return False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, down_rate):
    """Validazione con conteggio pixel > 0."""
    model.eval()
    
    all_mae = []
    all_mse = []
    
    for images, gt_density, points in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device)
        
        outputs = model(images)
        pred_density = outputs['p2r_density']
        
        for i, pts in enumerate(points):
            gt_count = len(pts)
            pred_count = (pred_density[i] > 0).sum().item()
            
            all_mae.append(abs(pred_count - gt_count))
            all_mse.append((pred_count - gt_count) ** 2)
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    
    return {'mae': mae, 'rmse': rmse}


# =============================================================================
# TRAINING EPOCH
# =============================================================================

def train_one_epoch(
    model, 
    p2r_criterion,
    optimizer, 
    loader, 
    device, 
    epoch,
    down_rate,
    joint_cfg,
    grad_clip=5.0
):
    """Training con loss congiunta ZIP + P2R (originale)."""
    model.train()
    
    loss_meter = AverageMeter()
    loss_zip_meter = AverageMeter()
    loss_p2r_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    w_p2r = joint_cfg.get("FIXED_P2R_WEIGHT", 1.0)
    w_zip = joint_cfg.get("FIXED_ZIP_WEIGHT", 0.1)

    pbar = tqdm(loader, desc=f"Stage3 Joint [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]

        _, _, H_in, W_in = images.shape
        
        # Forward
        outputs = model(images)
        
        # =====================================================================
        # 1. ZIP LOSS
        # =====================================================================
        pi = torch.sigmoid(outputs['logit_pi_maps'][:, 1:2])
        lam = outputs['lambda_maps']
        
        block_h, block_w = pi.shape[-2:]
        factor_h = H_in / block_h
        factor_w = W_in / block_w
        
        gt_counts_block = F.adaptive_avg_pool2d(gt_density, (block_h, block_w))
        gt_counts_block = gt_counts_block * (factor_h * factor_w)
        
        loss_zip = zip_nll(pi, lam, gt_counts_block)
        
        # =====================================================================
        # 2. P2R LOSS (ORIGINALE)
        # =====================================================================
        pred_p2r = outputs['p2r_density']  # [B, 1, H, W]
        pred_p2r_squeezed = pred_p2r.squeeze(1)  # [B, H, W]
        
        loss_p2r = p2r_criterion(pred_p2r_squeezed, points_list, down_rate)
        
        # =====================================================================
        # 3. LOSS TOTALE
        # =====================================================================
        loss = (w_p2r * loss_p2r) + (w_zip * loss_zip)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Metriche
        with torch.no_grad():
            mae_batch = 0
            for i, pts in enumerate(points_list):
                gt = len(pts)
                pred = (pred_p2r[i] > 0).sum().item()
                mae_batch += abs(pred - gt)
            mae_batch /= len(points_list)
        
        loss_meter.update(loss.item())
        loss_zip_meter.update(loss_zip.item())
        loss_p2r_meter.update(loss_p2r.item())
        mae_meter.update(mae_batch)
        
        pbar.set_postfix({
            'L': f"{loss_meter.avg:.2f}",
            'ZIP': f"{loss_zip_meter.avg:.2f}",
            'P2R': f"{loss_p2r_meter.avg:.2f}",
            'MAE': f"{mae_meter.avg:.1f}",
        })
    
    return {
        'loss': loss_meter.avg,
        'loss_zip': loss_zip_meter.avg,
        'loss_p2r': loss_p2r_meter.avg,
        'mae': mae_meter.avg,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--stage2-ckpt', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config non trovato: {args.config}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get("DEVICE", "cuda"))
    init_seeds(cfg.get("SEED", 2025))
    
    # Parametri
    data_cfg = cfg['DATA']
    joint_cfg = cfg.get('OPTIM_JOINT', {})
    loss_cfg = cfg.get('JOINT_LOSS', {})
    p2r_loss_cfg = cfg.get('P2R_LOSS', {})
    
    epochs = joint_cfg.get('EPOCHS', 300)
    batch_size = joint_cfg.get('BATCH_SIZE', 4)
    lr_backbone = float(joint_cfg.get('LR_BACKBONE', 1e-5))
    lr_heads = float(joint_cfg.get('LR_HEADS', 5e-6))
    weight_decay = float(joint_cfg.get('WEIGHT_DECAY', 1e-4))
    val_interval = joint_cfg.get('VAL_INTERVAL', 5)
    patience = joint_cfg.get('EARLY_STOPPING_PATIENCE', 100)
    grad_clip = joint_cfg.get('GRAD_CLIP', 5.0)
    
    down_rate = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 3 V3 - Joint Training con P2RLoss Originale")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Epochs: {epochs}, Batch: {batch_size}")
    print(f"LR heads: {lr_heads}, LR backbone: {lr_backbone}")
    print(f"Grad clip: {grad_clip}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    sys.stdout.flush()
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(cfg["DATASET"])
    
    train_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        transforms=train_tf
    )
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Modello
    bin_config = cfg["BINS_CONFIG"][cfg["DATASET"]]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"].get("GATE", "multiply"),
        upsample_to_input=False,
        use_ste_mask=cfg["MODEL"].get("USE_STE_MASK", True),
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs={
            "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
            "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
            "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
            "lambda_noise_std": 0.0,
        },
    ).to(device)
    
    # Carica Stage 2
    stage2_loaded = False
    stage2_candidates = [args.stage2_ckpt] if args.stage2_ckpt else [
        os.path.join(output_dir, "stage2_best.pth"),
        os.path.join(output_dir, "stage2_last.pth"),
    ]
    
    for ckpt_path in stage2_candidates:
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(state.get("model", state), strict=False)
                print(f"✅ Caricato Stage 2 da: {ckpt_path}")
                stage2_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Errore: {e}")
    
    if not stage2_loaded:
        print("❌ Stage 2 checkpoint NON TROVATO!")
        return
    
    # Optimizer
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Sblocca ultimi layer backbone
    trainable_backbone_params = []
    if lr_backbone > 0:
        for i, layer in enumerate(model.backbone.body):
            if i >= 34:
                for param in layer.parameters():
                    param.requires_grad = True
                    trainable_backbone_params.append(param)
    
    param_groups = []
    if trainable_backbone_params:
        param_groups.append({'params': trainable_backbone_params, 'lr': lr_backbone})
    param_groups.append({'params': model.p2r_head.parameters(), 'lr': lr_heads})
    param_groups.append({'params': model.zip_head.parameters(), 'lr': lr_heads})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # P2RLoss originale
    p2r_criterion = P2RLossOriginal(
        factor=1,
        min_radius=p2r_loss_cfg.get('MIN_RADIUS', 8),
        max_radius=p2r_loss_cfg.get('MAX_RADIUS', 96),
        cost_class=p2r_loss_cfg.get('COST_CLASS', 1),
        cost_point=p2r_loss_cfg.get('COST_POINT', 8)
    ).to(device)
    
    # Resume
    start_epoch = 1
    best_mae = float('inf')
    no_improve_count = 0
    
    stage3_last_path = os.path.join(output_dir, "stage3_last.pth")
    if not args.no_resume and os.path.isfile(stage3_last_path):
        try:
            ckpt = torch.load(stage3_last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt.get('best_mae', float('inf'))
            no_improve_count = ckpt.get('no_improve_count', 0)
            print(f"✅ Resume da epoca {start_epoch}")
        except Exception as e:
            print(f"⚠️ Errore resume: {e}")
    
    # Validazione iniziale
    print("\n📋 Valutazione iniziale:")
    val_results = validate(model, val_loader, device, down_rate)
    print(f"   MAE: {val_results['mae']:.2f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
    
    # Training
    print(f"\n🚀 START Joint Training")
    
    for epoch in range(start_epoch, epochs + 1):
        train_results = train_one_epoch(
            model, p2r_criterion, optimizer, train_loader,
            device, epoch, down_rate, loss_cfg, grad_clip
        )
        
        scheduler.step()
        
        if epoch % val_interval == 0:
            val_results = validate(model, val_loader, device, down_rate)
            mae = val_results['mae']
            
            improved = mae < best_mae
            status = "✅ NEW BEST" if improved else ""
            
            print(f"Epoch {epoch:4d} | "
                  f"Train MAE: {train_results['mae']:.1f} | "
                  f"Val MAE: {mae:.2f} | "
                  f"Best: {best_mae:.2f} {status}")
            
            if improved:
                best_mae = mae
                no_improve_count = 0
                save_checkpoint(
                    {'epoch': epoch, 'model': model.state_dict(), 'mae': mae},
                    os.path.join(output_dir, "stage3_best.pth"),
                    f"Stage3 Best (MAE={mae:.2f})"
                )
            else:
                no_improve_count += val_interval
            
            if no_improve_count >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        if epoch % 50 == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_mae': best_mae,
                    'no_improve_count': no_improve_count,
                },
                os.path.join(output_dir, "stage3_last.pth"),
                f"Stage3 Last (Ep {epoch})"
            )
    
    print("\n" + "=" * 60)
    print("🏁 STAGE 3 COMPLETATO")
    print(f"   Best MAE: {best_mae:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()