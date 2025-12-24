#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Fine-tuning - Conservativo

Partendo da MAE 72, proviamo a scendere con:
1. Learning rate MOLTO basso (1e-6)
2. Solo count loss (Smooth L1)
3. Training lungo con early stopping paziente
4. Augmentation leggera

L'obiettivo è migliorare di qualche punto senza rompere il modello.
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


def smooth_l1_count_loss(pred_counts, gt_counts, beta=10.0):
    """Smooth L1 loss per count."""
    diff = torch.abs(pred_counts - gt_counts)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss.mean()


def train_epoch(model, loader, optimizer, device, down, epoch, grad_clip=1.0):
    model.train()
    
    loss_m = AverageMeter()
    mae_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Finetune [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H, W = images.shape
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H, W), down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        # Calcola counts
        pred_counts = []
        gt_counts = []
        for i, pts in enumerate(points_list):
            gt = len(pts)
            p = (pred[i].sum() / cell_area).clamp(min=0)
            gt_counts.append(gt)
            pred_counts.append(p)
        
        pred_t = torch.stack(pred_counts)
        gt_t = torch.tensor(gt_counts, device=device, dtype=torch.float)
        
        # Loss semplice
        loss = smooth_l1_count_loss(pred_t, gt_t)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        mae = torch.abs(pred_t - gt_t).mean().item()
        loss_m.update(loss.item())
        mae_m.update(mae)
        
        pbar.set_postfix({'L': f"{loss_m.avg:.2f}", 'MAE': f"{mae_m.avg:.1f}"})
    
    return {'loss': loss_m.avg, 'mae': mae_m.avg}


@torch.no_grad()
def validate(model, loader, device, down):
    model.eval()
    
    all_mae = []
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
            total_pred += p
            total_gt += gt
    
    return {
        'mae': np.mean(all_mae),
        'bias': total_pred / total_gt if total_gt > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_v10.yaml')
    parser.add_argument('--checkpoint', default='exp/shha_v10/stage2_best.pth')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (molto basso!)')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=300)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run = cfg.get('RUN_NAME', 'shha_v10')
    out_dir = os.path.join(cfg['EXP']['OUT_DIR'], run)
    
    print("=" * 60)
    print("🔧 Fine-tuning Conservativo")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
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
        train_ds, batch_size=8, shuffle=True, 
        num_workers=4, drop_last=True, collate_fn=collate_fn, pin_memory=True
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
    
    # Carica checkpoint
    print(f"\n✅ Caricamento: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
        print(f"   MAE precedente: {state.get('mae', 'N/A')}")
    else:
        model.load_state_dict(state)
    
    # Freeze tutto tranne P2R head
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.zip_head.parameters():
        p.requires_grad = False
    
    # Solo P2R head trainabile
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n and p.requires_grad]
    print(f"   Parametri trainabili: {sum(p.numel() for p in p2r_params):,}")
    
    optimizer = torch.optim.Adam(p2r_params, lr=args.lr, weight_decay=1e-5)
    
    # Scheduler: riduce LR se non migliora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-8
    )
    
    # Validazione iniziale
    print("\n📋 Validazione iniziale:")
    val = validate(model, val_loader, device, down)
    print(f"   MAE: {val['mae']:.2f}, Bias: {val['bias']:.3f}")
    
    best_mae = val['mae']
    no_improve = 0
    
    # Training
    print(f"\n🚀 Fine-tuning: 1 → {args.epochs}")
    
    for epoch in range(1, args.epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, device, down, epoch)
        
        if epoch % 5 == 0 or epoch == 1:
            val = validate(model, val_loader, device, down)
            
            mae = val['mae']
            improved = mae < best_mae
            
            scheduler.step(mae)
            lr_now = optimizer.param_groups[0]['lr']
            
            status = '✅ NEW BEST!' if improved else ''
            print(f"Ep {epoch:4d} | LR: {lr_now:.2e} | Train: {train_res['mae']:.1f} | Val: {mae:.2f} | Best: {best_mae:.2f} {status}")
            
            if improved:
                best_mae = mae
                no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'bias': val['bias'],
                }, os.path.join(out_dir, 'stage2_finetuned.pth'))
                
                if mae < 60:
                    print(f"\n🎯 TARGET RAGGIUNTO! MAE = {mae:.2f}")
            else:
                no_improve += 5
            
            if no_improve >= args.patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
            
            if lr_now < 1e-8:
                print(f"\n⛔ LR troppo basso, stopping")
                break
    
    print("\n" + "=" * 60)
    print("🏁 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Miglioramento: {72.25 - best_mae:.2f} punti")
    
    if best_mae < 60:
        print("   🎯 TARGET < 60 RAGGIUNTO!")
    elif best_mae < 70:
        print("   ✅ Migliorato!")
    else:
        print("   ⚠️ Nessun miglioramento significativo")
    
    print("=" * 60)


if __name__ == '__main__':
    main()