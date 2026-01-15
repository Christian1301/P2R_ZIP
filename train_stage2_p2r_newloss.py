#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V4 - Training P2R con Output BINARIO (come paper originale)

Questo script usa:
1. P2R head con output binario (2 canali → differenza)
2. P2RLoss originale con matching point-to-region
3. Conteggio basato su (logits > 0).count()

È l'implementazione fedele al paper P2R.
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

from models.p2r_zip_model_v2 import P2R_ZIP_Model_V2
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn
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
            print(f"💾 {description}: {filepath} ({size/1024/1024:.1f} MB)")
            sys.stdout.flush()
            return True
        return False
    except Exception as e:
        print(f"❌ Errore salvataggio: {e}")
        return False


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device):
    """
    Validazione con conteggio binario (logits > 0).
    """
    model.eval()
    
    all_mae = []
    all_mse = []
    total_pred = 0
    total_gt = 0
    
    for images, gt_density, points in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device)
        
        outputs = model(images)
        logits = outputs['p2r_logits']  # [B, 1, H, W]
        
        for i, pts in enumerate(points):
            gt_count = len(pts)
            # Conteggio P2R: celle con logit > 0
            pred_count = (logits[i] > 0).sum().item()
            
            all_mae.append(abs(pred_count - gt_count))
            all_mse.append((pred_count - gt_count) ** 2)
            
            total_pred += pred_count
            total_gt += gt_count
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    ratio = total_pred / max(total_gt, 1)
    
    return {
        'mae': mae, 
        'rmse': rmse,
        'ratio': ratio,
        'total_pred': total_pred,
        'total_gt': total_gt
    }


# =============================================================================
# TRAINING EPOCH
# =============================================================================

def train_one_epoch(
    model, 
    criterion, 
    optimizer, 
    loader, 
    device, 
    epoch,
    grad_clip=5.0
):
    """Training con P2RLoss originale."""
    model.train()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        # Forward
        outputs = model(images)
        logits = outputs['p2r_logits']  # [B, 1, H, W]
        
        # P2RLoss vuole [B, H, W] e coordinate (y, x)
        logits_squeezed = logits.squeeze(1)
        
        # Il downsampling del P2R head
        _, _, H_in, W_in = images.shape
        _, H_out, W_out = logits_squeezed.shape
        down_rate = H_in // H_out
        
        # Converti punti da (x, y) a (y, x) per P2RLoss
        points_yx = []
        for pts in points_list:
            if pts.numel() > 0:
                pts_yx = pts[:, [1, 0]]  # Swap x,y → y,x
                if pts.shape[1] > 2:
                    pts_yx = torch.cat([pts_yx, pts[:, 2:]], dim=1)
                points_yx.append(pts_yx)
            else:
                points_yx.append(pts)
        
        # Loss
        loss = criterion(logits_squeezed, points_yx, down_rate)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Metriche training
        with torch.no_grad():
            mae_batch = 0
            for i, pts in enumerate(points_list):
                gt = len(pts)
                pred = (logits[i] > 0).sum().item()
                mae_batch += abs(pred - gt)
            mae_batch /= len(points_list)
        
        loss_meter.update(loss.item())
        mae_meter.update(mae_batch)
        
        pbar.set_postfix({
            'L': f"{loss_meter.avg:.3f}",
            'MAE': f"{mae_meter.avg:.1f}",
        })
    
    return {
        'loss': loss_meter.avg,
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
    parser.add_argument('--stage1-ckpt', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config non trovato: {args.config}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get("DEVICE", "cuda"))
    init_seeds(cfg.get("SEED", 2025))
    
    # Parametri (allineati al paper P2R)
    data_cfg = cfg['DATA']
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    loss_cfg = cfg.get('P2R_LOSS', {})
    
    epochs = p2r_cfg.get('EPOCHS', 1500)
    batch_size = p2r_cfg.get('BATCH_SIZE', 16)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_backbone = float(p2r_cfg.get('LR_BACKBONE', 1e-5))
    weight_decay = float(p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    grad_clip = float(p2r_cfg.get('GRAD_CLIP', 5.0))
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 300)
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip_v2')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V4 - P2R BINARIO (come paper originale)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch_size}")
    print(f"LR: {lr}, LR backbone: {lr_backbone}")
    print(f"Grad clip: {grad_clip}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    sys.stdout.flush()
    
    # =========================================================================
    # DATASET
    # =========================================================================
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
        train_ds,
        batch_size=batch_size,
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
    
    # =========================================================================
    # MODELLO V2 (con P2R head binario)
    # =========================================================================
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    
    model = P2R_ZIP_Model_V2(
        backbone_name=cfg["MODEL"].get("BACKBONE", "vgg16_bn"),
        fusion_channels=256,
        pi_thresh=cfg["MODEL"].get("ZIP_PI_THRESH", 0.3),
        use_ste_mask=cfg["MODEL"].get("USE_STE_MASK", True),
        zip_head_kwargs={
            "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
            "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
            "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
            "lambda_noise_std": 0.0,
        },
    ).to(device)
    
    print(f"🔧 Modello V2 creato (P2R head binario)")
    print(f"   Down factor: {model.down_factor}")
    
    # =========================================================================
    # CARICA STAGE 1 (solo backbone e ZIP head)
    # =========================================================================
    stage1_loaded = False
    
    if args.stage1_ckpt:
        stage1_candidates = [args.stage1_ckpt]
    else:
        stage1_candidates = [
            os.path.join(output_dir, "stage1_best_acc.pth"),
            os.path.join(output_dir, "stage1_best.pth"),
        ]
        # Cerca anche nella cartella vecchia
        old_dir = output_dir.replace('_v2', '').replace('_binary', '')
        stage1_candidates.extend([
            os.path.join(old_dir, "stage1_best_acc.pth"),
            os.path.join(old_dir, "stage1_best.pth"),
        ])
    
    print(f"\n🔍 Cercando Stage 1 checkpoint...")
    for ckpt_path in stage1_candidates:
        if os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                state_dict = state.get("model", state)
                
                # Carica solo backbone e ZIP head (P2R head è nuovo)
                filtered = {}
                for k, v in state_dict.items():
                    if 'backbone' in k or 'zip_head' in k or 'fusion' in k:
                        filtered[k] = v
                
                missing, unexpected = model.load_state_dict(filtered, strict=False)
                print(f"✅ Stage 1 caricato da: {ckpt_path}")
                print(f"   Missing (atteso - nuovo P2R head): {len(missing)}")
                stage1_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Errore: {e}")
    
    if not stage1_loaded:
        print("⚠️ Stage 1 NON trovato - training P2R da zero")
    
    # =========================================================================
    # OPTIMIZER (stile paper P2R)
    # =========================================================================
    print("\n🔧 Setup optimizer:")
    
    # Congela ZIP head (già addestrato)
    for param in model.zip_head.parameters():
        param.requires_grad = False
    print("   ZIP head: FROZEN")
    
    # Param groups con LR differenziate
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.p2r_head.parameters()) + list(model.fusion.parameters())
    
    param_groups = [
        {'params': head_params, 'lr': lr, 'name': 'head'},
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
    ]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Trainabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    
    # Scheduler (Step LR come paper)
    decay_epochs = p2r_cfg.get('LR_DECAY_EPOCHS', 3500)
    decay_rate = p2r_cfg.get('LR_DECAY_RATE', 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epochs, gamma=decay_rate)
    
    # P2RLoss originale
    criterion = P2RLossOriginal(
        factor=1,
        min_radius=loss_cfg.get('MIN_RADIUS', 8),
        max_radius=loss_cfg.get('MAX_RADIUS', 96),
        cost_class=loss_cfg.get('COST_CLASS', 1),
        cost_point=loss_cfg.get('COST_POINT', 8)
    ).to(device)
    
    print(f"\n📊 P2RLoss: min_r={loss_cfg.get('MIN_RADIUS', 8)}, max_r={loss_cfg.get('MAX_RADIUS', 96)}")
    
    # =========================================================================
    # RESUME
    # =========================================================================
    start_epoch = 1
    best_mae = float('inf')
    no_improve_count = 0
    
    stage2_last = os.path.join(output_dir, "stage2_binary_last.pth")
    if not args.no_resume and os.path.isfile(stage2_last):
        try:
            ckpt = torch.load(stage2_last, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt.get('best_mae', float('inf'))
            no_improve_count = ckpt.get('no_improve_count', 0)
            print(f"✅ Resume da epoca {start_epoch}, best MAE: {best_mae:.2f}")
        except Exception as e:
            print(f"⚠️ Errore resume: {e}")
    
    # =========================================================================
    # VALIDAZIONE INIZIALE
    # =========================================================================
    print("\n📋 Valutazione iniziale:")
    val_results = validate(model, val_loader, device)
    print(f"   MAE: {val_results['mae']:.2f}")
    print(f"   Ratio pred/gt: {val_results['ratio']:.3f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n🚀 START Training: {epochs} epochs")
    print(f"   Baseline MAE: {best_mae:.2f}")
    print()
    sys.stdout.flush()
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_results = train_one_epoch(
            model, criterion, optimizer, train_loader,
            device, epoch, grad_clip
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0 or epoch == 1:
            val_results = validate(model, val_loader, device)
            
            current_lr = get_lr(optimizer)
            mae = val_results['mae']
            
            improved = mae < best_mae
            status = "✅ NEW BEST" if improved else ""
            
            print(f"Epoch {epoch:4d} | "
                  f"Train L: {train_results['loss']:.3f} MAE: {train_results['mae']:.1f} | "
                  f"Val MAE: {mae:.2f} (r={val_results['ratio']:.2f}) | "
                  f"LR: {current_lr:.2e} | "
                  f"Best: {best_mae:.2f} {status}")
            sys.stdout.flush()
            
            if improved:
                best_mae = mae
                no_improve_count = 0
                
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'mae': mae,
                        'ratio': val_results['ratio'],
                    },
                    os.path.join(output_dir, "stage2_binary_best.pth"),
                    f"Stage2 Best (MAE={mae:.2f})"
                )
            else:
                no_improve_count += val_interval
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
        
        # Checkpoint periodico
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
                os.path.join(output_dir, "stage2_binary_last.pth"),
                f"Stage2 Ckpt (Ep {epoch})"
            )
    
    # Salvataggio finale
    save_checkpoint(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mae': best_mae,
        },
        os.path.join(output_dir, "stage2_binary_last.pth"),
        "Stage2 Final"
    )
    
    # =========================================================================
    # RISULTATI
    # =========================================================================
    print("\n" + "=" * 60)
    print("🏁 STAGE 2 (BINARY) COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_binary_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()