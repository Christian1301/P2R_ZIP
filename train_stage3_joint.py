#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V2 - Joint Training con Anti-Overfitting

OBIETTIVO:
Fine-tuning collaborativo ZIP + P2R per migliorare la sinergia tra i due componenti.

MODIFICHE DA V1:
1. cell_area calcolato correttamente da config
2. Early stopping implementato
3. Logging dettagliato con salvataggio checkpoint robusto
4. Anti-overfitting: weight decay, grad clip, LR basso
5. Validazione ogni N epoche (configurabile)
6. Warmup opzionale
7. Supporto resume robusto

QUANDO USARE:
- Dopo Stage 1 (ZIP) e Stage 2 (P2R) completati
- Per fine-tuning finale del sistema completo
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
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid
from losses.zip_nll import zip_nll


# =============================================================================
# UTILITIES
# =============================================================================

class AverageMeter:
    """Calcola e memorizza media e valore corrente."""
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
    """Salva checkpoint con verifica."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(state, filepath)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"💾 {description} salvato: {filepath} ({size/1024/1024:.1f} MB)")
            sys.stdout.flush()
            return True
        else:
            print(f"❌ ERRORE: {description} non trovato dopo salvataggio!")
            return False
    except Exception as e:
        print(f"❌ ERRORE salvataggio {description}: {e}")
        return False


def get_lr(optimizer):
    """Ottieni LR corrente."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# MULTI-SCALE P2R LOSS
# =============================================================================

class MultiScaleP2RLoss(nn.Module):
    """
    Loss P2R Multi-Scala.
    Calcola count loss a multiple risoluzioni per consistenza multi-scala.
    """
    
    def __init__(self, scales=[1, 2, 4], weights=[1.0, 0.5, 0.25], count_weight=2.0):
        super().__init__()
        self.scales = scales
        self.count_weight = count_weight
        # Normalizza pesi
        tot = sum(weights)
        self.weights = [w / tot for w in weights]

    def forward(self, pred, points_list, cell_area):
        """
        Args:
            pred: [B, 1, H, W] density predictions
            points_list: lista di tensori con coordinate GT
            cell_area: area della cella per scaling count
        """
        device = pred.device
        total_loss = torch.tensor(0.0, device=device)
        
        gt_counts = []
        pred_counts = []
        
        for scale_idx, (scale, weight) in enumerate(zip(self.scales, self.weights)):
            if scale == 1:
                pred_scaled = pred
            else:
                pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                pred_scaled = pred_scaled * (scale ** 2)
            
            scale_losses = []
            for i, pts in enumerate(points_list):
                gt = len(pts)
                pred_count = pred_scaled[i].sum() / cell_area
                scale_losses.append(torch.abs(pred_count - gt))
                
                # Salva per metriche (solo scala 1)
                if scale == 1:
                    gt_counts.append(gt)
                    pred_counts.append(pred_count.item())
            
            scale_loss = torch.stack(scale_losses).mean()
            total_loss = total_loss + weight * scale_loss
        
        return total_loss * self.count_weight, gt_counts, pred_counts


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, default_down):
    """Validazione completa."""
    model.eval()
    
    all_mae = []
    all_mse = []
    
    for images, densities, points in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images)
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
    default_down,
    joint_cfg,
    grad_clip=1.0
):
    """Training di una epoca con loss congiunta ZIP + P2R."""
    model.train()
    
    loss_meter = AverageMeter()
    loss_zip_meter = AverageMeter()
    loss_p2r_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    # Pesi loss dal config
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
        # 1. ZIP LOSS - Mantiene struttura della maschera
        # =====================================================================
        pi = torch.sigmoid(outputs['logit_pi_maps'][:, 1:2])  # [B, 1, H_zip, W_zip]
        lam = outputs['lambda_maps']
        
        block_h, block_w = pi.shape[-2:]
        factor_h = H_in / block_h
        factor_w = W_in / block_w
        
        # Ground truth counts per blocco
        gt_counts_block = F.adaptive_avg_pool2d(gt_density, (block_h, block_w))
        gt_counts_block = gt_counts_block * (factor_h * factor_w)
        
        loss_zip = zip_nll(pi, lam, gt_counts_block)
        
        # =====================================================================
        # 2. P2R LOSS - Precisione conteggio
        # =====================================================================
        pred_p2r = outputs['p2r_density']
        pred_p2r, down_tuple, _ = canonicalize_p2r_grid(pred_p2r, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        loss_p2r, gt_counts, pred_counts = p2r_criterion(pred_p2r, points_list, cell_area)
        
        # =====================================================================
        # 3. LOSS TOTALE PESATA
        # =====================================================================
        loss = (w_p2r * loss_p2r) + (w_zip * loss_zip)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Metriche
        mae = np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)])
        
        loss_meter.update(loss.item())
        loss_zip_meter.update(loss_zip.item())
        loss_p2r_meter.update(loss_p2r.item())
        mae_meter.update(mae)
        
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
    parser.add_argument('--no-resume', action='store_true', help='Disabilita resume automatico')
    parser.add_argument('--stage2-ckpt', type=str, default=None, help='Path specifico checkpoint Stage 2')
    args = parser.parse_args()

    # Carica config
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
    
    epochs = joint_cfg.get('EPOCHS', 300)
    batch_size = joint_cfg.get('BATCH_SIZE', 4)
    lr_backbone = float(joint_cfg.get('LR_BACKBONE', 0))  # Default: freeze
    lr_heads = float(joint_cfg.get('LR_HEADS', 5e-6))
    weight_decay = float(joint_cfg.get('WEIGHT_DECAY', 5e-4))
    val_interval = joint_cfg.get('VAL_INTERVAL', 5)
    patience = joint_cfg.get('EARLY_STOPPING_PATIENCE', 100)
    grad_clip = joint_cfg.get('GRAD_CLIP', 0.5)
    warmup_epochs = joint_cfg.get('WARMUP_EPOCHS', 10)
    
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 3 V2 - Joint Training (Anti-Overfitting)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR heads: {lr_heads}, LR backbone: {lr_backbone}")
    print(f"Weight decay: {weight_decay}")
    print(f"Grad clip: {grad_clip}")
    print(f"Patience: {patience}")
    print(f"Val interval: {val_interval}")
    print(f"Warmup: {warmup_epochs}")
    print(f"P2R weight: {loss_cfg.get('FIXED_P2R_WEIGHT', 1.0)}")
    print(f"ZIP weight: {loss_cfg.get('FIXED_ZIP_WEIGHT', 0.1)}")
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
        num_workers=joint_cfg.get('NUM_WORKERS', 4),
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
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # =========================================================================
    # MODELLO
    # =========================================================================
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
    
    # =========================================================================
    # CARICA STAGE 2 CHECKPOINT
    # =========================================================================
    stage2_loaded = False
    
    if args.stage2_ckpt:
        stage2_candidates = [args.stage2_ckpt]
    else:
        stage2_candidates = [
            os.path.join(output_dir, "stage2_best.pth"),
            os.path.join(output_dir, "stage2_last.pth"),
        ]
    
    print(f"\n🔍 Cercando Stage 2 checkpoint...")
    for ckpt_path in stage2_candidates:
        if os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                if "model" in state:
                    model.load_state_dict(state["model"], strict=False)
                    if "mae" in state:
                        print(f"✅ Caricato Stage 2 da: {ckpt_path} (MAE: {state['mae']:.2f})")
                    else:
                        print(f"✅ Caricato Stage 2 da: {ckpt_path}")
                else:
                    model.load_state_dict(state, strict=False)
                    print(f"✅ Caricato Stage 2 da: {ckpt_path}")
                stage2_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Errore caricamento {ckpt_path}: {e}")
                continue
    
    if not stage2_loaded:
        print("❌ Stage 2 checkpoint NON TROVATO!")
        print("   Devi completare Stage 2 prima di lanciare Stage 3.")
        print(f"   Cercato in: {stage2_candidates}")
        return
    
    # =========================================================================
    # SETUP OPTIMIZER
    # =========================================================================
    print("\n🔧 Setup parametri:")
    
    # Freeze tutto il backbone inizialmente
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Sblocca ultimi layer del backbone (layer 34+ per VGG)
    trainable_backbone_params = []
    if lr_backbone > 0:
        backbone_start_layer = 34  # Per VGG16/19
        for i, layer in enumerate(model.backbone.body):
            if i >= backbone_start_layer:
                for param in layer.parameters():
                    param.requires_grad = True
                    trainable_backbone_params.append(param)
        print(f"   Backbone layer {backbone_start_layer}+: LR={lr_backbone}")
    else:
        print("   Backbone: FROZEN")
    
    # Heads
    p2r_params = list(model.p2r_head.parameters())
    zip_params = list(model.zip_head.parameters())
    
    print(f"   P2R head: LR={lr_heads}")
    print(f"   ZIP head: LR={lr_heads}")
    
    # Costruisci param groups
    param_groups = []
    if trainable_backbone_params:
        param_groups.append({
            'params': trainable_backbone_params,
            'lr': lr_backbone,
            'name': 'backbone'
        })
    param_groups.append({
        'params': p2r_params,
        'lr': lr_heads,
        'name': 'p2r_head'
    })
    param_groups.append({
        'params': zip_params,
        'lr': lr_heads,
        'name': 'zip_head'
    })
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n   Trainabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Scheduler con warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss P2R multi-scala
    p2r_criterion = MultiScaleP2RLoss(
        scales=[1, 2, 4],
        weights=[1.0, 0.5, 0.25],
        count_weight=2.0
    ).to(device)
    
    # =========================================================================
    # RESUME AUTOMATICO
    # =========================================================================
    start_epoch = 1
    best_mae = float('inf')
    no_improve_count = 0
    
    stage3_last_path = os.path.join(output_dir, "stage3_last.pth")
    if not args.no_resume and os.path.isfile(stage3_last_path):
        print(f"\n🔄 Trovato checkpoint Stage 3: {stage3_last_path}")
        try:
            ckpt = torch.load(stage3_last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt.get('best_mae', float('inf'))
            no_improve_count = ckpt.get('no_improve_count', 0)
            print(f"✅ Resume da epoca {start_epoch}, best MAE: {best_mae:.2f}")
        except Exception as e:
            print(f"⚠️ Errore resume: {e}, ripartendo da Stage 2")
    
    # =========================================================================
    # VALUTAZIONE INIZIALE
    # =========================================================================
    print("\n📋 Valutazione iniziale:")
    val_results = validate(model, val_loader, device, default_down)
    print(f"   MAE: {val_results['mae']:.2f}, RMSE: {val_results['rmse']:.2f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
        save_checkpoint(
            {
                'epoch': 0,
                'model': model.state_dict(),
                'mae': best_mae,
                'rmse': val_results['rmse'],
            },
            os.path.join(output_dir, "stage3_best.pth"),
            "Best iniziale"
        )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n🚀 START Joint Training: {epochs} epochs")
    print(f"   Baseline MAE: {best_mae:.2f}")
    print()
    sys.stdout.flush()
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_results = train_one_epoch(
            model, p2r_criterion, optimizer, train_loader,
            device, epoch, default_down, loss_cfg, grad_clip
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0 or epoch == 1:
            val_results = validate(model, val_loader, device, default_down)
            
            current_lr = get_lr(optimizer)
            mae = val_results['mae']
            
            # Check improvement
            improved = mae < best_mae
            status = "✅ NEW BEST" if improved else ""
            
            print(f"Epoch {epoch:4d} | "
                  f"Train MAE: {train_results['mae']:.1f} | "
                  f"Val MAE: {mae:.2f} | "
                  f"ZIP: {train_results['loss_zip']:.2f} | "
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
                        'rmse': val_results['rmse'],
                    },
                    os.path.join(output_dir, "stage3_best.pth"),
                    f"Stage3 Best (MAE={mae:.2f})"
                )
            else:
                no_improve_count += val_interval
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Salva checkpoint per resume
        if epoch % 10 == 0:
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
    
    # Salvataggio finale
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
        "Stage3 Final"
    )
    
    # =========================================================================
    # RISULTATI FINALI
    # =========================================================================
    print("\n" + "=" * 60)
    print("🏁 STAGE 3 JOINT TRAINING COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage3_best.pth")
    print("=" * 60)
    print(f"\n📌 Prossimo step: python evaluate.py --config {args.config} --checkpoint stage3_best.pth")


if __name__ == "__main__":
    main()