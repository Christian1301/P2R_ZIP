#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 V3 - Training P2R con Loss ORIGINALE dal paper

DIFFERENZE CHIAVE dalla versione precedente:
1. Usa P2RLoss originale con matching point-to-region
2. Conteggio basato su pixel > 0 (non integrazione density)
3. Iperparametri allineati al paper originale
4. Vincoli spaziali che prevengono overfitting

La P2RLoss originale funziona così:
- Per ogni punto GT, trova la cella più vicina
- Assegna target binario (1 = persona presente)
- Calcola BCE con weight dinamici
- Il modello deve predire DOVE sono le persone, non solo QUANTE
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

# Import del modello e dataset
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

# Import P2RLoss originale
from losses.p2rloss_original import P2RLossOriginal


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


def convert_points_format(points, source_format='xy', target_format='yx'):
    """
    Converte il formato delle coordinate dei punti.
    
    Args:
        points: tensor [N, 2+] o [N, 3+]
        source_format: 'xy' o 'yx'
        target_format: 'xy' o 'yx'
    
    Returns:
        points convertiti
    """
    if source_format == target_format:
        return points
    
    if points.numel() == 0:
        return points
    
    points_out = points.clone()
    # Scambia colonne 0 e 1
    points_out[:, 0], points_out[:, 1] = points[:, 1].clone(), points[:, 0].clone()
    return points_out


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, down_rate):
    """
    Validazione con conteggio basato su pixel > 0 (come paper originale).
    """
    model.eval()
    
    all_mae = []
    all_mse = []
    
    for images, gt_density, points in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device)
        
        outputs = model(images)
        pred_density = outputs['p2r_density']  # [B, 1, H, W]
        
        for i, pts in enumerate(points):
            gt_count = len(pts)
            
            # Conteggio P2R originale: conta pixel con valore > 0 (dopo sigmoid)
            # Il modello outputta logits, quindi applichiamo sigmoid
            pred_prob = torch.sigmoid(pred_density[i])
            pred_count = (pred_prob > 0.5).sum().item()
            
            # Alternativa: conta pixel con logit > 0 (equivalente a prob > 0.5)
            # pred_count = (pred_density[i] > 0).sum().item()
            
            all_mae.append(abs(pred_count - gt_count))
            all_mse.append((pred_count - gt_count) ** 2)
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    
    return {'mae': mae, 'rmse': rmse}


@torch.no_grad()
def validate_integration(model, val_loader, device, down_rate):
    """
    Validazione alternativa con integrazione density (per confronto).
    """
    model.eval()
    
    all_mae = []
    all_mse = []
    
    cell_area = down_rate * down_rate
    
    for images, gt_density, points in tqdm(val_loader, desc="Validate (int)", leave=False):
        images = images.to(device)
        
        outputs = model(images)
        pred_density = outputs['p2r_density']
        
        for i, pts in enumerate(points):
            gt_count = len(pts)
            
            # Integrazione density
            pred_prob = torch.sigmoid(pred_density[i])
            pred_count = pred_prob.sum().item() / cell_area
            
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
    criterion, 
    optimizer, 
    loader, 
    device, 
    epoch,
    down_rate,
    grad_clip=5.0,
    points_format='yx'  # Formato punti nel tuo dataset
):
    """
    Training di una epoca con P2RLoss originale.
    """
    model.train()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Stage2 P2R [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        # Converti formato punti se necessario
        # P2RLoss originale usa (y, x)
        if points_format == 'xy':
            points_list = [convert_points_format(p, 'xy', 'yx') for p in points_list]
        
        # Forward
        outputs = model(images)
        pred_density = outputs['p2r_density']  # [B, 1, H, W]
        
        # Rimuovi dimensione canale per P2RLoss
        pred_density_squeezed = pred_density.squeeze(1)  # [B, H, W]
        
        # Calcola loss
        loss = criterion(pred_density_squeezed, points_list, down_rate)
        
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
                pred = (pred_density[i] > 0).sum().item()
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
    parser.add_argument('--no-resume', action='store_true', help='Disabilita resume automatico')
    parser.add_argument('--stage1-ckpt', type=str, default=None, help='Path specifico Stage 1 checkpoint')
    parser.add_argument('--points-format', type=str, default='yx', choices=['xy', 'yx'],
                        help='Formato coordinate punti nel dataset (default: yx)')
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
    p2r_cfg = cfg.get('OPTIM_P2R', {})
    loss_cfg = cfg.get('P2R_LOSS', {})
    
    # Iperparametri (allineati al paper P2R originale)
    epochs = p2r_cfg.get('EPOCHS', 1500)
    batch_size = p2r_cfg.get('BATCH_SIZE', 16)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_backbone = float(p2r_cfg.get('LR_BACKBONE', 1e-5))
    weight_decay = float(p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    grad_clip = float(p2r_cfg.get('GRAD_CLIP', 5.0))  # Paper usa 5.0!
    val_interval = p2r_cfg.get('VAL_INTERVAL', 5)
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 300)
    warmup_epochs = p2r_cfg.get('WARMUP_EPOCHS', 0)
    
    down_rate = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    # P2RLoss parameters
    min_radius = loss_cfg.get('MIN_RADIUS', 8)
    max_radius = loss_cfg.get('MAX_RADIUS', 96)
    cost_class = loss_cfg.get('COST_CLASS', 1)
    cost_point = loss_cfg.get('COST_POINT', 8)
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V3 - P2R Training con Loss ORIGINALE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR: {lr}, LR backbone: {lr_backbone}")
    print(f"Weight decay: {weight_decay}")
    print(f"Grad clip: {grad_clip}")
    print(f"Patience: {patience}")
    print(f"Val interval: {val_interval}")
    print(f"Down rate: {down_rate}")
    print(f"P2R Loss: min_r={min_radius}, max_r={max_radius}, cost_cls={cost_class}, cost_pt={cost_point}")
    print(f"Points format: {args.points_format}")
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
    
    print(f"Dataset train: {len(train_ds)} immagini")
    print(f"Dataset val: {len(val_ds)} immagini")
    
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
    
    print(f"🔧 Modello creato con USE_STE_MASK = {cfg['MODEL'].get('USE_STE_MASK', True)}")
    
    # =========================================================================
    # CARICA STAGE 1 CHECKPOINT
    # =========================================================================
    stage1_loaded = False
    
    if args.stage1_ckpt:
        stage1_candidates = [args.stage1_ckpt]
    else:
        stage1_candidates = [
            os.path.join(output_dir, "stage1_best_acc.pth"),
            os.path.join(output_dir, "stage1_best.pth"),
            os.path.join(output_dir, "stage1_last.pth"),
        ]
    
    print(f"\n🔍 Cercando Stage 1 checkpoint...")
    for ckpt_path in stage1_candidates:
        if os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                if "model" in state:
                    model.load_state_dict(state["model"], strict=False)
                    info = f"epoch {state.get('epoch', '?')}"
                    if "accuracy" in state:
                        info += f", acc {state['accuracy']:.1f}%"
                    print(f"✅ Caricato Stage 1 da: {ckpt_path} ({info})")
                else:
                    model.load_state_dict(state, strict=False)
                    print(f"✅ Caricato Stage 1 da: {ckpt_path}")
                stage1_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Errore caricamento {ckpt_path}: {e}")
                continue
    
    if not stage1_loaded:
        print("⚠️ Stage 1 checkpoint NON TROVATO - training da zero")
    
    # =========================================================================
    # SETUP OPTIMIZER (stile paper originale)
    # =========================================================================
    print("\n🔧 Setup parametri:")
    
    # Parametri backbone vs head (come nel paper)
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name or 'encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Freeze ZIP head (già addestrato in Stage 1)
    for param in model.zip_head.parameters():
        param.requires_grad = False
    
    print(f"   Backbone params: {sum(p.numel() for p in backbone_params):,}")
    print(f"   Head params: {sum(p.numel() for p in head_params):,}")
    print(f"   ZIP head: FROZEN")
    
    # Optimizer con LR differenziate (come paper)
    param_groups = [
        {'params': [p for p in head_params if p.requires_grad], 'lr': lr},
        {'params': backbone_params, 'lr': lr_backbone},
    ]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n   Trainabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    
    # Scheduler (Step LR come nel paper)
    decay_epochs = p2r_cfg.get('LR_DECAY_EPOCHS', 3500)
    decay_rate = p2r_cfg.get('LR_DECAY_RATE', 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=decay_epochs, 
        gamma=decay_rate
    )
    
    # P2RLoss originale
    criterion = P2RLossOriginal(
        factor=1,
        min_radius=min_radius,
        max_radius=max_radius,
        cost_class=cost_class,
        cost_point=cost_point
    ).to(device)
    
    print(f"\n📊 P2RLoss: min_r={min_radius}, max_r={max_radius}")
    
    # =========================================================================
    # RESUME
    # =========================================================================
    start_epoch = 1
    best_mae = float('inf')
    no_improve_count = 0
    
    stage2_last_path = os.path.join(output_dir, "stage2_last.pth")
    if not args.no_resume and os.path.isfile(stage2_last_path):
        print(f"\n🔄 Trovato checkpoint Stage 2: {stage2_last_path}")
        try:
            ckpt = torch.load(stage2_last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
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
    val_results = validate(model, val_loader, device, down_rate)
    val_results_int = validate_integration(model, val_loader, device, down_rate)
    print(f"   MAE (pixel>0): {val_results['mae']:.2f}, RMSE: {val_results['rmse']:.2f}")
    print(f"   MAE (integr.): {val_results_int['mae']:.2f}, RMSE: {val_results_int['rmse']:.2f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
        save_checkpoint(
            {
                'epoch': 0,
                'model': model.state_dict(),
                'mae': best_mae,
            },
            os.path.join(output_dir, "stage2_best.pth"),
            "Best iniziale"
        )
    
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
            device, epoch, down_rate, grad_clip,
            points_format=args.points_format
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0 or epoch == 1:
            val_results = validate(model, val_loader, device, down_rate)
            
            current_lr = get_lr(optimizer)
            mae = val_results['mae']
            
            # Check improvement
            improved = mae < best_mae
            status = "✅ NEW BEST" if improved else ""
            
            print(f"Epoch {epoch:4d} | "
                  f"Train L: {train_results['loss']:.3f} MAE: {train_results['mae']:.1f} | "
                  f"Val MAE: {mae:.2f} | "
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
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'mae': mae,
                    },
                    os.path.join(output_dir, "stage2_best.pth"),
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
                os.path.join(output_dir, "stage2_last.pth"),
                f"Stage2 Last (Ep {epoch})"
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
        os.path.join(output_dir, "stage2_last.pth"),
        "Stage2 Final"
    )
    
    # =========================================================================
    # RISULTATI
    # =========================================================================
    print("\n" + "=" * 60)
    print("🏁 STAGE 2 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()