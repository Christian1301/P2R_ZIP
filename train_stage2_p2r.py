"""
Train Stage 2 P2R V15 - PRONTO PER L'USO
Basato su V9 (MAE ~69) con correzioni critiche

Esegui con:
    python train_stage2_p2r.py --config config.yaml
"""

import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# ============================================================
# IMPORT DEI TUOI MODULI - ADATTA SE NECESSARIO
# ============================================================
from models.backbone import BackboneWrapper
from models.p2r_head import P2RHead


# ============================================================
# SIMPLE P2R LOSS - Usa gt_density direttamente
# ============================================================

class SimpleP2RLoss(nn.Module):
    """
    Loss semplificata per P2R che usa gt_density direttamente.
    """
    def __init__(self, count_weight=2.0, spatial_weight=0.15):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
    
    def forward(self, pred_density, gt_density):
        """
        Args:
            pred_density: [B, 1, H, W] predicted density
            gt_density: [B, 1, H, W] ground truth density
        """
        # Allinea dimensioni se necessario
        if pred_density.shape[-2:] != gt_density.shape[-2:]:
            gt_density = F.interpolate(
                gt_density, 
                size=pred_density.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Count loss (L1)
        pred_count = pred_density.sum(dim=[1, 2, 3])
        gt_count = gt_density.sum(dim=[1, 2, 3])
        count_loss = torch.abs(pred_count - gt_count).mean()
        
        # Spatial loss (MSE normalizzato)
        pred_norm = pred_density / (pred_density.sum(dim=[2, 3], keepdim=True) + 1e-6)
        gt_norm = gt_density / (gt_density.sum(dim=[2, 3], keepdim=True) + 1e-6)
        spatial_loss = F.mse_loss(pred_norm, gt_norm)
        
        total_loss = self.count_weight * count_loss + self.spatial_weight * spatial_loss
        
        return {
            'total_loss': total_loss,
            'count_loss': count_loss,
            'spatial_loss': spatial_loss
        }
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# UTILITIES
# ============================================================

class AverageMeter:
    """Computa e memorizza media e valore corrente."""
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
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """Early stopping con patience alta (200 per V9)."""
    def __init__(self, patience: int = 200, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def get_scheduler(optimizer, cfg):
    """Crea scheduler con warmup + cosine annealing."""
    warmup_epochs = cfg['OPTIM_P2R']['WARMUP_EPOCHS']
    total_epochs = cfg['OPTIM_P2R']['EPOCHS']
    min_lr = cfg['OPTIM_P2R'].get('MIN_LR', 1e-7)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(min_lr / cfg['OPTIM_P2R']['LR'], 
                      0.5 * (1 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(
    p2r_head,
    backbone,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    cfg,
    scaler=None
):
    """Training di una singola epoca."""
    p2r_head.train()
    backbone.eval()  # Backbone quasi congelato
    
    losses = AverageMeter()
    count_losses = AverageMeter()
    spatial_losses = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, (images, gt_density, points_list) in enumerate(pbar):
        images = images.to(device)
        gt_density = gt_density.to(device)
        # Calcola gt_counts dalla somma della density
        gt_counts = gt_density.sum(dim=[1, 2, 3])
        
        optimizer.zero_grad()
        
        # Forward
        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            # Backbone features (no grad per quasi-freeze)
            with torch.no_grad():
                features = backbone(images)
            
            # P2R prediction
            density = p2r_head(features)
            
            # Loss
            loss_dict = criterion(density, gt_density)
            loss = loss_dict['total_loss']
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                p2r_head.parameters(), 
                cfg['OPTIM_P2R'].get('GRAD_CLIP', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                p2r_head.parameters(), 
                cfg['OPTIM_P2R'].get('GRAD_CLIP', 1.0)
            )
            optimizer.step()
        
        # Metriche
        with torch.no_grad():
            pred_counts = density.sum(dim=[1, 2, 3])
            mae = torch.abs(pred_counts - gt_counts).mean()
        
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        count_losses.update(loss_dict['count_loss'].item(), batch_size)
        spatial_losses.update(loss_dict.get('spatial_loss', torch.tensor(0)).item(), batch_size)
        mae_meter.update(mae.item(), batch_size)
        
        # Progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'MAE': f'{mae_meter.avg:.2f}',
            'scale': f'{p2r_head.get_scale():.2f}'
        })
    
    return {
        'loss': losses.avg,
        'count_loss': count_losses.avg,
        'spatial_loss': spatial_losses.avg,
        'mae': mae_meter.avg,
        'log_scale': p2r_head.log_scale.item(),
        'scale': p2r_head.get_scale()
    }


@torch.no_grad()
def validate(p2r_head, backbone, dataloader, criterion, device, cfg):
    """Validazione."""
    p2r_head.eval()
    backbone.eval()
    
    losses = AverageMeter()
    all_preds = []
    all_gts = []
    
    for images, gt_density, points_list in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        gt_counts = gt_density.sum(dim=[1, 2, 3])
        
        # Forward
        features = backbone(images)
        density = p2r_head(features)
        
        # Loss
        loss_dict = criterion(density, gt_density)
        
        # Predicted counts
        pred_counts = density.sum(dim=[1, 2, 3])
        
        losses.update(loss_dict['total_loss'].item(), images.size(0))
        
        for pred, gt in zip(pred_counts, gt_counts):
            all_preds.append(pred.item())
            all_gts.append(gt.item())
    
    # Calcola MAE e RMSE
    all_preds = torch.tensor(all_preds)
    all_gts = torch.tensor(all_gts)
    
    mae = torch.abs(all_preds - all_gts).mean().item()
    rmse = torch.sqrt(((all_preds - all_gts) ** 2).mean()).item()
    
    # Bias (ratio medio)
    valid_mask = all_gts > 0
    if valid_mask.sum() > 0:
        bias = (all_preds[valid_mask] / all_gts[valid_mask]).mean().item()
    else:
        bias = 1.0
    
    return {
        'loss': losses.avg,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'log_scale': p2r_head.log_scale.item(),
        'scale': p2r_head.get_scale()
    }


# ============================================================
# MAIN
# ============================================================

def main(cfg_path: str):
    """Main training function."""
    
    # Carica config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup
    device = torch.device(cfg.get('DEVICE', 'cuda') if torch.cuda.is_available() else 'cpu')
    run_name = cfg.get('RUN_NAME', 'shha_v15')
    run_dir = Path(cfg.get('EXPERIMENT_DIR', 'exp')) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Seed
    seed = cfg.get('SEED', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Salva config
    with open(run_dir / 'config_stage2.yaml', 'w') as f:
        yaml.dump(cfg, f)
    
    logger.info("=" * 60)
    logger.info("STAGE 2 P2R TRAINING V15")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {cfg['OPTIM_P2R']['EPOCHS']}")
    logger.info(f"Warmup: {cfg['OPTIM_P2R']['WARMUP_EPOCHS']}")
    logger.info(f"Patience: {cfg['OPTIM_P2R']['EARLY_STOPPING_PATIENCE']}")
    logger.info(f"LOG_SCALE_INIT: {cfg['P2R_LOSS']['LOG_SCALE_INIT']}")
    logger.info(f"USE_MULTI_SCALE: {cfg['P2R_LOSS']['USE_MULTI_SCALE']}")
    logger.info(f"COUNT_WEIGHT: {cfg['P2R_LOSS']['COUNT_WEIGHT']}")
    logger.info(f"SPATIAL_WEIGHT: {cfg['P2R_LOSS']['SPATIAL_WEIGHT']}")
    logger.info("=" * 60)
    
    # ============================================================
    # MODELLI
    # ============================================================
    
    # Backbone
    backbone = BackboneWrapper(
        name=cfg['MODEL'].get('BACKBONE', 'vgg16_bn'),
        pretrained=cfg['MODEL'].get('BACKBONE_PRETRAINED', True)
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Freeze backbone (quasi completamente)
    for param in backbone.parameters():
        param.requires_grad = False
    
    # P2R Head - già ha log_scale=4.0 hardcoded
    p2r_head = P2RHead(
        in_channel=512,
        fea_channel=cfg['MODEL'].get('P2R_FEA_CHANNEL', 64),
        out_stride=cfg['MODEL'].get('P2R_OUT_STRIDE', 16)
    ).to(device)
    
    logger.info(f"P2R Head initialized with log_scale={p2r_head.log_scale.item():.4f} (scale={p2r_head.get_scale():.2f})")
    
    # ============================================================
    # DATASET
    # ============================================================
    
    dataset_section = cfg.get('DATASET', 'shha')
    data_section = cfg.get('DATA')
    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = {}
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
        data_cfg.update(dataset_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}

    alias_map = {
        'shha': 'shha',
        'shanghaitecha': 'shha',
        'shanghaitechparta': 'shha',
        'ucf': 'ucf',
        'ucfqnrf': 'ucf',
        'jhu': 'jhu',
        'nwpu': 'nwpu'
    }
    normalized_name = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = alias_map.get(normalized_name, str(dataset_name_raw).lower())

    if not data_cfg:
        raise ValueError("DATA or DATASET configuration must provide dataset parameters including ROOT.")

    block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))
    
    DatasetClass = get_dataset(dataset_name)

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]
    
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    train_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('TRAIN_SPLIT', 'train'),
        block_size=block_size,
        transforms=train_tf
    )
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('VAL_SPLIT', 'val'),
        block_size=block_size,
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['OPTIM_P2R']['BATCH_SIZE'],
        shuffle=True,
        num_workers=cfg['OPTIM_P2R'].get('NUM_WORKERS', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ============================================================
    # LOSS - SimpleP2RLoss (usa gt_density direttamente)
    # ============================================================
    
    criterion = SimpleP2RLoss(
        count_weight=cfg['P2R_LOSS']['COUNT_WEIGHT'],
        spatial_weight=cfg['P2R_LOSS']['SPATIAL_WEIGHT']
    )
    logger.info(f"Using SimpleP2RLoss (count_weight={cfg['P2R_LOSS']['COUNT_WEIGHT']}, spatial_weight={cfg['P2R_LOSS']['SPATIAL_WEIGHT']})")
    
    # ============================================================
    # OPTIMIZER
    # ============================================================
    
    # Parametri con LR differenziato per log_scale
    param_groups = []
    log_scale_params = []
    other_params = []
    
    for name, param in p2r_head.named_parameters():
        if 'log_scale' in name:
            log_scale_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': other_params, 'lr': cfg['OPTIM_P2R']['LR']},
        {'params': log_scale_params, 'lr': cfg['OPTIM_P2R']['LR'] * cfg['P2R_LOSS'].get('LOG_SCALE_LR_MULT', 0.1)}
    ]
    
    optimizer = AdamW(
        param_groups,
        weight_decay=cfg['OPTIM_P2R'].get('WEIGHT_DECAY', 1e-4)
    )
    
    # Scheduler
    scheduler = get_scheduler(optimizer, cfg)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg['OPTIM_P2R']['EARLY_STOPPING_PATIENCE']
    )
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    
    best_mae = float('inf')
    history = []
    
    for epoch in range(cfg['OPTIM_P2R']['EPOCHS']):
        # Training
        train_metrics = train_one_epoch(
            p2r_head, backbone, train_loader, criterion,
            optimizer, device, epoch, cfg, scaler
        )
        
        # Validation
        val_metrics = validate(
            p2r_head, backbone, val_loader, criterion, device, cfg
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(
            f"Epoch {epoch:4d} | "
            f"Train MAE: {train_metrics['mae']:6.2f} | "
            f"Val MAE: {val_metrics['mae']:6.2f} | "
            f"Val RMSE: {val_metrics['rmse']:6.2f} | "
            f"log_scale: {val_metrics['log_scale']:6.3f} | "
            f"scale: {val_metrics['scale']:6.2f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # History
        history.append({
            'epoch': epoch,
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'log_scale': val_metrics['log_scale'],
            'lr': current_lr
        })
        
        # Save best model
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': p2r_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': best_mae,
                'rmse': val_metrics['rmse'],
                'log_scale': p2r_head.log_scale.item(),
                'scale': p2r_head.get_scale()
            }, run_dir / 'stage2_best.pth')
            logger.info(f"  ✓ New best MAE: {best_mae:.2f}")
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': p2r_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mae': val_metrics['mae'],
            'log_scale': p2r_head.log_scale.item()
        }, run_dir / 'stage2_last.pth')
        
        # Save history
        with open(run_dir / 'stage2_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if early_stopping(val_metrics['mae']):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Best MAE: {best_mae:.2f}")
    logger.info(f"Final log_scale: {p2r_head.log_scale.item():.4f}")
    logger.info(f"Final scale: {p2r_head.get_scale():.2f}")
    logger.info(f"Best model saved: {run_dir / 'stage2_best.pth'}")
    logger.info("=" * 60)
    
    return best_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 P2R V15")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)