#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 FUSION - Joint Training con Soft Weighting FUNZIONANTE

PROBLEMA RISOLTO:
- Stage 2 produce density calibrata per raw sum
- Soft weighting riduce la density → sottostima
- Soluzione: α warmup + scale compensation + loss appropriata

STRATEGIA:
1. α WARMUP: Parte da 0 (raw) e sale gradualmente a target (es. 0.3)
2. SCALE COMPENSATION: Fattore learnable che compensa la riduzione
3. LOSS IBRIDA: Count sulla weighted + spatial alignment bonus
4. LR DIFFERENZIATI: P2R può adattarsi, ZIP quasi frozen

FORMULA FINALE:
  count = Σ(density × scale_factor × soft_weight) / cell_area
  dove soft_weight = (1 - α) + α × π
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


ALIAS_DATASETS = {
    'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
    'shhb': 'shhb', 'shanghaitechpartb': 'shhb',
    'ucf': 'ucf', 'ucfqnrf': 'ucf',
    'nwpu': 'nwpu', 'jhu': 'jhu'
}

DEFAULT_BINS_CONFIG = {
    'shha': {
        'bins': [[0, 0], [1, 3], [4, 6], [7, 10], [11, 15], [16, 22], [23, 32], [33, 9999]],
        'bin_centers': [0.0, 2.0, 5.0, 8.5, 13.0, 19.0, 27.5, 45.0],
    },
    'shhb': {
        'bins': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9999]],
        'bin_centers': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.16],
    },
    'ucf': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'nwpu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'jhu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
}


# =============================================================================
# ALPHA SCHEDULER - Warmup graduale
# =============================================================================

class AlphaScheduler:
    """
    Scheduler per α che parte da 0 e sale gradualmente.
    
    Questo permette al modello di adattarsi progressivamente
    al soft weighting invece di subire uno shock iniziale.
    """
    def __init__(
        self,
        alpha_start: float = 0.0,
        alpha_end: float = 0.3,
        warmup_epochs: int = 50,
        schedule: str = 'linear'  # 'linear', 'cosine', 'exp'
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.current_alpha = alpha_start
    
    def step(self, epoch: int) -> float:
        """Calcola α per l'epoca corrente."""
        if epoch >= self.warmup_epochs:
            self.current_alpha = self.alpha_end
        else:
            progress = epoch / self.warmup_epochs
            
            if self.schedule == 'linear':
                self.current_alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
            elif self.schedule == 'cosine':
                # Cosine annealing (più lento all'inizio)
                self.current_alpha = self.alpha_start + (1 - math.cos(progress * math.pi / 2)) * (self.alpha_end - self.alpha_start)
            elif self.schedule == 'exp':
                # Esponenziale (più veloce all'inizio)
                self.current_alpha = self.alpha_start + (1 - math.exp(-3 * progress)) * (self.alpha_end - self.alpha_start)
        
        return self.current_alpha
    
    def get_alpha(self) -> float:
        return self.current_alpha


# =============================================================================
# SCALE COMPENSATION MODULE
# =============================================================================

class ScaleCompensation(nn.Module):
    """
    Modulo che compensa automaticamente la riduzione causata dal soft weighting.
    
    Impara un fattore di scala che bilancia la riduzione media.
    Inizializzato a 1.0, può crescere per compensare.
    """
    def __init__(self, init_scale: float = 1.0, learnable: bool = True):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale)),
            requires_grad=learnable
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale.clamp(-2, 2))  # Range [0.13, 7.4]
        return x * scale
    
    def get_scale(self) -> float:
        with torch.no_grad():
            return torch.exp(self.log_scale.clamp(-2, 2)).item()


# =============================================================================
# FUSION LOSS - Loss per Joint Training
# =============================================================================

class FusionLoss(nn.Module):
    """
    Loss per Stage 3 Fusion che supporta soft weighting progressivo.
    
    Componenti:
    1. COUNT LOSS: L1 sul count finale (dopo weighting)
    2. SPATIAL BONUS: Premia allineamento tra π alto e density alta
    3. REGULARIZATION: Penalizza scale compensation troppo estremo
    
    Formula count:
      weighted_density = density × scale × ((1-α) + α×π)
      pred_count = Σ weighted_density / cell_area
    """
    
    def __init__(
        self,
        count_weight: float = 1.0,
        spatial_weight: float = 0.1,
        scale_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_reg_weight = scale_reg_weight
    
    def forward(
        self,
        raw_density: torch.Tensor,
        pi_probs: torch.Tensor,
        points_list: list,
        cell_area: float,
        alpha: float,
        scale_factor: float = 1.0,
    ) -> dict:
        """
        Args:
            raw_density: [B, 1, H, W] density da P2R
            pi_probs: [B, 1, H', W'] probabilità π da ZIP
            points_list: Lista di tensori [N_i, 2]
            cell_area: Area per normalizzazione
            alpha: Peso corrente del soft weighting (da scheduler)
            scale_factor: Fattore di compensazione scala
        """
        B = raw_density.shape[0]
        device = raw_density.device
        
        # Allinea π alla dimensione della density
        if pi_probs.shape[-2:] != raw_density.shape[-2:]:
            pi_aligned = F.interpolate(
                pi_probs,
                size=raw_density.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            pi_aligned = pi_probs
        
        # Calcola soft weights: (1 - α) + α × π
        soft_weights = (1 - alpha) + alpha * pi_aligned
        
        # Weighted density con scale compensation
        weighted_density = raw_density * scale_factor * soft_weights
        
        # =====================================================================
        # 1. COUNT LOSS (principale)
        # =====================================================================
        count_losses = []
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = weighted_density[i].sum() / cell_area
            
            gt_counts.append(gt)
            pred_counts.append(pred_count.item())
            
            count_losses.append(torch.abs(pred_count - gt))
        
        count_loss = torch.stack(count_losses).mean()
        
        # =====================================================================
        # 2. SPATIAL ALIGNMENT BONUS
        # Premia quando density alta corrisponde a π alto
        # =====================================================================
        if self.spatial_weight > 0 and alpha > 0:
            # Normalizza entrambe per confronto forma
            density_norm = raw_density / (raw_density.sum(dim=[2,3], keepdim=True) + 1e-8)
            pi_norm = pi_aligned / (pi_aligned.sum(dim=[2,3], keepdim=True) + 1e-8)
            
            # Correlation loss: penalizza differenze nella distribuzione
            spatial_loss = F.mse_loss(density_norm, pi_norm)
        else:
            spatial_loss = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # 3. SCALE REGULARIZATION
        # Penalizza scale troppo lontano da 1
        # =====================================================================
        scale_reg = self.scale_reg_weight * (scale_factor - 1.0) ** 2
        
        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total_loss = (
            self.count_weight * count_loss +
            self.spatial_weight * spatial_loss +
            scale_reg
        )
        
        # Metriche
        mae = np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)])
        bias = sum(pred_counts) / sum(gt_counts) if sum(gt_counts) > 0 else 0
        
        # Anche raw MAE per confronto
        raw_pred_counts = [(raw_density[i].sum() / cell_area).item() for i in range(B)]
        mae_raw = np.mean([abs(p - g) for p, g in zip(raw_pred_counts, gt_counts)])
        
        return {
            'total_loss': total_loss,
            'count_loss': count_loss,
            'spatial_loss': spatial_loss if isinstance(spatial_loss, float) else spatial_loss.item(),
            'scale_reg': scale_reg if isinstance(scale_reg, float) else scale_reg.item(),
            'mae_weighted': mae,
            'mae_raw': mae_raw,
            'bias': bias,
            'alpha': alpha,
            'scale': scale_factor,
            'avg_weight': soft_weights.mean().item(),
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model,
    scale_comp,
    criterion,
    dataloader,
    optimizer,
    device,
    default_down,
    epoch,
    alpha_scheduler,
):
    """Training con α progressivo."""
    model.train()
    scale_comp.train()
    
    # Congela BN del backbone
    model.backbone.eval()
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    
    # α per questa epoca
    alpha = alpha_scheduler.step(epoch)
    
    total_loss = 0.0
    metrics_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Stage3 Fusion [Ep {epoch}] α={alpha:.3f}")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        raw_density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']
        
        # Canonicalize
        _, _, H_in, W_in = images.shape
        raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        # Scale compensation - applica PRIMA della loss come tensore
        scaled_density = scale_comp(raw_density)
        
        # Loss - usa scaled_density invece di raw_density
        loss_dict = criterion(
            scaled_density, pi_probs, points_list, cell_area,
            alpha=alpha, scale_factor=1.0  # scale già applicato
        )
        loss = loss_dict['total_loss']
        
        # Aggiungi scale alle metriche
        loss_dict['scale'] = scale_comp.get_scale()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(scale_comp.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        pbar.set_postfix({
            'L': f"{loss.item():.3f}",
            'MAE_w': f"{loss_dict['mae_weighted']:.1f}",
            'MAE_r': f"{loss_dict['mae_raw']:.1f}",
            'scale': f"{loss_dict['scale']:.3f}",
        })
    
    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / n
    
    return avg_metrics


@torch.no_grad()
def validate(model, scale_comp, dataloader, device, default_down, alpha):
    """Validazione con soft weighting."""
    model.eval()
    scale_comp.eval()
    
    results = {
        'raw': {'mae': [], 'pred': 0, 'gt': 0},
        'weighted': {'mae': [], 'pred': 0, 'gt': 0},
    }
    
    scale = scale_comp.get_scale()
    
    for images, _, points in tqdm(dataloader, desc="Validate", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images)
        raw_density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']
        
        _, _, H_in, W_in = images.shape
        raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        # Applica scale
        scaled_density = raw_density * scale
        
        # Align π
        if pi_probs.shape[-2:] != scaled_density.shape[-2:]:
            pi_aligned = F.interpolate(pi_probs, size=scaled_density.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pi_aligned = pi_probs
        
        # Soft weights
        soft_weights = (1 - alpha) + alpha * pi_aligned
        weighted_density = scaled_density * soft_weights
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            
            # Raw (con scale applicato)
            pred_raw = (scaled_density[i].sum() / cell_area).item()
            results['raw']['mae'].append(abs(pred_raw - gt))
            results['raw']['pred'] += pred_raw
            results['raw']['gt'] += gt
            
            # Weighted (con scale applicato)
            pred_w = (weighted_density[i].sum() / cell_area).item()
            results['weighted']['mae'].append(abs(pred_w - gt))
            results['weighted']['pred'] += pred_w
            results['weighted']['gt'] += gt
    
    return {
        'mae_raw': np.mean(results['raw']['mae']),
        'mae_weighted': np.mean(results['weighted']['mae']),
        'bias_raw': results['raw']['pred'] / results['raw']['gt'] if results['raw']['gt'] > 0 else 0,
        'bias_weighted': results['weighted']['pred'] / results['weighted']['gt'] if results['weighted']['gt'] > 0 else 0,
        'scale': scale,
        'alpha': alpha,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, scale_comp, optimizer, scheduler, epoch, results, best_mae, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'scale_comp': scale_comp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'results': results,
        'best_mae': best_mae,
    }
    
    torch.save(checkpoint, os.path.join(output_dir, 'stage3_fusion_last.pth'))
    
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'stage3_fusion_best.pth'))
        print(f"💾 Best: MAE_w={results['mae_weighted']:.2f}, MAE_r={results['mae_raw']:.2f}, scale={results['scale']:.3f}")


def load_stage2_checkpoint(model, output_dir, device):
    """Carica Stage 2."""
    for name in ['stage2_bypass_best.pth', 'stage2_best.pth', 'best_model.pth']:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            print(f"✅ Caricamento Stage 2: {path}")
            state = torch.load(path, map_location=device, weights_only=False)
            if 'model' in state:
                state = state['model']
            model.load_state_dict(state, strict=False)
            return True
    print("⚠️ Stage 2 non trovato")
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage 3 FUSION - Joint Training con Soft Weighting')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--alpha-start', type=float, default=0.0,
                        help='α iniziale (default: 0.0 = raw density)')
    parser.add_argument('--alpha-end', type=float, default=0.3,
                        help='α finale (default: 0.3)')
    parser.add_argument('--alpha-warmup', type=int, default=50,
                        help='Epoche per warmup di α (default: 50)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Numero totale epoche (default: 300)')
    parser.add_argument('--lr-p2r', type=float, default=1e-5,
                        help='LR per P2R head (default: 1e-5)')
    parser.add_argument('--lr-scale', type=float, default=1e-3,
                        help='LR per scale compensation (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=80,
                        help='Early stopping patience (default: 80)')
    parser.add_argument('--spatial-weight', type=float, default=0.05,
                        help='Peso spatial alignment loss (default: 0.05)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('DEVICE', 'cuda'))
    init_seeds(config.get('SEED', 42))
    
    print("="*70)
    print("🚀 STAGE 3 FUSION - Joint Training con Soft Weighting")
    print("="*70)
    print(f"   α warmup: {args.alpha_start} → {args.alpha_end} in {args.alpha_warmup} epoche")
    print(f"   LR P2R: {args.lr_p2r}, LR scale: {args.lr_scale}")
    print(f"   Spatial weight: {args.spatial_weight}")
    print("="*70)
    
    # Dataset config
    dataset_section = config.get('DATASET', 'shha')
    data_section = config.get('DATA')
    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = {}
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
        data_cfg.update(dataset_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}

    normalized = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = ALIAS_DATASETS.get(normalized, str(dataset_name_raw).lower())

    if 'ROOT' not in data_cfg:
        raise KeyError("Dataset ROOT path missing.")
    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', 16)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(dataset_name)
    train_ds = DatasetClass(root=data_cfg['ROOT'], split=data_cfg.get('TRAIN_SPLIT', 'train'),
                            block_size=block_size, transforms=train_tf)
    val_ds = DatasetClass(root=data_cfg['ROOT'], split=data_cfg.get('VAL_SPLIT', 'val'),
                          block_size=block_size, transforms=val_tf)
    
    batch_size = config.get('OPTIM_JOINT', {}).get('BATCH_SIZE', 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                              drop_last=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"BINS_CONFIG missing for '{dataset_name}'")
    
    zip_head_cfg = config.get('ZIP_HEAD', {})
    model_cfg = config.get('MODEL', {})
    
    model = P2R_ZIP_Model(
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Scale compensation module
    scale_comp = ScaleCompensation(init_scale=1.0, learnable=True).to(device)
    
    # Output dir
    run_name = config.get('RUN_NAME', 'shha_v15')
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica Stage 2
    load_stage2_checkpoint(model, output_dir, device)
    
    # =========================================================================
    # FREEZE STRATEGY
    # =========================================================================
    
    # Backbone: FROZEN
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("❄️ Backbone: FROZEN")
    
    # ZIP head: quasi frozen (LR molto basso opzionale)
    for param in model.zip_head.parameters():
        param.requires_grad = False  # Completamente frozen per ora
    print("❄️ ZIP head: FROZEN")
    
    # P2R head: trainabile
    p2r_params = list(model.p2r_head.parameters())
    print(f"🔥 P2R head: {sum(p.numel() for p in p2r_params):,} parametri")
    
    # Scale compensation: trainabile
    print(f"🔥 Scale comp: {sum(p.numel() for p in scale_comp.parameters()):,} parametri")
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': p2r_params, 'lr': args.lr_p2r},
        {'params': scale_comp.parameters(), 'lr': args.lr_scale},
    ], weight_decay=1e-4)
    
    # Scheduler
    epochs = args.epochs
    
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Alpha scheduler
    alpha_scheduler = AlphaScheduler(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        warmup_epochs=args.alpha_warmup,
        schedule='cosine'
    )
    
    # Loss
    criterion = FusionLoss(
        count_weight=1.0,
        spatial_weight=args.spatial_weight,
        scale_reg_weight=0.01,
    )
    
    # =========================================================================
    # VALIDAZIONE INIZIALE
    # =========================================================================
    
    print("\n📋 Validazione iniziale (α=0, raw density):")
    val_results = validate(model, scale_comp, val_loader, device, default_down, alpha=0.0)
    initial_mae = val_results['mae_raw']
    print(f"   MAE raw: {initial_mae:.2f}")
    print(f"   Bias: {val_results['bias_raw']:.3f}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    print(f"\n🚀 Training: 1 → {epochs}")
    print(f"   α warmup: 0 → {args.alpha_end} in {args.alpha_warmup} epoche")
    
    best_mae = initial_mae
    no_improve = 0
    val_interval = 5
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, scale_comp, criterion, train_loader, optimizer,
            device, default_down, epoch, alpha_scheduler
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0:
            current_alpha = alpha_scheduler.get_alpha()
            val_results = validate(model, scale_comp, val_loader, device, default_down, alpha=current_alpha)
            
            # Usa MAE weighted come metrica quando α > 0.1, altrimenti raw
            if current_alpha > 0.1:
                mae_metric = val_results['mae_weighted']
                metric_name = "weighted"
            else:
                mae_metric = val_results['mae_raw']
                metric_name = "raw"
            
            improved = mae_metric < best_mae
            
            print(f"\nEpoch {epoch}:")
            print(f"   α = {current_alpha:.3f}, scale = {val_results['scale']:.3f}")
            print(f"   MAE raw: {val_results['mae_raw']:.2f}, MAE weighted: {val_results['mae_weighted']:.2f}")
            print(f"   Bias raw: {val_results['bias_raw']:.3f}, Bias weighted: {val_results['bias_weighted']:.3f}")
            print(f"   Best ({metric_name}): {best_mae:.2f} {'✅ NEW!' if improved else ''}")
            
            if improved:
                best_mae = mae_metric
                no_improve = 0
                save_checkpoint(model, scale_comp, optimizer, scheduler, epoch, val_results, best_mae, output_dir, is_best=True)
            else:
                no_improve += val_interval
                save_checkpoint(model, scale_comp, optimizer, scheduler, epoch, val_results, best_mae, output_dir, is_best=False)
            
            if no_improve >= args.patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # =========================================================================
    # RISULTATI FINALI
    # =========================================================================
    
    print("\n" + "="*70)
    print("🏁 STAGE 3 FUSION COMPLETATO")
    print("="*70)
    print(f"   MAE iniziale (raw): {initial_mae:.2f}")
    print(f"   MAE finale:         {best_mae:.2f}")
    print(f"   α finale:           {alpha_scheduler.get_alpha():.3f}")
    print(f"   Scale finale:       {scale_comp.get_scale():.3f}")
    print(f"   Miglioramento:      {initial_mae - best_mae:+.2f}")
    
    if best_mae < initial_mae:
        print(f"\n   ✅ FUSION RIUSCITA!")
    else:
        print(f"\n   ⚠️ Nessun miglioramento con fusion")
    
    print("="*70)


if __name__ == '__main__':
    main()