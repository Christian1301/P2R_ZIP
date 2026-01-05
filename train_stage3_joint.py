#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V2 - JOINT TRAINING con Soft Weighting

CAMBIAMENTO CHIAVE:
Invece di hard masking (density × mask), usa soft weighting:
  count = sum(density × (1 - α + α × π))

Dove:
- α = 0.0: ignora completamente π (usa raw density) 
- α = 1.0: hard masking con π
- α = 0.2-0.3: combina 70-80% raw + 20-30% π-weighted

VANTAGGI:
1. Non perde informazione anche se π-head è imperfetto
2. Gradiente più stabile (no discontinuità)
3. Permette al modello di imparare quando fidarsi di π

FORMULA LOSS:
L_total = (1-α_loss)·L_ZIP + α_loss·L_P2R + β·L_consistency

Dove L_consistency penalizza discrepanze tra π alto e density bassa.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds, 
    collate_fn,
    canonicalize_p2r_grid,
)


ALIAS_DATASETS = {
    'shha': 'shha',
    'shanghaitecha': 'shha',
    'shanghaitechparta': 'shha',
    'shanghaitechaparta': 'shha',
    'shhb': 'shhb',
    'shanghaitechpartb': 'shhb',
    'ucf': 'ucf',
    'ucfqnrf': 'ucf',
    'nwpu': 'nwpu',
    'jhu': 'jhu'
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
# SOFT WEIGHTING UTILITIES
# =============================================================================

def apply_soft_weighting(density, pi_probs, alpha=0.2):
    """
    Applica soft weighting alla density map.
    
    Formula: density_weighted = density × (1 - α + α × π)
    
    Args:
        density: [B, 1, H, W] raw density
        pi_probs: [B, 1, H', W'] probabilità π (verrà ridimensionata se necessario)
        alpha: peso del soft weighting (0=ignora π, 1=hard masking)
    
    Returns:
        density_weighted: [B, 1, H, W]
    """
    # Ridimensiona π se necessario
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs,
            size=density.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Soft weighting: preserva (1-α) della density anche dove π=0
    weights = (1 - alpha) + alpha * pi_probs
    
    return density * weights


def compute_count_with_soft_weighting(density, pi_probs, cell_area, alpha=0.2):
    """
    Calcola count usando soft weighting.
    
    Returns:
        count: scalar tensor
        coverage: % di π > 0.5
    """
    weighted_density = apply_soft_weighting(density, pi_probs, alpha)
    count = weighted_density.sum() / cell_area
    
    # Coverage (per monitoring)
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
        )
    coverage = (pi_probs > 0.5).float().mean().item() * 100
    
    return count, coverage


# =============================================================================
# LOSS COMPONENTS
# =============================================================================

class PiHeadLoss(nn.Module):
    """BCE Loss per π-head."""
    def __init__(self, pos_weight: float = 3.0, block_size: int = 16, threshold: float = 0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def forward(self, logit_pi, gt_density):
        logit_pieno = logit_pi[:, 1:2, :, :]
        
        gt_counts = F.avg_pool2d(
            gt_density, kernel_size=self.block_size, stride=self.block_size
        ) * (self.block_size ** 2)
        gt_occupancy = (gt_counts > self.threshold).float()
        
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest')
        
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        
        return self.bce(logit_pieno, gt_occupancy)


class P2RCountLoss(nn.Module):
    """Count + Spatial loss per P2R."""
    def __init__(self, count_weight: float = 2.5, spatial_weight: float = 0.1):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
    
    def forward(self, pred_density, points_list, cell_area):
        B, _, H, W = pred_density.shape
        device = pred_density.device
        
        total_count_loss = torch.tensor(0.0, device=device)
        total_spatial_loss = torch.tensor(0.0, device=device)
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = pred_density[i].sum() / cell_area
            
            total_count_loss += torch.abs(pred_count - gt)
            
            if gt > 0 and self.spatial_weight > 0:
                target = torch.zeros(H, W, device=device)
                for pt in pts:
                    x = int(pt[0].clamp(0, W-1).item())
                    y = int(pt[1].clamp(0, H-1).item())
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                target[ny, nx] += 1.0
                
                if target.sum() > 0:
                    target = target / target.sum()
                    pred_norm = pred_density[i, 0] / (pred_density[i, 0].sum() + 1e-8)
                    total_spatial_loss += F.mse_loss(pred_norm, target)
        
        return self.count_weight * total_count_loss / B + self.spatial_weight * total_spatial_loss / B


class ConsistencyLoss(nn.Module):
    """
    Loss di consistenza tra π e density.
    
    Penalizza i casi dove:
    - π è alto ma density è bassa
    - π è basso ma density è alta
    
    Questo forza coerenza tra i due head.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pi_probs, density, cell_area):
        """
        Args:
            pi_probs: [B, 1, H_pi, W_pi] probabilità π
            density: [B, 1, H_d, W_d] density map
            cell_area: scaling factor
        """
        if self.weight == 0:
            return torch.tensor(0.0, device=density.device)
        
        # Allinea dimensioni
        if pi_probs.shape[-2:] != density.shape[-2:]:
            pi_probs = F.interpolate(
                pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Normalizza density per confronto
        density_norm = density / (density.max() + 1e-8)
        
        # Penalizza discrepanze: |π - density_norm|² pesato
        # Ma solo dove almeno uno dei due è significativo
        mask = (pi_probs > 0.3) | (density_norm > 0.3)
        
        if mask.sum() > 0:
            diff = (pi_probs - density_norm) ** 2
            loss = (diff * mask.float()).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=density.device)
        
        return self.weight * loss


# =============================================================================
# JOINT LOSS V2 - Con Soft Weighting
# =============================================================================

class JointLossV2(nn.Module):
    """
    Joint Loss con soft weighting.
    
    Formula: L_total = (1-α)·L_ZIP + α·L_P2R + β·L_consistency
    
    Il count viene calcolato con soft weighting:
      count = sum(density × (1 - sw_α + sw_α × π))
    """
    def __init__(
        self,
        alpha: float = 0.7,           # Peso P2R vs ZIP
        soft_weight_alpha: float = 0.2,  # Alpha per soft weighting
        pi_pos_weight: float = 3.0,
        block_size: int = 16,
        count_weight: float = 2.5,
        spatial_weight: float = 0.1,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.soft_weight_alpha = soft_weight_alpha
        
        self.zip_loss = PiHeadLoss(
            pos_weight=pi_pos_weight,
            block_size=block_size,
        )
        
        self.p2r_loss = P2RCountLoss(
            count_weight=count_weight,
            spatial_weight=spatial_weight,
        )
        
        self.consistency_loss = ConsistencyLoss(weight=consistency_weight)
    
    def forward(self, outputs, gt_density, points_list, cell_area):
        """
        Args:
            outputs: dict con 'logit_pi_maps', 'p2r_density', 'pi_probs'
            gt_density: [B, 1, H, W]
            points_list: lista tensori [N_i, 2]
            cell_area: float
        """
        pi_probs = outputs['pi_probs']
        raw_density = outputs['p2r_density']
        
        # Applica soft weighting per il count
        weighted_density = apply_soft_weighting(
            raw_density, pi_probs, alpha=self.soft_weight_alpha
        )
        
        # L_ZIP
        l_zip = self.zip_loss(outputs['logit_pi_maps'], gt_density)
        
        # L_P2R (sulla density pesata)
        l_p2r = self.p2r_loss(weighted_density, points_list, cell_area)
        
        # L_consistency
        l_cons = self.consistency_loss(pi_probs, raw_density, cell_area)
        
        # Total
        total_loss = (1 - self.alpha) * l_zip + self.alpha * l_p2r + l_cons
        
        # Metriche
        with torch.no_grad():
            B = raw_density.shape[0]
            
            # Count con soft weighting
            pred_counts_sw = []
            pred_counts_raw = []
            gt_counts = []
            
            for i, pts in enumerate(points_list):
                gt = len(pts)
                gt_counts.append(gt)
                
                # Raw count
                raw_count = (raw_density[i].sum() / cell_area).item()
                pred_counts_raw.append(raw_count)
                
                # Soft weighted count
                sw_count = (weighted_density[i].sum() / cell_area).item()
                pred_counts_sw.append(sw_count)
            
            mae_raw = np.mean([abs(p - g) for p, g in zip(pred_counts_raw, gt_counts)])
            mae_sw = np.mean([abs(p - g) for p, g in zip(pred_counts_sw, gt_counts)])
            
            # Coverage
            if pi_probs.shape[-2:] != raw_density.shape[-2:]:
                pi_for_cov = F.interpolate(pi_probs, size=raw_density.shape[-2:], mode='bilinear', align_corners=False)
            else:
                pi_for_cov = pi_probs
            coverage = (pi_for_cov > 0.5).float().mean().item() * 100
        
        metrics = {
            'total': total_loss.item(),
            'zip': l_zip.item(),
            'p2r': l_p2r.item(),
            'consistency': l_cons.item() if isinstance(l_cons, torch.Tensor) else l_cons,
            'mae_raw': mae_raw,
            'mae_soft_weighted': mae_sw,
            'coverage': coverage,
        }
        
        return total_loss, metrics


# =============================================================================
# FORWARD PASS HELPER
# =============================================================================

def forward_pass(model, images):
    """Forward pass standard."""
    return model(images)


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================

def train_one_epoch(
    model, criterion, dataloader, optimizer, scheduler,
    device, default_down, epoch, config
):
    """Training con soft weighting."""
    model.train()
    
    if config.get('FREEZE_BN', True):
        model.backbone.eval()
    
    total_loss = 0.0
    metrics_accum = {}
    
    pbar = tqdm(dataloader, desc=f"Stage3 V2 [Ep {epoch}]")
    
    for images, gt_density, points in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        
        outputs = forward_pass(model, images)
        
        # Canonicalize
        pred_density = outputs['p2r_density']
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(pred_density, (h_in, w_in), default_down)
        outputs['p2r_density'] = pred_density
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Loss
        loss, metrics = criterion(outputs, gt_density, points_list, cell_area)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
        
        pbar.set_postfix({
            'L': f"{loss.item():.3f}",
            'MAE_sw': f"{metrics['mae_soft_weighted']:.1f}",
            'MAE_raw': f"{metrics['mae_raw']:.1f}",
            'cov': f"{metrics['coverage']:.1f}%",
        })
    
    if scheduler:
        scheduler.step()
    
    n = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n
    
    return total_loss / n, metrics_accum


@torch.no_grad()
def validate(model, dataloader, device, default_down, soft_weight_alpha=0.2):
    """
    Validazione con entrambe le metriche: raw e soft-weighted.
    """
    model.eval()
    
    results_raw = {'mae': [], 'mse': [], 'pred': 0, 'gt': 0}
    results_sw = {'mae': [], 'mse': [], 'pred': 0, 'gt': 0}
    coverages = []
    
    for images, densities, points in tqdm(dataloader, desc="Validate", leave=False):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = forward_pass(model, images)
        raw_density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']
        
        _, _, H_in, W_in = images.shape
        raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Soft weighted
        weighted_density = apply_soft_weighting(raw_density, pi_probs, alpha=soft_weight_alpha)
        
        # Coverage
        if pi_probs.shape[-2:] != raw_density.shape[-2:]:
            pi_resized = F.interpolate(pi_probs, size=raw_density.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pi_resized = pi_probs
        coverages.append((pi_resized > 0.5).float().mean().item() * 100)
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            
            # Raw
            pred_raw = (raw_density[i].sum() / cell_area).item()
            results_raw['mae'].append(abs(pred_raw - gt))
            results_raw['mse'].append((pred_raw - gt) ** 2)
            results_raw['pred'] += pred_raw
            results_raw['gt'] += gt
            
            # Soft weighted
            pred_sw = (weighted_density[i].sum() / cell_area).item()
            results_sw['mae'].append(abs(pred_sw - gt))
            results_sw['mse'].append((pred_sw - gt) ** 2)
            results_sw['pred'] += pred_sw
            results_sw['gt'] += gt
    
    return {
        'raw': {
            'mae': np.mean(results_raw['mae']),
            'rmse': np.sqrt(np.mean(results_raw['mse'])),
            'bias': results_raw['pred'] / results_raw['gt'] if results_raw['gt'] > 0 else 0,
        },
        'soft_weighted': {
            'mae': np.mean(results_sw['mae']),
            'rmse': np.sqrt(np.mean(results_sw['mse'])),
            'bias': results_sw['pred'] / results_sw['gt'] if results_sw['gt'] > 0 else 0,
        },
        'coverage': np.mean(coverages),
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, results, best_mae, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'results': results,
        'best_mae': best_mae,
    }
    
    torch.save(checkpoint, os.path.join(output_dir, 'stage3_latest.pth'))
    
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'stage3_best.pth'))
        print(f"💾 Best: MAE_raw={results['raw']['mae']:.2f}, MAE_sw={results['soft_weighted']['mae']:.2f}")


def load_stage2_checkpoint(model, output_dir, device):
    """Carica Stage 2."""
    for name in ['stage2_best.pth', 'best_model.pth']:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            print(f"✅ Caricamento Stage 2: {path}")
            state = torch.load(path, map_location=device)
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
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('DEVICE', 'cuda'))
    init_seeds(config.get('SEED', 42))
    
    print("="*60)
    print("🚀 Stage 3 V2 - JOINT TRAINING con Soft Weighting")
    print("="*60)
    
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

    if not data_cfg:
        raise KeyError("DATA or DATASET configuration missing for Stage 3.")
    if 'ROOT' not in data_cfg:
        raise KeyError("Dataset ROOT path missing in configuration.")
    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    optim_cfg = config.get('OPTIM_JOINT', {})
    joint_cfg = config.get('JOINT_LOSS', {})
    
    alpha = float(joint_cfg.get('ALPHA', 0.7))
    soft_weight_alpha = float(joint_cfg.get('SOFT_WEIGHT_ALPHA', 0.2))
    epochs = optim_cfg.get('EPOCHS', 1000)
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', 200)
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', 16)
    
    print(f"Device: {device}")
    print(f"Loss α: {alpha} (ZIP weight: {1-alpha:.2f}, P2R weight: {alpha:.2f})")
    print(f"Soft Weighting α: {soft_weight_alpha}")
    print(f"   → {(1-soft_weight_alpha)*100:.0f}% raw + {soft_weight_alpha*100:.0f}% π-weighted")
    print("="*60)
    
    # Dataset
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(dataset_name)
    train_ds = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('TRAIN_SPLIT', 'train'),
        block_size=block_size,
        transforms=train_tf
    )
    val_ds = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('VAL_SPLIT', 'val'),
        block_size=block_size,
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=optim_cfg.get('BATCH_SIZE', 6),
        shuffle=True,
        num_workers=optim_cfg.get('NUM_WORKERS', 4),
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg.get('VAL_NUM_WORKERS', optim_cfg.get('NUM_WORKERS', 4)),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"BINS_CONFIG missing definition for dataset '{dataset_name}' and no default is available.")
    zip_head_cfg = config.get('ZIP_HEAD', {})
    model_cfg = config.get('MODEL', {})
    
    model = P2R_ZIP_Model(
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        use_ste_mask=model_cfg.get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Output dir
    run_name = config.get('RUN_NAME', 'shha_v11')
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica Stage 2
    load_stage2_checkpoint(model, output_dir, device)
    
    # Optimizer con LR differenziati
    param_groups = []
    
    # Backbone (quasi frozen)
    lr_backbone = float(optim_cfg.get('LR_BACKBONE', 5e-7))
    if lr_backbone > 0:
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': lr_backbone,
        })
        print(f"Backbone LR: {lr_backbone}")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone: FROZEN")
    
    # ZIP head
    lr_zip = float(optim_cfg.get('LR_ZIP_HEAD', 2e-5))
    param_groups.append({
        'params': model.zip_head.parameters(),
        'lr': lr_zip,
    })
    print(f"ZIP head LR: {lr_zip}")
    
    # P2R head
    lr_p2r = float(optim_cfg.get('LR_P2R_HEAD', 3e-5))
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n]
    param_groups.append({
        'params': p2r_params,
        'lr': lr_p2r,
    })
    print(f"P2R head LR: {lr_p2r}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(optim_cfg.get('WEIGHT_DECAY', 1e-4))
    )
    
    # Scheduler
    warmup_epochs = optim_cfg.get('WARMUP_EPOCHS', 20)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss
    criterion = JointLossV2(
        alpha=alpha,
        soft_weight_alpha=soft_weight_alpha,
        pi_pos_weight=float(joint_cfg.get('PI_POS_WEIGHT', 3.0)),
        block_size=block_size,
        count_weight=float(joint_cfg.get('COUNT_WEIGHT', 2.5)),
        spatial_weight=float(joint_cfg.get('SPATIAL_WEIGHT', 0.1)),
        consistency_weight=float(joint_cfg.get('CONSISTENCY_WEIGHT', 0.1)),
    ).to(device)
    
    # Resume
    start_epoch = 1
    best_mae = float('inf')
    no_improve = 0
    
    resume_path = os.path.join(output_dir, 'stage3_latest.pth')
    if os.path.isfile(resume_path):
        print(f"\n🔄 Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        print(f"   Epoch {ckpt['epoch']}, Best MAE: {best_mae:.2f}")
    
    # Validazione iniziale
    print("\n📋 Validazione iniziale:")
    val_results = validate(model, val_loader, device, default_down, soft_weight_alpha)
    print(f"   RAW:  MAE={val_results['raw']['mae']:.2f}, Bias={val_results['raw']['bias']:.3f}")
    print(f"   SW:   MAE={val_results['soft_weighted']['mae']:.2f}, Bias={val_results['soft_weighted']['bias']:.3f}")
    print(f"   Coverage: {val_results['coverage']:.1f}%")
    
    # Training
    print(f"\n🚀 Training: {start_epoch} → {epochs}")
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            device, default_down, epoch, optim_cfg
        )
        
        # Validate
        if epoch % val_interval == 0:
            val_results = validate(model, val_loader, device, default_down, soft_weight_alpha)
            
            # Usa MAE raw come metrica principale (più affidabile)
            mae_raw = val_results['raw']['mae']
            mae_sw = val_results['soft_weighted']['mae']
            
            improved = mae_raw < best_mae
            
            print(f"\nEpoch {epoch}:")
            print(f"   RAW:  MAE={mae_raw:.2f}, Bias={val_results['raw']['bias']:.3f}")
            print(f"   SW:   MAE={mae_sw:.2f}, Bias={val_results['soft_weighted']['bias']:.3f}")
            print(f"   Coverage: {val_results['coverage']:.1f}%")
            print(f"   Best: {best_mae:.2f} {'✅ NEW!' if improved else ''}")
            
            if improved:
                best_mae = mae_raw
                no_improve = 0
                save_checkpoint(model, optimizer, scheduler, epoch, val_results, best_mae, output_dir, is_best=True)
            else:
                no_improve += val_interval
                save_checkpoint(model, optimizer, scheduler, epoch, val_results, best_mae, output_dir, is_best=False)
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # Risultati finali
    print("\n" + "="*60)
    print("🏁 STAGE 3 V2 COMPLETATO")
    print("="*60)
    print(f"   Best MAE (raw): {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage3_best.pth")
    
    if best_mae < 70:
        print("   🎯 TARGET RAGGIUNTO! MAE < 70")
    elif best_mae < 75:
        print("   ✅ Buon risultato, vicino al target")
    
    print("="*60)


if __name__ == '__main__':
    main()