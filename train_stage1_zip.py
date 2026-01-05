#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 V4 - ZIP Pre-training con ZIP NLL Loss

CAMBIAMENTO PRINCIPALE DA V3:
- Usa ZIP Negative Log-Likelihood invece di BCE/Focal
- ZIP NLL modella direttamente la distribuzione Zero-Inflated Poisson
- Non richiede tuning di pos_weight per bilanciamento classi
- π e λ vengono appresi congiuntamente in modo naturale

TARGET:
- Recall > 90% (priorità alta - non perdere persone)
- Precision > 50% (accettabile - Stage 2 correggerà)
- F1 > 65%
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
import json
from typing import Tuple, Dict, List

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn
from losses.zip_nll import zip_nll  # Usa il tuo file zip_nll.py


# Default bin configurations for common datasets
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
# COMPUTE LOSS AND METRICS - Usa direttamente zip_nll
# =============================================================================

def compute_zip_loss_and_metrics(
    outputs: dict,
    gt_density: torch.Tensor,
    block_size: int = 16,
    weight_lambda_reg: float = 0.01,
    lambda_max_target: float = 8.0,
    pi_threshold: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Calcola ZIP NLL loss e metriche.
    
    Usa direttamente la funzione zip_nll dal tuo file losses/zip_nll.py
    """
    pi_probs = outputs['pi_probs']       # [B, 1, H, W]
    lambda_maps = outputs['lambda_maps']  # [B, 1, H, W]
    
    # GT counts per blocco
    gt_counts = F.avg_pool2d(
        gt_density,
        kernel_size=block_size,
        stride=block_size
    ) * (block_size ** 2)
    
    # 1. ZIP NLL Loss (usa la tua funzione - gestisce già l'allineamento dimensioni)
    loss_nll = zip_nll(pi_probs, lambda_maps, gt_counts, reduction='mean')
    
    # 2. Lambda regularization (opzionale)
    if weight_lambda_reg > 0:
        loss_lambda_reg = F.relu(lambda_maps - lambda_max_target).mean()
    else:
        loss_lambda_reg = torch.tensor(0.0, device=pi_probs.device)
    
    # Total loss
    total_loss = loss_nll + weight_lambda_reg * loss_lambda_reg
    
    # Metriche per monitoraggio
    with torch.no_grad():
        # Allinea gt_counts per metriche
        if gt_counts.shape[-2:] != pi_probs.shape[-2:]:
            gt_counts_aligned = F.interpolate(gt_counts, size=pi_probs.shape[-2:], mode='nearest')
        else:
            gt_counts_aligned = gt_counts
        
        gt_occupancy = (gt_counts_aligned > 0.5).float()
        pred_binary = (pi_probs > pi_threshold).float()
        
        tp = ((pred_binary == 1) & (gt_occupancy == 1)).sum().item()
        tn = ((pred_binary == 0) & (gt_occupancy == 0)).sum().item()
        fp = ((pred_binary == 1) & (gt_occupancy == 0)).sum().item()
        fn = ((pred_binary == 0) & (gt_occupancy == 1)).sum().item()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        pi_mean = pi_probs.mean().item()
        pi_std = pi_probs.std().item()
        lambda_mean = lambda_maps.mean().item()
        lambda_max = lambda_maps.max().item()
    
    metrics = {
        'total': total_loss.item(),
        'nll': loss_nll.item(),
        'lambda_reg': loss_lambda_reg.item() if torch.is_tensor(loss_lambda_reg) else loss_lambda_reg,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pi_mean': pi_mean,
        'pi_std': pi_std,
        'lambda_mean': lambda_mean,
        'lambda_max': lambda_max,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }
    
    return total_loss, metrics


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    block_size: int,
    loss_cfg: dict,
) -> dict:
    """Training di una epoch."""
    model.train()
    
    total_loss = 0.0
    metrics_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Stage1 V4 [Ep {epoch}]")
    
    for images, gt_density, _ in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss, metrics = compute_zip_loss_and_metrics(
            outputs, gt_density,
            block_size=block_size,
            weight_lambda_reg=loss_cfg.get('WEIGHT_LAMBDA_REG', 0.01),
            lambda_max_target=loss_cfg.get('LAMBDA_MAX_TARGET', 8.0),
            pi_threshold=loss_cfg.get('PI_THRESHOLD', 0.5),
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        pbar.set_postfix({
            'NLL': f"{metrics['nll']:.3f}",
            'P': f"{metrics['precision']*100:.1f}%",
            'R': f"{metrics['recall']*100:.1f}%",
            'F1': f"{metrics['f1']*100:.1f}%",
            'π': f"{metrics['pi_mean']:.2f}",
            'λ': f"{metrics['lambda_mean']:.1f}",
        })
    
    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / n
    
    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int,
    loss_cfg: dict,
) -> dict:
    """Validazione con metriche complete."""
    model.eval()
    
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    total_loss = 0.0
    total_nll = 0.0
    pi_means, pi_stds = [], []
    lambda_means = []
    
    pi_threshold = loss_cfg.get('PI_THRESHOLD', 0.5)
    
    for images, gt_density, _ in tqdm(dataloader, desc="Validate", leave=False):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        loss, batch_metrics = compute_zip_loss_and_metrics(
            outputs, gt_density,
            block_size=block_size,
            weight_lambda_reg=loss_cfg.get('WEIGHT_LAMBDA_REG', 0.01),
            lambda_max_target=loss_cfg.get('LAMBDA_MAX_TARGET', 8.0),
            pi_threshold=pi_threshold,
        )
        
        total_loss += loss.item()
        total_nll += batch_metrics['nll']
        
        pi_probs = outputs['pi_probs']
        lambda_maps = outputs['lambda_maps']
        
        pi_means.append(pi_probs.mean().item())
        pi_stds.append(pi_probs.std().item())
        lambda_means.append(lambda_maps.mean().item())
        
        total_tp += batch_metrics['tp']
        total_tn += batch_metrics['tn']
        total_fp += batch_metrics['fp']
        total_fn += batch_metrics['fn']
    
    n = len(dataloader)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'loss': total_loss / n,
        'nll': total_nll / n,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp, 'tn': total_tn, 'fp': total_fp, 'fn': total_fn,
        'pi_mean': np.mean(pi_means),
        'pi_std': np.mean(pi_stds),
        'lambda_mean': np.mean(lambda_means),
    }


# =============================================================================
# THRESHOLD FINDER - Trova τ ottimale per recall alto
# =============================================================================

@torch.no_grad()
def find_optimal_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int,
    target_recall: float = 0.90,
    thresholds: List[float] = None
) -> dict:
    """
    Trova il threshold che garantisce un certo recall target.
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    model.eval()
    
    threshold_stats = {t: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    
    for images, gt_density, _ in dataloader:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        pi_probs = outputs['pi_probs']
        
        # GT occupancy
        gt_counts = F.avg_pool2d(
            gt_density, kernel_size=block_size, stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts > 0.5).float()
        
        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=pi_probs.shape[-2:], mode='nearest'
            )
        
        for thresh in thresholds:
            pred_binary = (pi_probs > thresh).float()
            
            tp = ((pred_binary == 1) & (gt_occupancy == 1)).sum().item()
            tn = ((pred_binary == 0) & (gt_occupancy == 0)).sum().item()
            fp = ((pred_binary == 1) & (gt_occupancy == 0)).sum().item()
            fn = ((pred_binary == 0) & (gt_occupancy == 1)).sum().item()
            
            threshold_stats[thresh]['tp'] += tp
            threshold_stats[thresh]['tn'] += tn
            threshold_stats[thresh]['fp'] += fp
            threshold_stats[thresh]['fn'] += fn
    
    # Calcola metriche
    results = []
    best_for_target_recall = None
    
    for thresh in thresholds:
        stats = threshold_stats[thresh]
        tp, tn, fp, fn = stats['tp'], stats['tn'], stats['fp'], stats['fn']
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        result = {
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        results.append(result)
        
        # Trova il threshold più alto che garantisce recall >= target
        if recall >= target_recall:
            if best_for_target_recall is None or thresh > best_for_target_recall['threshold']:
                best_for_target_recall = result
    
    # Se nessun threshold raggiunge target recall, prendi quello con recall più alto
    if best_for_target_recall is None:
        best_for_target_recall = max(results, key=lambda x: x['recall'])
    
    # Trova anche il best per F1
    best_f1 = max(results, key=lambda x: x['f1'])
    
    return {
        'for_target_recall': best_for_target_recall,
        'for_best_f1': best_f1,
        'target_recall': target_recall,
        'all_results': results
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_f1, no_improve, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_f1': best_f1,
        'no_improve': no_improve,
    }
    
    torch.save(state, os.path.join(output_dir, 'last.pth'))
    
    if is_best:
        torch.save(state, os.path.join(output_dir, 'best_model.pth'))
        print(f"💾 Best: F1={metrics['f1']*100:.2f}%, P={metrics['precision']*100:.1f}%, R={metrics['recall']*100:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])
    
    print("="*70)
    print("🚀 STAGE 1 V4 - ZIP Pre-training con ZIP NLL Loss")
    print("="*70)
    print(f"Device: {device}")
    print(f"Target: Recall > 90%, F1 > 65%")
    print("="*70)
    
    # Config
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

    alias_map = {
        'shha': 'shha',
        'shanghaitecha': 'shha',
        'shanghaitechparta': 'shha',
        'shanghaitechaparta': 'shha',
        'shhb': 'shhb',
        'shanghaitechpartb': 'shhb',
        'shanghaitechpartbb': 'shhb',
        'ucf': 'ucf',
        'ucfqnrf': 'ucf',
        'nwpu': 'nwpu',
        'jhu': 'jhu'
    }
    normalized_name = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = alias_map.get(normalized_name, str(dataset_name_raw).lower())

    if not data_cfg:
        raise KeyError("Dataset configuration missing. Provide DATA or DATASET fields with ROOT/ splits.")

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]
    if 'ROOT' not in data_cfg:
        raise KeyError("Dataset ROOT path missing in DATA / DATASET configuration.")

    optim_cfg = config['OPTIM_ZIP']
    loss_cfg = config.get('ZIP_LOSS_V4', {})
    
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))
    
    # Dataset
    DatasetClass = get_dataset(dataset_name)
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
        batch_size=optim_cfg['BATCH_SIZE'],
        shuffle=True,
        num_workers=optim_cfg.get('NUM_WORKERS', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg.get('VAL_NUM_WORKERS', optim_cfg.get('NUM_WORKERS', 4)),
        collate_fn=collate_fn
    )
    
    print(f"\n📊 Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
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
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        use_ste_mask=model_cfg.get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        }
    ).to(device)
    
    print(f"\n⚙️ Loss Config (ZIP NLL V4):")
    print(f"   Weight λ reg: {loss_cfg.get('WEIGHT_LAMBDA_REG', 0.01)}")
    print(f"   λ max target: {loss_cfg.get('LAMBDA_MAX_TARGET', 8.0)}")
    print(f"   π threshold:  {loss_cfg.get('PI_THRESHOLD', 0.5)}")
    
    # Optimizer
    base_lr = float(optim_cfg.get('LR', 1e-4))
    backbone_lr = float(optim_cfg.get('LR_BACKBONE', base_lr * 0.1))
    head_lr = float(optim_cfg.get('HEAD_LR', base_lr))

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.zip_head.parameters(), 'lr': head_lr},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(optim_cfg['WEIGHT_DECAY']))
    
    # Scheduler
    epochs = optim_cfg['EPOCHS']
    warmup = optim_cfg.get('WARMUP_EPOCHS', 50)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup])
    
    # Output
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), config.get('RUN_NAME', 'stage1_run'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Resume
    best_f1 = 0.0
    start_epoch = 1
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', 800)
    no_improve = 0
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    
    resume_path = os.path.join(output_dir, 'last.pth')
    if os.path.exists(resume_path):
        print(f"\n🔄 Resume: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler'):
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        no_improve = checkpoint.get('no_improve', 0)
        print(f"   Epoch {checkpoint['epoch']}, Best F1: {best_f1*100:.2f}%")
    
    print(f"\n🚀 Training: {start_epoch} → {epochs}")
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            block_size=block_size, loss_cfg=loss_cfg
        )
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0:
            val_metrics = validate(model, val_loader, device, block_size, loss_cfg)
            
            print(f"\n📊 Epoch {epoch}:")
            print(f"   NLL:       {val_metrics['nll']:.4f}")
            print(f"   Precision: {val_metrics['precision']*100:.2f}%")
            print(f"   Recall:    {val_metrics['recall']*100:.2f}%")
            print(f"   F1-Score:  {val_metrics['f1']*100:.2f}%")
            print(f"   π mean:    {val_metrics['pi_mean']:.3f} ± {val_metrics['pi_std']:.3f}")
            print(f"   λ mean:    {val_metrics['lambda_mean']:.3f}")
            
            is_best = val_metrics['f1'] > best_f1
            
            if is_best:
                best_f1 = val_metrics['f1']
                no_improve = 0
            else:
                no_improve += val_interval
            
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_f1, no_improve, output_dir, is_best)
            
            # Check targets
            if val_metrics['recall'] > 0.90 and val_metrics['f1'] > 0.65:
                print(f"\n🎯 TARGETS RAGGIUNTI!")
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # Final: threshold tuning
    print(f"\n🔍 Threshold Tuning per Recall Target 90%...")
    
    best_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_path):
        best_state = torch.load(best_path)
        model.load_state_dict(best_state['model'])
    
    thresh_results = find_optimal_threshold(
        model, val_loader, device, block_size,
        target_recall=0.90,
        thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )
    
    print(f"\n📊 Threshold Analysis:")
    print(f"{'Thresh':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 44)
    for r in thresh_results['all_results']:
        print(f"{r['threshold']:<8.2f} {r['precision']*100:<12.1f} {r['recall']*100:<12.1f} {r['f1']*100:<12.1f}")
    
    best_for_recall = thresh_results['for_target_recall']
    print(f"\n✅ Per Recall ≥ 90%: τ = {best_for_recall['threshold']}")
    print(f"   R={best_for_recall['recall']*100:.1f}%, P={best_for_recall['precision']*100:.1f}%, F1={best_for_recall['f1']*100:.1f}%")
    
    # Save
    results = {
        'best_f1': best_f1,
        'optimal_threshold_for_recall': best_for_recall['threshold'],
        'optimal_threshold_for_f1': thresh_results['for_best_f1']['threshold'],
        'threshold_analysis': thresh_results['all_results'],
    }
    
    with open(os.path.join(output_dir, 'stage1_v4_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏁 STAGE 1 V4 COMPLETATO")
    print(f"   Best F1: {best_f1*100:.2f}%")
    print(f"   Threshold consigliato: {best_for_recall['threshold']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()