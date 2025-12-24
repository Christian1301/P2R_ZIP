#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 V3 - ZIP Pre-training con Focal Loss

MIGLIORAMENTI DA V2:
1. Focal Loss per casi difficili al confine
2. BCE con pos_weight per bilanciamento classi
3. Supervisione count leggera per guidare λ
4. Metriche dettagliate con analisi per fasce
5. Threshold tuning automatico a fine training

TARGET:
- Recall > 90% (priorità alta - non perdere persone)
- Precision > 55% (accettabile - Stage 2 correggerà)
- F1 > 70%
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


# =============================================================================
# FOCAL LOSS - Per casi difficili al confine
# =============================================================================

class FocalBCELoss(nn.Module):
    """
    Focal Loss per classificazione binaria.
    
    Formula: FL(p) = -α × (1-p)^γ × log(p)
    
    Dove:
    - γ (gamma): focusing parameter. γ=0 è BCE standard.
      Valori tipici: 1.0-3.0. Default 2.0.
    - α: peso per bilanciare le classi
    
    Effetto: riduce il peso dei casi facili (p vicino a 1),
    focalizzando l'apprendimento sui casi difficili.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] raw logits (before sigmoid)
            targets: [B, 1, H, W] binary targets (0 or 1)
        """
        probs = torch.sigmoid(logits)
        
        # Clamp per stabilità numerica
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)
        
        # BCE components
        pos_loss = -targets * torch.log(probs)
        neg_loss = -(1 - targets) * torch.log(1 - probs)
        
        # Focal weights: (1-p)^γ per positivi, p^γ per negativi
        pos_weight = (1 - probs) ** self.gamma
        neg_weight = probs ** self.gamma
        
        # Applica class weight ai positivi
        focal_loss = self.pos_weight * pos_weight * pos_loss + neg_weight * neg_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# ZIP LOSS V3 - Con Focal Loss + BCE bilanciata
# =============================================================================

class ZIPLossV3(nn.Module):
    """
    Loss ZIP per Stage 1 con Focal Loss.
    
    Componenti:
    1. Focal BCE: Classificazione con focus su casi difficili
    2. Count Loss: Supervisione leggera sul conteggio
    3. Lambda Reg: Evita valori estremi di λ
    """
    def __init__(
        self,
        block_size: int = 16,
        # Focal params
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: float = 3.0,
        # Threshold
        occupancy_threshold: float = 0.5,
        # Loss weights
        weight_classification: float = 1.0,
        weight_count: float = 0.3,
        weight_lambda_reg: float = 0.005,
    ):
        super().__init__()
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        self.use_focal = use_focal
        
        # Loss functions
        if use_focal:
            self.classification_loss = FocalBCELoss(
                gamma=focal_gamma,
                pos_weight=pos_weight,
                reduction='mean'
            )
        else:
            self.classification_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]),
                reduction='mean'
            )
        
        self.weight_classification = weight_classification
        self.weight_count = weight_count
        self.weight_lambda_reg = weight_lambda_reg
    
    def compute_gt_occupancy(self, gt_density: torch.Tensor) -> torch.Tensor:
        """Calcola maschera occupancy dal GT."""
        gt_counts = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        return (gt_counts > self.occupancy_threshold).float()
    
    def forward(self, outputs: dict, gt_density: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: dict dal modello
            gt_density: [B, 1, H, W] ground truth density
        """
        logit_pi = outputs['logit_pi_maps']  # [B, 2, H, W]
        lambda_maps = outputs['lambda_maps']
        pi_probs = outputs['pi_probs']  # Probabilità classe "pieno"
        
        # GT occupancy
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != logit_pi.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=logit_pi.shape[-2:],
                mode='nearest'
            )
        
        # 1. Classification Loss (Focal o BCE)
        logit_pieno = logit_pi[:, 1:2, :, :]  # Logit per "pieno"
        
        if self.use_focal:
            loss_cls = self.classification_loss(logit_pieno, gt_occupancy)
        else:
            if self.classification_loss.pos_weight.device != logit_pieno.device:
                self.classification_loss.pos_weight = self.classification_loss.pos_weight.to(logit_pieno.device)
            loss_cls = self.classification_loss(logit_pieno, gt_occupancy)
        
        # 2. Count loss (supervisione leggera)
        if self.weight_count > 0:
            pred_count_map = pi_probs * lambda_maps
            gt_count_map = F.avg_pool2d(
                gt_density,
                kernel_size=self.block_size,
                stride=self.block_size
            ) * (self.block_size ** 2)
            
            if gt_count_map.shape[-2:] != pred_count_map.shape[-2:]:
                gt_count_map = F.interpolate(
                    gt_count_map,
                    size=pred_count_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Solo su blocchi non vuoti per non dominare la loss
            mask_nonzero = (gt_count_map > 0.5).float()
            if mask_nonzero.sum() > 0:
                loss_count = (torch.abs(pred_count_map - gt_count_map) * mask_nonzero).sum() / mask_nonzero.sum()
            else:
                loss_count = torch.tensor(0.0, device=logit_pi.device)
        else:
            loss_count = torch.tensor(0.0, device=logit_pi.device)
        
        # 3. Lambda regularization
        if self.weight_lambda_reg > 0:
            # Penalizza λ troppo grandi
            loss_lambda_reg = F.relu(lambda_maps - 5.0).mean()
        else:
            loss_lambda_reg = torch.tensor(0.0, device=logit_pi.device)
        
        # Total loss
        total_loss = (
            self.weight_classification * loss_cls +
            self.weight_count * loss_count +
            self.weight_lambda_reg * loss_lambda_reg
        )
        
        # Metriche dettagliate
        with torch.no_grad():
            pred_binary = (pi_probs > 0.5).float()
            
            tp = ((pred_binary == 1) & (gt_occupancy == 1)).sum().item()
            tn = ((pred_binary == 0) & (gt_occupancy == 0)).sum().item()
            fp = ((pred_binary == 1) & (gt_occupancy == 0)).sum().item()
            fn = ((pred_binary == 0) & (gt_occupancy == 1)).sum().item()
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            coverage = pred_binary.mean().item() * 100
            
            # Statistiche π e λ
            pi_mean = pi_probs.mean().item()
            pi_std = pi_probs.std().item()
            lambda_mean = lambda_maps.mean().item()
            lambda_max = lambda_maps.max().item()
        
        metrics = {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'count': loss_count.item() if isinstance(loss_count, torch.Tensor) else loss_count,
            'lambda_reg': loss_lambda_reg.item() if isinstance(loss_lambda_reg, torch.Tensor) else loss_lambda_reg,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'pi_mean': pi_mean,
            'pi_std': pi_std,
            'lambda_mean': lambda_mean,
            'lambda_max': lambda_max,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }
        
        return total_loss, metrics


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
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
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
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Training di una epoch."""
    model.train()
    
    total_loss = 0.0
    metrics_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Stage1 V3 [Ep {epoch}]")
    
    for images, gt_density, _ in pbar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss, metrics = criterion(outputs, gt_density)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        pbar.set_postfix({
            'L': f"{loss.item():.3f}",
            'P': f"{metrics['precision']*100:.1f}%",
            'R': f"{metrics['recall']*100:.1f}%",
            'F1': f"{metrics['f1']*100:.1f}%",
            'π': f"{metrics['pi_mean']:.2f}",
        })
    
    n = len(dataloader)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / n
    
    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int
) -> dict:
    """Validazione con metriche complete."""
    model.eval()
    
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    total_loss = 0.0
    pi_means, pi_stds = [], []
    lambda_means = []
    
    for images, gt_density, _ in tqdm(dataloader, desc="Validate", leave=False):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        loss, _ = criterion(outputs, gt_density)
        total_loss += loss.item()
        
        pi_probs = outputs['pi_probs']
        lambda_maps = outputs['lambda_maps']
        
        pi_means.append(pi_probs.mean().item())
        pi_stds.append(pi_probs.std().item())
        lambda_means.append(lambda_maps.mean().item())
        
        # GT occupancy
        gt_counts = F.avg_pool2d(
            gt_density, kernel_size=block_size, stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts > 0.5).float()
        
        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=pi_probs.shape[-2:], mode='nearest'
            )
        
        pred_binary = (pi_probs > 0.5).float()
        
        total_tp += ((pred_binary == 1) & (gt_occupancy == 1)).sum().item()
        total_tn += ((pred_binary == 0) & (gt_occupancy == 0)).sum().item()
        total_fp += ((pred_binary == 1) & (gt_occupancy == 0)).sum().item()
        total_fn += ((pred_binary == 0) & (gt_occupancy == 1)).sum().item()
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp, 'tn': total_tn, 'fp': total_fp, 'fn': total_fn,
        'pi_mean': np.mean(pi_means),
        'pi_std': np.mean(pi_stds),
        'lambda_mean': np.mean(lambda_means),
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
    print("🚀 STAGE 1 V3 - ZIP Pre-training con Focal Loss")
    print("="*70)
    print(f"Device: {device}")
    print(f"Target: Recall > 90%, F1 > 70%")
    print("="*70)
    
    # Config
    data_cfg = config['DATA']
    optim_cfg = config['OPTIM_ZIP']
    loss_cfg = config.get('ZIP_LOSS_V3', {})
    
    block_size = data_cfg['ZIP_BLOCK_SIZE']
    
    # Dataset
    DatasetClass = get_dataset(config['DATASET'])
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    train_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=block_size,
        transforms=train_tf
    )
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=block_size,
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg['BATCH_SIZE'],
        shuffle=True,
        num_workers=optim_cfg['NUM_WORKERS'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"\n📊 Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Model
    bin_config = config['BINS_CONFIG'][config['DATASET']]
    zip_head_cfg = config.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        gate=config['MODEL']['GATE'],
        upsample_to_input=config['MODEL']['UPSAMPLE_TO_INPUT'],
        use_ste_mask=config['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        }
    ).to(device)
    
    # Loss V3
    criterion = ZIPLossV3(
        block_size=block_size,
        use_focal=loss_cfg.get('USE_FOCAL', True),
        focal_gamma=float(loss_cfg.get('FOCAL_GAMMA', 2.0)),
        pos_weight=float(loss_cfg.get('POS_WEIGHT', 3.0)),
        occupancy_threshold=float(loss_cfg.get('OCCUPANCY_THRESHOLD', 0.5)),
        weight_classification=float(loss_cfg.get('WEIGHT_CLASSIFICATION', 1.0)),
        weight_count=float(loss_cfg.get('WEIGHT_COUNT', 0.3)),
        weight_lambda_reg=float(loss_cfg.get('WEIGHT_LAMBDA_REG', 0.005)),
    ).to(device)
    
    print(f"\n⚙️ Loss Config:")
    print(f"   Focal Loss: {loss_cfg.get('USE_FOCAL', True)} (γ={loss_cfg.get('FOCAL_GAMMA', 2.0)})")
    print(f"   pos_weight: {loss_cfg.get('POS_WEIGHT', 3.0)}")
    
    # Optimizer
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': float(optim_cfg['LR_BACKBONE'])},
        {'params': model.zip_head.parameters(), 'lr': float(optim_cfg['BASE_LR'])},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(optim_cfg['WEIGHT_DECAY']))
    
    # Scheduler
    epochs = optim_cfg['EPOCHS']
    warmup = optim_cfg.get('WARMUP_EPOCHS', 100)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup])
    
    # Output
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
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
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0:
            val_metrics = validate(model, criterion, val_loader, device, block_size)
            
            print(f"\n📊 Epoch {epoch}:")
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
            if val_metrics['recall'] > 0.90 and val_metrics['f1'] > 0.70:
                print(f"\n🎯 TARGETS RAGGIUNTI!")
            
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # Final: threshold tuning
    print(f"\n🔍 Threshold Tuning per Recall Target 90%...")
    
    best_state = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(best_state['model'])
    
    thresh_results = find_optimal_threshold(
        model, val_loader, device, block_size,
        target_recall=0.90,
        thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
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
    
    with open(os.path.join(output_dir, 'stage1_v3_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏁 STAGE 1 V3 COMPLETATO")
    print(f"   Best F1: {best_f1*100:.2f}%")
    print(f"   Threshold consigliato: {best_for_recall['threshold']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()