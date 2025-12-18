#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 - ZIP Pre-training V2

MIGLIORAMENTI:
1. Focal Loss invece di BCE → Migliore bilanciamento Precision/Recall
2. Label Smoothing → Robustezza ai casi borderline
3. Threshold Ottimizzato → Trova threshold che massimizza F1
4. Metriche dettagliate durante training

TARGET:
- Recall > 85%
- Precision > 70%
- F1-Score > 78%
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

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn


# =============================================================================
# FOCAL LOSS - Migliore bilanciamento Precision/Recall
# =============================================================================

class FocalBCELoss(nn.Module):
    """
    Focal Loss per classificazione binaria.
    
    FL(p) = -α * (1-p)^γ * log(p)     per classe positiva
    FL(p) = -(1-α) * p^γ * log(1-p)   per classe negativa
    
    Parametri:
    - alpha: peso per classe positiva (default 0.75 per favorire recall)
    - gamma: focusing parameter (default 2.0)
      * gamma=0: equivalente a BCE
      * gamma>0: riduce peso campioni facili
    - pos_weight: peso aggiuntivo per positivi (come in BCE)
    """
    def __init__(
        self, 
        alpha: float = 0.75,      # Peso classe positiva
        gamma: float = 2.0,       # Focusing parameter
        pos_weight: float = 1.0,  # Peso aggiuntivo positivi
        label_smoothing: float = 0.0,  # Label smoothing
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] raw logits (prima di sigmoid)
            targets: [B, 1, H, W] ground truth (0 o 1)
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Probabilità
        probs = torch.sigmoid(logits)
        
        # BCE base
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal weight: (1-p)^γ per positivi, p^γ per negativi
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Pos weight aggiuntivo
        weight = torch.where(targets > 0.5, self.pos_weight, 1.0)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# ZIP LOSS MIGLIORATA
# =============================================================================

class ZIPLossV2(nn.Module):
    """
    Loss ZIP migliorata con Focal Loss e metriche dettagliate.
    """
    def __init__(
        self,
        block_size: int = 16,
        # Focal Loss params
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float = 3.0,
        label_smoothing: float = 0.05,
        # Occupancy threshold
        occupancy_threshold: float = 0.5,
        # Loss weights
        weight_focal: float = 1.0,
        weight_count: float = 0.5,
        weight_lambda_reg: float = 0.01,
    ):
        super().__init__()
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        self.focal_loss = FocalBCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing
        )
        
        self.weight_focal = weight_focal
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
            outputs: dict dal modello con 'logit_pi_maps', 'lambda_maps', 'pi_probs'
            gt_density: [B, 1, H, W] ground truth density
            
        Returns:
            loss: scalar
            metrics: dict con dettagli
        """
        logit_pi = outputs['logit_pi_maps']
        lambda_maps = outputs['lambda_maps']
        pi_probs = outputs['pi_probs']
        
        # Logit per classe "pieno" (canale 1)
        logit_pieno = logit_pi[:, 1:2, :, :]
        
        # GT occupancy
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=logit_pieno.shape[-2:],
                mode='nearest'
            )
        
        # 1. Focal Loss per classificazione
        loss_focal = self.focal_loss(logit_pieno, gt_occupancy)
        
        # 2. Count loss (opzionale)
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
        
        loss_count = F.l1_loss(pred_count_map, gt_count_map)
        
        # 3. Lambda regularization
        loss_lambda_reg = (lambda_maps ** 2).mean() * 0.001
        
        # Total loss
        total_loss = (
            self.weight_focal * loss_focal +
            self.weight_count * loss_count +
            self.weight_lambda_reg * loss_lambda_reg
        )
        
        # Metriche per monitoring
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
        
        metrics = {
            'total': total_loss.item(),
            'focal': loss_focal.item(),
            'count': loss_count.item(),
            'lambda_reg': loss_lambda_reg.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
        
        return total_loss, metrics


# =============================================================================
# OPTIMAL THRESHOLD FINDER
# =============================================================================

@torch.no_grad()
def find_optimal_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    block_size: int,
    thresholds: list = None
) -> dict:
    """
    Trova il threshold ottimale che massimizza F1-Score.
    
    Returns:
        dict con threshold ottimale e metriche per ogni threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    model.eval()
    
    # Raccogli tutte le predizioni
    all_pi_probs = []
    all_gt_occupancy = []
    
    for images, gt_density, _ in dataloader:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        pi_probs = outputs['pi_probs']
        
        # GT occupancy
        gt_counts = F.avg_pool2d(
            gt_density,
            kernel_size=block_size,
            stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts > 0.5).float()
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=pi_probs.shape[-2:],
                mode='nearest'
            )
        
        all_pi_probs.append(pi_probs.cpu())
        all_gt_occupancy.append(gt_occupancy.cpu())
    
    all_pi_probs = torch.cat(all_pi_probs, dim=0)
    all_gt_occupancy = torch.cat(all_gt_occupancy, dim=0)
    
    # Test ogni threshold
    results = []
    
    for thresh in thresholds:
        pred_binary = (all_pi_probs > thresh).float()
        
        tp = ((pred_binary == 1) & (all_gt_occupancy == 1)).sum().item()
        tn = ((pred_binary == 0) & (all_gt_occupancy == 0)).sum().item()
        fp = ((pred_binary == 1) & (all_gt_occupancy == 0)).sum().item()
        fn = ((pred_binary == 0) & (all_gt_occupancy == 1)).sum().item()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Trova migliore per F1
    best = max(results, key=lambda x: x['f1'])
    
    return {
        'optimal_threshold': best['threshold'],
        'best_f1': best['f1'],
        'best_precision': best['precision'],
        'best_recall': best['recall'],
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
    
    pbar = tqdm(dataloader, desc=f"Stage1 V2 [Ep {epoch}]")
    
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
            metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        pbar.set_postfix({
            'L': f"{loss.item():.3f}",
            'P': f"{metrics['precision']*100:.1f}%",
            'R': f"{metrics['recall']*100:.1f}%",
            'F1': f"{metrics['f1']*100:.1f}%",
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
    
    for images, gt_density, _ in tqdm(dataloader, desc="Validate", leave=False):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        loss, _ = criterion(outputs, gt_density)
        total_loss += loss.item()
        
        pi_probs = outputs['pi_probs']
        
        # GT occupancy
        gt_counts = F.avg_pool2d(
            gt_density,
            kernel_size=block_size,
            stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts > 0.5).float()
        
        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=pi_probs.shape[-2:],
                mode='nearest'
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
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_f1, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_f1': best_f1,
    }
    
    # Latest
    torch.save(state, os.path.join(output_dir, 'last.pth'))
    
    # Best
    if is_best:
        torch.save(state, os.path.join(output_dir, 'best_model.pth'))
        print(f"💾 Saved best: F1={metrics['f1']*100:.2f}%, P={metrics['precision']*100:.1f}%, R={metrics['recall']*100:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])
    
    print("="*70)
    print("🚀 STAGE 1 V2 - ZIP Pre-training con Focal Loss")
    print("="*70)
    print(f"Device: {device}")
    print("="*70)
    
    # Config shortcuts
    data_cfg = config['DATA']
    optim_cfg = config['OPTIM_ZIP']
    zip_loss_cfg = config.get('ZIP_LOSS_V2', {})
    
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
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    
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
        use_ste_mask=config['MODEL'].get('USE_STE_MASK', True),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        }
    ).to(device)
    
    # Loss V2 con Focal Loss
    criterion = ZIPLossV2(
        block_size=block_size,
        focal_alpha=float(zip_loss_cfg.get('FOCAL_ALPHA', 0.75)),
        focal_gamma=float(zip_loss_cfg.get('FOCAL_GAMMA', 2.0)),
        pos_weight=float(zip_loss_cfg.get('POS_WEIGHT', 3.0)),
        label_smoothing=float(zip_loss_cfg.get('LABEL_SMOOTHING', 0.05)),
        occupancy_threshold=float(zip_loss_cfg.get('OCCUPANCY_THRESHOLD', 0.5)),
        weight_focal=float(zip_loss_cfg.get('WEIGHT_FOCAL', 1.0)),
        weight_count=float(zip_loss_cfg.get('WEIGHT_COUNT', 0.5)),
        weight_lambda_reg=float(zip_loss_cfg.get('WEIGHT_LAMBDA_REG', 0.01)),
    ).to(device)
    
    print(f"\n⚙️ Focal Loss Config:")
    print(f"   α (alpha):        {zip_loss_cfg.get('FOCAL_ALPHA', 0.75)}")
    print(f"   γ (gamma):        {zip_loss_cfg.get('FOCAL_GAMMA', 2.0)}")
    print(f"   pos_weight:       {zip_loss_cfg.get('POS_WEIGHT', 3.0)}")
    print(f"   label_smoothing:  {zip_loss_cfg.get('LABEL_SMOOTHING', 0.05)}")
    
    # Optimizer
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': float(optim_cfg['LR_BACKBONE'])},
        {'params': model.zip_head.parameters(), 'lr': float(optim_cfg['BASE_LR'])},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(optim_cfg['WEIGHT_DECAY'])
    )
    
    # Scheduler con warmup
    epochs = optim_cfg['EPOCHS']
    warmup = optim_cfg.get('WARMUP_EPOCHS', 100)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup,
        eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup]
    )
    
    # Output dir
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Training
    best_f1 = 0.0
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', 800)
    no_improve = 0
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    
    print(f"\n🚀 START Training")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Target: F1 > 78%, Recall > 85%, Precision > 70%\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0:
            val_metrics = validate(model, criterion, val_loader, device, block_size)
            
            print(f"\n📊 Epoch {epoch} Validation:")
            print(f"   Accuracy:  {val_metrics['accuracy']*100:.2f}%")
            print(f"   Precision: {val_metrics['precision']*100:.2f}%")
            print(f"   Recall:    {val_metrics['recall']*100:.2f}%")
            print(f"   F1-Score:  {val_metrics['f1']*100:.2f}%")
            
            # Check improvement
            is_best = val_metrics['f1'] > best_f1
            
            if is_best:
                best_f1 = val_metrics['f1']
                no_improve = 0
            else:
                no_improve += val_interval
            
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, best_f1, output_dir, is_best
            )
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
            
            # Target check
            if (val_metrics['f1'] > 0.78 and 
                val_metrics['recall'] > 0.85 and 
                val_metrics['precision'] > 0.70):
                print(f"\n🎯 TARGET RAGGIUNTO!")
                print(f"   F1={val_metrics['f1']*100:.1f}% > 78%")
                print(f"   Recall={val_metrics['recall']*100:.1f}% > 85%")
                print(f"   Precision={val_metrics['precision']*100:.1f}% > 70%")
    
    # Final: Find optimal threshold
    print(f"\n🔍 Ricerca threshold ottimale...")
    
    # Carica best model
    best_state = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(best_state['model'])
    
    thresh_results = find_optimal_threshold(
        model, val_loader, device, block_size,
        thresholds=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    )
    
    print(f"\n📊 Threshold Analysis:")
    print(f"{'Thresh':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 44)
    for r in thresh_results['all_results']:
        marker = " ← BEST" if r['threshold'] == thresh_results['optimal_threshold'] else ""
        print(f"{r['threshold']:<8.2f} {r['precision']*100:<12.1f} {r['recall']*100:<12.1f} {r['f1']*100:<12.1f}{marker}")
    
    print(f"\n✅ Threshold ottimale: {thresh_results['optimal_threshold']}")
    print(f"   → Aggiorna config.yaml: ZIP_PI_THRESH: {thresh_results['optimal_threshold']}")
    
    # Save results
    results = {
        'best_f1': best_f1,
        'optimal_threshold': thresh_results['optimal_threshold'],
        'threshold_analysis': thresh_results['all_results'],
    }
    
    with open(os.path.join(output_dir, 'stage1_v2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏁 STAGE 1 V2 COMPLETATO")
    print(f"{'='*70}")
    print(f"   Best F1: {best_f1*100:.2f}%")
    print(f"   Optimal Threshold: {thresh_results['optimal_threshold']}")
    print(f"   Checkpoint: {output_dir}/best_model.pth")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()