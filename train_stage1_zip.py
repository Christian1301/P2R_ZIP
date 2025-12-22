#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 - ZIP Pre-training V2

MIGLIORAMENTI:
1. ZIP NLL Loss (formula originale paper) → Migliore bilanciamento Precision/Recall
2. Label Smoothing → Robustezza ai casi borderline
3. Threshold Ottimizzato → Trova threshold che massimizza F1
4. Metriche dettagliate durante training
5. Resume da checkpoint

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
from typing import Tuple

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn


# =============================================================================
# ZIP π NLL - Loss originale dal paper ZIP (semplificata per Stage 1)
# =============================================================================

class ZIPiNLL(nn.Module):
    """
    ZIP π Negative Log-Likelihood (semplificata per Stage 1).
    
    Segue la formula del paper ZIP originale:
    - Per blocchi VUOTI (gt=0): L = -log(π₀ + π₁ · e^{-λ})
    - Per blocchi PIENI (gt>0): L = -log(π₁)
    
    Dove:
    - π₀ = P(blocco vuoto)
    - π₁ = P(blocco pieno) = 1 - π₀
    - λ = valore medio atteso per blocchi pieni (default=2.0)
    
    DIFFERENZA CHIAVE vs BCE:
    - BCE per vuoti: -log(1-p) → penalizza QUALSIASI p > 0
    - ZIP per vuoti: -log(π₀ + π₁·e^{-λ}) → ammette π₁ > 0 se λ piccolo
    
    Questo rende ZIP più "soft" sui confini, il che può migliorare
    il recall senza sacrificare troppa precision.
    
    Parametri:
    - lambda_default: valore λ medio per blocchi pieni
      * λ=1.0: e^{-1} ≈ 0.37 → molto tollerante
      * λ=2.0: e^{-2} ≈ 0.14 → bilanciato (default)
      * λ=3.0: e^{-3} ≈ 0.05 → più strict
      * λ=5.0: e^{-5} ≈ 0.007 → quasi come BCE
    """
    def __init__(
        self,
        lambda_default: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.lambda_default = lambda_default
        self.reduction = reduction
        
        # Pre-calcola exp(-λ) per efficienza
        self.register_buffer(
            'exp_neg_lambda', 
            torch.tensor(float(np.exp(-lambda_default)))
        )
    
    def forward(
        self, 
        logit_pi: torch.Tensor,      # [B, 2, H, W] logits per π (2 classi)
        gt_occupancy: torch.Tensor   # [B, 1, H, W] GT binario (0=vuoto, 1=pieno)
    ) -> torch.Tensor:
        """
        Args:
            logit_pi: [B, 2, H, W] logits per le 2 classi (vuoto, pieno)
            gt_occupancy: [B, 1, H, W] ground truth binario
            
        Returns:
            loss: scalar
        """
        # π tramite softmax sulle 2 classi
        pi = F.softmax(logit_pi, dim=1)  # [B, 2, H, W]
        pi_vuoto = pi[:, 0:1, :, :]  # P(vuoto)
        pi_pieno = pi[:, 1:2, :, :]  # P(pieno)
        
        # Maschere
        is_empty = (gt_occupancy == 0).float()  # Blocchi vuoti
        is_full = (gt_occupancy == 1).float()   # Blocchi pieni
        
        # Loss per blocchi VUOTI (gt=0):
        # L = -log(π₀ + π₁ · e^{-λ})
        # Un blocco può essere osservato vuoto perché:
        # 1. È davvero vuoto (contributo π₀)
        # 2. È pieno ma Poisson(0|λ) = e^{-λ} (contributo π₁ · e^{-λ})
        prob_observe_empty = pi_vuoto + pi_pieno * self.exp_neg_lambda
        loss_empty = -torch.log(prob_observe_empty + 1e-8) * is_empty
        
        # Loss per blocchi PIENI (gt>0):
        # L = -log(π₁)
        # Se osserviamo count > 0, il blocco DEVE essere pieno
        loss_full = -torch.log(pi_pieno + 1e-8) * is_full
        
        # Loss totale
        loss = loss_empty + loss_full
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# ZIP LOSS V2 - Usa ZIPiNLL (fedele al paper)
# =============================================================================

class ZIPLossV2(nn.Module):
    """
    Loss ZIP per Stage 1 usando ZIPiNLL (formula originale del paper).
    
    Componenti:
    1. ZIPiNLL: Classificazione blocchi vuoti/pieni
    2. Count Loss (opzionale): L1 sul count predetto vs GT
    3. Lambda Reg (opzionale): Regolarizzazione su λ
    """
    def __init__(
        self,
        block_size: int = 16,
        # ZIPiNLL params
        lambda_default: float = 2.0,
        # Occupancy threshold
        occupancy_threshold: float = 0.5,
        # Loss weights
        weight_zip_nll: float = 1.0,
        weight_count: float = 0.5,
        weight_lambda_reg: float = 0.01,
    ):
        super().__init__()
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        self.zip_nll = ZIPiNLL(
            lambda_default=lambda_default,
            reduction='mean'
        )
        
        self.weight_zip_nll = weight_zip_nll
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
        
        # 1. ZIP NLL Loss (formula originale del paper)
        loss_zip_nll = self.zip_nll(logit_pi, gt_occupancy)
        
        # 2. Count loss (opzionale)
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
            
            loss_count = F.l1_loss(pred_count_map, gt_count_map)
        else:
            loss_count = torch.tensor(0.0, device=logit_pi.device)
        
        # 3. Lambda regularization
        if self.weight_lambda_reg > 0:
            loss_lambda_reg = (lambda_maps ** 2).mean() * 0.001
        else:
            loss_lambda_reg = torch.tensor(0.0, device=logit_pi.device)
        
        # Total loss
        total_loss = (
            self.weight_zip_nll * loss_zip_nll +
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
            'zip_nll': loss_zip_nll.item(),
            'count': loss_count.item() if isinstance(loss_count, torch.Tensor) else loss_count,
            'lambda_reg': loss_lambda_reg.item() if isinstance(loss_lambda_reg, torch.Tensor) else loss_lambda_reg,
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
    
    # Accumula metriche per ogni threshold (non concatenare tensori di dim diverse)
    threshold_stats = {t: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    
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
        
        # Calcola metriche per ogni threshold
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
    
    # Calcola metriche finali per ogni threshold
    results = []
    
    for thresh in thresholds:
        stats = threshold_stats[thresh]
        tp, tn, fp, fn = stats['tp'], stats['tn'], stats['fp'], stats['fn']
        
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
            'NLL': f"{metrics['zip_nll']:.3f}",
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
    print("🚀 STAGE 1 V2 - ZIP Pre-training con ZIP NLL")
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
    
    # Loss V2 con ZIP NLL (formula originale del paper)
    criterion = ZIPLossV2(
        block_size=block_size,
        lambda_default=float(zip_loss_cfg.get('LAMBDA_DEFAULT', 2.0)),
        occupancy_threshold=float(zip_loss_cfg.get('OCCUPANCY_THRESHOLD', 0.5)),
        weight_zip_nll=float(zip_loss_cfg.get('WEIGHT_ZIP_NLL', 1.0)),
        weight_count=float(zip_loss_cfg.get('WEIGHT_COUNT', 0.5)),
        weight_lambda_reg=float(zip_loss_cfg.get('WEIGHT_LAMBDA_REG', 0.01)),
    ).to(device)
    
    print(f"\n⚙️ ZIP NLL Config (formula originale paper):")
    print(f"   λ_default:        {zip_loss_cfg.get('LAMBDA_DEFAULT', 2.0)}")
    print(f"   e^{{-λ}}:           {np.exp(-float(zip_loss_cfg.get('LAMBDA_DEFAULT', 2.0))):.4f}")
    print(f"   occupancy_thresh: {zip_loss_cfg.get('OCCUPANCY_THRESHOLD', 0.5)}")
    
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
    
    # Training state
    best_f1 = 0.0
    start_epoch = 1
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', 800)
    no_improve = 0
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    
    # Resume da checkpoint se esiste
    resume_path = os.path.join(output_dir, 'last.pth')
    if os.path.exists(resume_path):
        print(f"\n🔄 Ripristino checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        no_improve = checkpoint.get('no_improve', 0)
        
        print(f"   Epoch: {checkpoint['epoch']} → riprendo da {start_epoch}")
        print(f"   Best F1: {best_f1*100:.2f}%")
        print(f"   No improve: {no_improve}/{patience}")
        if checkpoint.get('metrics'):
            m = checkpoint['metrics']
            print(f"   Last metrics: P={m.get('precision', 0)*100:.1f}%, R={m.get('recall', 0)*100:.1f}%, F1={m.get('f1', 0)*100:.1f}%")
    
    print(f"\n🚀 START Training")
    print(f"   Epochs: {start_epoch} → {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Target: F1 > 78%, Recall > 85%, Precision > 70%\n")
    
    for epoch in range(start_epoch, epochs + 1):
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
                val_metrics, best_f1, no_improve, output_dir, is_best
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