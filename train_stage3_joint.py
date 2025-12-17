#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 - JOINT TRAINING con Loss Congiunta

FORMULA LOSS: L_total = (1-α)·L_ZIP + α·L_P2R

dove:
- L_ZIP: Binary Cross-Entropy per classificazione blocchi (vuoto/pieno)
- L_P2R: Count + Spatial loss per density map refinement
- α ∈ [0,1]: parametro di bilanciamento
  * α=0: solo ZIP (focus su localizzazione)
  * α=1: solo P2R (focus su densità)
  * α=0.5: bilanciamento neutro

STRATEGIA:
1. Carica best checkpoint da Stage 2
2. Sblocca tutti i componenti (backbone, ZIP head, P2R head)
3. Fine-tuning end-to-end con loss congiunta
4. LR differenziati per stabilità

TARGET: Migliorare coerenza ZIP-P2R mantenendo MAE < 70
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


# =============================================================================
# LOSS COMPONENTS
# =============================================================================

class PiHeadLoss(nn.Module):
    """
    Loss per π-head (componente ZIP della loss congiunta).
    
    Binary Cross-Entropy con pos_weight per bilanciare classi:
    - Classe 0: blocco vuoto (nessuna persona)
    - Classe 1: blocco pieno (almeno una persona)
    """
    def __init__(
        self, 
        pos_weight: float = 8.0, 
        block_size: int = 16, 
        occupancy_threshold: float = 0.5
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        """Calcola maschera binaria GT da density map."""
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        return (gt_counts_per_block > self.occupancy_threshold).float()
    
    def forward(self, logit_pi_maps, gt_density):
        """
        Args:
            logit_pi_maps: [B, 2, Hb, Wb] logits π-head
            gt_density: [B, 1, H, W] density map GT
            
        Returns:
            loss: scalar
            metrics: dict con statistiche
        """
        # Estrai logit per classe "pieno" (canale 1)
        logit_pieno = logit_pi_maps[:, 1:2, :, :]
        
        # GT occupancy
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, 
                size=logit_pieno.shape[-2:], 
                mode='nearest'
            )
        
        # Sposta pos_weight su device corretto
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        
        loss = self.bce(logit_pieno, gt_occupancy)
        
        # Metriche
        with torch.no_grad():
            pred_prob = torch.sigmoid(logit_pieno)
            pred_occupancy = (pred_prob > 0.5).float()
            
            coverage = pred_occupancy.mean().item() * 100
            
            # Recall
            if gt_occupancy.sum() > 0:
                tp = (pred_occupancy * gt_occupancy).sum()
                fn = ((1 - pred_occupancy) * gt_occupancy).sum()
                recall = (tp / (tp + fn + 1e-6)).item() * 100
            else:
                recall = 100.0
        
        return loss, {
            'coverage': coverage,
            'recall': recall,
        }


class P2RCountLoss(nn.Module):
    """
    Loss P2R semplificata per Stage 3 (componente P2R della loss congiunta).
    
    Focus su:
    1. Count accuracy (MAE sul conteggio totale)
    2. Spatial consistency (localizzazione predizioni)
    """
    def __init__(
        self,
        count_weight: float = 2.0,
        spatial_weight: float = 0.15,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
    
    def forward(self, pred_density, points_list, cell_area):
        """
        Args:
            pred_density: [B, 1, H, W] density predictions
            points_list: lista di tensori [N_i, 2]
            cell_area: area cella per scaling count
            
        Returns:
            loss: scalar
        """
        B, _, H, W = pred_density.shape
        device = pred_density.device
        
        total_count_loss = torch.tensor(0.0, device=device)
        total_spatial_loss = torch.tensor(0.0, device=device)
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = pred_density[i].sum() / cell_area
            
            # Count loss (L1)
            total_count_loss += torch.abs(pred_count - gt)
            
            # Spatial loss (solo se ci sono persone)
            if gt > 0:
                # Target gaussiano semplificato
                target = torch.zeros(H, W, device=device)
                
                for pt in pts:
                    x = int(pt[0].clamp(0, W-1).item())
                    y = int(pt[1].clamp(0, H-1).item())
                    
                    # 3x3 gaussian
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                dist = (dx*dx + dy*dy) ** 0.5
                                target[ny, nx] += np.exp(-dist / 2)
                
                if target.sum() > 0:
                    target = target / target.sum()
                    pred_norm = pred_density[i, 0] / (pred_density[i, 0].sum() + 1e-8)
                    total_spatial_loss += F.mse_loss(pred_norm, target)
        
        avg_count = total_count_loss / B
        avg_spatial = total_spatial_loss / B
        
        return self.count_weight * avg_count + self.spatial_weight * avg_spatial


# =============================================================================
# JOINT LOSS: (1-α)·L_ZIP + α·L_P2R
# =============================================================================

class JointLoss(nn.Module):
    """
    Loss Congiunta per Stage 3.
    
    Formula: L_total = (1-α)·L_ZIP + α·L_P2R
    
    Componenti:
    - L_ZIP: Binary classification (blocchi vuoti vs pieni)
    - L_P2R: Density regression (count + spatial)
    - α: parametro di bilanciamento
    
    Interpretazione di α:
    - α→0: priorità a ZIP (localizzazione coarse)
    - α→1: priorità a P2R (densità fine)
    - α=0.5: bilanciamento neutro
    
    Raccomandazioni:
    - α=0.3-0.4: se Stage 1 è già forte
    - α=0.5: default
    - α=0.6-0.7: se Stage 2 ha performance molto migliori
    """
    def __init__(
        self,
        alpha: float = 0.5,
        # ZIP params
        pi_pos_weight: float = 8.0,
        block_size: int = 16,
        occupancy_threshold: float = 0.5,
        # P2R params
        count_weight: float = 2.0,
        spatial_weight: float = 0.15,
    ):
        super().__init__()
        self.alpha = alpha
        
        self.zip_loss = PiHeadLoss(
            pos_weight=pi_pos_weight,
            block_size=block_size,
            occupancy_threshold=occupancy_threshold
        )
        
        self.p2r_loss = P2RCountLoss(
            count_weight=count_weight,
            spatial_weight=spatial_weight,
        )
    
    def forward(self, outputs, gt_density, points_list, cell_area):
        """
        Args:
            outputs: dict con 'logit_pi_maps', 'p2r_density'
            gt_density: [B, 1, H, W]
            points_list: lista tensori [N_i, 2]
            cell_area: float
            
        Returns:
            total_loss: scalar
            metrics: dict
        """
        # L_ZIP (L1)
        l_zip, zip_metrics = self.zip_loss(
            outputs['logit_pi_maps'],
            gt_density
        )
        
        # L_P2R (L2)
        l_p2r = self.p2r_loss(
            outputs['p2r_density'],
            points_list,
            cell_area
        )
        
        # Loss totale: (1-α)·L1 + α·L2
        total_loss = (1 - self.alpha) * l_zip + self.alpha * l_p2r
        
        metrics = {
            'total': total_loss.item(),
            'zip': l_zip.item(),
            'p2r': l_p2r.item(),
            'alpha': self.alpha,
            'zip_coverage': zip_metrics['coverage'],
            'zip_recall': zip_metrics['recall'],
        }
        
        return total_loss, metrics


# =============================================================================
# TRAINING & VALIDATION
# =============================================================================

def train_one_epoch(
    model, criterion, dataloader, optimizer, scheduler,
    device, default_down, epoch, config
):
    """Training con loss congiunta."""
    model.train()
    
    # Backbone in eval per BatchNorm frozen (opzionale)
    if config.get('FREEZE_BN', False):
        model.backbone.eval()
    
    total_loss = 0.0
    metrics_accum = {}
    
    progress_bar = tqdm(dataloader, desc=f"Stage3 Joint [Ep {epoch}]")
    
    for images, gt_density, points in progress_bar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points_list = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # Canonicalize P2R density
        pred_density = outputs['p2r_density']
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        outputs['p2r_density'] = pred_density
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Loss congiunta
        loss, metrics = criterion(outputs, gt_density, points_list, cell_area)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumula metriche
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
        
        progress_bar.set_postfix({
            'L': f"{loss.item():.3f}",
            'ZIP': f"{metrics['zip']:.3f}",
            'P2R': f"{metrics['p2r']:.3f}",
            'α': f"{metrics['alpha']:.2f}",
        })
    
    if scheduler:
        scheduler.step()
    
    # Media metriche
    n = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n
    
    print(f"\n   Epoch {epoch} Summary:")
    print(f"      Total Loss: {total_loss/n:.4f}")
    print(f"      ZIP Loss:   {metrics_accum['zip']:.4f} (weight={1-criterion.alpha:.2f})")
    print(f"      P2R Loss:   {metrics_accum['p2r']:.4f} (weight={criterion.alpha:.2f})")
    print(f"      ZIP Coverage: {metrics_accum['zip_coverage']:.1f}%")
    print(f"      ZIP Recall:   {metrics_accum['zip_recall']:.1f}%")
    
    return total_loss / n, metrics_accum


@torch.no_grad()
def validate(model, dataloader, device, default_down):
    """Validazione end-to-end."""
    model.eval()
    
    all_mae = []
    all_mse = []
    total_pred = 0.0
    total_gt = 0.0
    
    for images, densities, points in tqdm(dataloader, desc="Validate", leave=False):
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
            
            total_pred += pred_count
            total_gt += gt
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 Validation Results")
    print(f"{'='*60}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Bias: {bias:.3f}")
    print(f"{'='*60}\n")
    
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, mae, best_mae, output_dir, is_best=False):
    """Salva checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'mae': mae,
        'best_mae': best_mae,
    }
    
    # Latest
    latest_path = os.path.join(output_dir, 'stage3_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Best
    if is_best:
        best_path = os.path.join(output_dir, 'stage3_best.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 Saved: stage3_best.pth (MAE={mae:.2f})")
    else:
        print(f"💾 Saved: stage3_latest.pth (epoch {epoch})")


def load_stage2_checkpoint(model, output_dir, device):
    """Carica checkpoint Stage 2."""
    candidates = [
        'stage2_best.pth',
        'best_model.pth',
    ]
    
    for name in candidates:
        ckpt_path = os.path.join(output_dir, name)
        if os.path.isfile(ckpt_path):
            print(f"\n✅ Caricamento Stage 2: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            
            if 'model' in state:
                state = state['model']
            elif 'model_state_dict' in state:
                state = state['model_state_dict']
            
            model.load_state_dict(state, strict=False)
            return True
    
    print(f"⚠️ Nessun checkpoint Stage 2 trovato in {output_dir}")
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Carica config
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml non trovato")
        return
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])
    
    print("="*60)
    print("🚀 Stage 3 - JOINT TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Formula Loss: (1-α)·L_ZIP + α·L_P2R")
    print("="*60)
    
    # Config
    data_cfg = config['DATA']
    optim_cfg = config.get('OPTIM_JOINT', {})
    joint_cfg = config.get('JOINT_LOSS', {})
    
    alpha = float(joint_cfg.get('ALPHA', 0.5))
    epochs = optim_cfg.get('EPOCHS', 600)
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', 150)
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    print(f"\n⚙️ Hyperparameters:")
    print(f"   α (bilanciamento): {alpha}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience}")
    print(f"   Val interval: {val_interval}")
    
    # Dataset
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config['DATASET'])
    
    train_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['TRAIN_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=train_transforms
    )
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg.get('BATCH_SIZE', 6),
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
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    
    # Modello
    bin_config = config['BINS_CONFIG'][config['DATASET']]
    zip_head_kwargs = {
        'lambda_scale': config['ZIP_HEAD'].get('LAMBDA_SCALE', 1.2),
        'lambda_max': config['ZIP_HEAD'].get('LAMBDA_MAX', 8.0),
        'use_softplus': config['ZIP_HEAD'].get('USE_SOFTPLUS', True),
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        gate=config['MODEL']['GATE'],
        upsample_to_input=False,
        use_ste_mask=True,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # Output directory
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica Stage 2
    if not load_stage2_checkpoint(model, output_dir, device):
        print("⚠️ Continuando senza checkpoint Stage 2...")
    
    # Setup parametri trainabili
    print(f"\n🔧 Setup Training:")
    
    lr_backbone = float(optim_cfg.get('LR_BACKBONE', 1e-6))
    lr_heads = float(optim_cfg.get('LR_HEADS', 5e-5))
    
    param_groups = []
    
    if lr_backbone > 0:
        param_groups.append({
            'params': model.backbone.parameters(),
            'lr': lr_backbone,
            'name': 'backbone'
        })
        print(f"   Backbone: LR={lr_backbone}")
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print(f"   Backbone: FROZEN")
    
    # ZIP + P2R heads
    head_params = []
    for name in ['zip_head', 'p2r_head']:
        head_params.extend(
            p for n, p in model.named_parameters() 
            if name in n and p.requires_grad
        )
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': lr_heads,
            'name': 'heads'
        })
        print(f"   Heads (ZIP+P2R): LR={lr_heads}")
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Trainabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(optim_cfg.get('WEIGHT_DECAY', 1e-4))
    )
    
    # Scheduler
    warmup = optim_cfg.get('WARMUP_EPOCHS', 10)
    
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss congiunta
    criterion = JointLoss(
        alpha=alpha,
        pi_pos_weight=float(joint_cfg.get('PI_POS_WEIGHT', 8.0)),
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        occupancy_threshold=float(joint_cfg.get('OCCUPANCY_THRESHOLD', 0.5)),
        count_weight=float(joint_cfg.get('COUNT_WEIGHT', 2.0)),
        spatial_weight=float(joint_cfg.get('SPATIAL_WEIGHT', 0.15)),
    ).to(device)
    
    print(f"\n⚙️ Loss Congiunta:")
    print(f"   α = {alpha}")
    print(f"   Formula: L = {1-alpha:.2f}·L_ZIP + {alpha:.2f}·L_P2R")
    
    # Valutazione iniziale
    print(f"\n📋 Valutazione iniziale:")
    init_results = validate(model, val_loader, device, default_down)
    
    best_mae = init_results['mae']
    no_improve = 0
    
    # Training loop
    print(f"\n🚀 START Training")
    print(f"   Baseline MAE: {best_mae:.2f}")
    print(f"   Target: MAE < 70\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            device, default_down, epoch, optim_cfg
        )
        
        # Validate
        if epoch % val_interval == 0:
            results = validate(model, val_loader, device, default_down)
            
            current_mae = results['mae']
            is_better = current_mae < best_mae
            
            if is_better:
                best_mae = current_mae
                no_improve = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    current_mae, best_mae, output_dir, is_best=True
                )
                print(f"🏆 NEW BEST: MAE={best_mae:.2f}")
            else:
                no_improve += val_interval
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    current_mae, best_mae, output_dir, is_best=False
                )
                print(f"   No improvement ({no_improve}/{patience})")
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # Risultati finali
    print("\n" + "="*60)
    print("🏁 STAGE 3 COMPLETATO")
    print("="*60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage3_best.pth")
    
    if best_mae < 70:
        print(f"   🎯 TARGET RAGGIUNTO!")
    
    print("="*60)


if __name__ == '__main__':
    main()