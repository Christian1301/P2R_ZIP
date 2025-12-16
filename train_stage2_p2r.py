#!/usr/bin/env python3
"""
Stage 2 V8 - P2R Training con Multi-Scale Loss + Augmentation Potenziata

MODIFICHE DA V7:
1. Multi-scale loss con scales [1, 2, 4]
   - Calcola count a diverse risoluzioni
   - Forza consistenza multi-scala
2. 5000 epoche (era 3000)
3. Data augmentation potenziata:
   - Color jitter
   - Random grayscale
   - Gaussian blur
4. Warmup + gradient clipping

FIX V8.1:
- Device mismatch in EnhancedTransformsV8 (mean/std su CPU vs image su CUDA)
- Checkpoint path: cerca best_model.pth invece di stage1_best.pth

OBIETTIVO: MAE 68.97 → 63-66
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
import random
from PIL import Image

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from train_utils import (
    init_seeds,
    collate_fn,
    canonicalize_p2r_grid,
)


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


# =============================================================================
# MULTI-SCALE P2R LOSS
# =============================================================================

class MultiScaleP2RLoss(nn.Module):
    """
    Loss P2R Multi-Scala per V8.
    
    Idea: calcola count loss a multiple risoluzioni (1x, 2x, 4x pooling).
    Questo forza il modello a essere consistente a diverse scale,
    migliorando la localizzazione e riducendo errori sistematici.
    
    Componenti:
    1. Count Loss Multi-Scale: MAE a scale [1, 2, 4]
    2. Spatial Loss: localizzazione predizioni  
    3. Scale Loss: regolarizzazione log_scale
    """
    
    def __init__(
        self,
        scales=[1, 2, 4],
        scale_weights=[1.0, 0.5, 0.25],
        count_weight=2.0,
        spatial_weight=0.15,
        scale_loss_weight=0.5,
        min_radius=8.0,
        max_radius=64.0,
    ):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_loss_weight = scale_loss_weight
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Normalizza pesi
        total_weight = sum(scale_weights)
        self.scale_weights = [w / total_weight for w in scale_weights]
    
    def forward(self, pred, points_list, cell_area, H_in, W_in, log_scale=None):
        """
        Args:
            pred: [B, 1, H, W] density predictions
            points_list: lista di tensori [N_i, 2] con coordinate GT
            cell_area: area della cella per scaling count
            H_in, W_in: dimensioni input originale
            log_scale: parametro scala (opzionale)
        """
        B, _, H, W = pred.shape
        device = pred.device
        
        losses = {}
        
        # =====================================================================
        # MULTI-SCALE COUNT LOSS
        # =====================================================================
        total_count_loss = torch.tensor(0.0, device=device)
        
        gt_counts = []
        pred_counts_scale1 = []
        
        for scale_idx, (scale, weight) in enumerate(zip(self.scales, self.scale_weights)):
            scale_count_losses = []
            
            if scale == 1:
                # Scala originale
                pred_scaled = pred
                cell_area_scaled = cell_area
            else:
                # Pooling per ridurre risoluzione
                pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
                # Il count totale deve essere preservato, quindi moltiplichiamo per scale^2
                pred_scaled = pred_scaled * (scale ** 2)
                cell_area_scaled = cell_area
            
            for i, pts in enumerate(points_list):
                gt = len(pts)
                pred_count = pred_scaled[i].sum() / cell_area_scaled
                
                scale_count_losses.append(torch.abs(pred_count - gt))
                
                # Salva per metriche (solo scala 1)
                if scale == 1:
                    gt_counts.append(gt)
                    pred_counts_scale1.append(pred_count.item())
            
            scale_loss = torch.stack(scale_count_losses).mean()
            total_count_loss = total_count_loss + weight * scale_loss
            
            losses[f'count_s{scale}'] = scale_loss.item()
        
        losses['count'] = total_count_loss
        
        # =====================================================================
        # SPATIAL LOSS (solo a scala 1)
        # =====================================================================
        spatial_losses = []
        
        for i, pts in enumerate(points_list):
            if len(pts) == 0:
                spatial_losses.append(torch.tensor(0.0, device=device))
                continue
            
            # Crea target gaussiano
            target = torch.zeros(H, W, device=device)
            scale_h = H / H_in
            scale_w = W / W_in
            
            for pt in pts:
                x = int((pt[0] * scale_w).clamp(0, W-1).item())
                y = int((pt[1] * scale_h).clamp(0, H-1).item())
                
                # Gaussiana 5x5
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            dist = (dx*dx + dy*dy) ** 0.5
                            target[ny, nx] += np.exp(-dist / 2)
            
            if target.sum() > 0:
                target = target / target.sum()
                pred_norm = pred[i, 0] / (pred[i, 0].sum() + 1e-8)
                spatial_losses.append(F.mse_loss(pred_norm, target))
            else:
                spatial_losses.append(torch.tensor(0.0, device=device))
        
        losses['spatial'] = torch.stack(spatial_losses).mean()
        
        # =====================================================================
        # SCALE REGULARIZATION
        # =====================================================================
        if log_scale is not None:
            scale_val = torch.exp(log_scale)
            scale_penalty = F.relu(self.min_radius - scale_val) + F.relu(scale_val - self.max_radius)
            losses['scale_reg'] = scale_penalty.mean()
        else:
            losses['scale_reg'] = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # TOTAL
        # =====================================================================
        total = (
            self.count_weight * losses['count'] +
            self.spatial_weight * losses['spatial'] +
            self.scale_loss_weight * losses['scale_reg']
        )
        
        losses['total'] = total
        losses['gt_counts'] = gt_counts
        losses['pred_counts'] = pred_counts_scale1
        
        return losses


# =============================================================================
# ENHANCED TRANSFORMS - FIX V8.1 per device mismatch
# =============================================================================

class EnhancedTransformsV8:
    """
    Augmentation potenziate per V8.
    
    Nota: Questo wrapper viene usato DOPO le transforms standard del progetto.
    Applica augmentation addizionali solo su immagini (non su density/points).
    
    FIX V8.1: mean/std vengono spostati sul device corretto dell'immagine
    """
    
    def __init__(self, cfg):
        data_cfg = cfg.get('DATA', {})
        
        # Color jitter
        cj_cfg = data_cfg.get('COLOR_JITTER', {})
        self.use_color_jitter = cj_cfg.get('ENABLED', False) if isinstance(cj_cfg, dict) else False
        if self.use_color_jitter:
            self.color_jitter = T.ColorJitter(
                brightness=cj_cfg.get('BRIGHTNESS', 0.2),
                contrast=cj_cfg.get('CONTRAST', 0.2),
                saturation=cj_cfg.get('SATURATION', 0.2),
                hue=cj_cfg.get('HUE', 0.1),
            )
        
        # Random grayscale
        self.gray_prob = data_cfg.get('RANDOM_GRAY_PROB', 0.0)
        
        # Gaussian blur
        blur_cfg = data_cfg.get('GAUSSIAN_BLUR', {})
        self.use_blur = blur_cfg.get('ENABLED', False) if isinstance(blur_cfg, dict) else False
        self.blur_prob = blur_cfg.get('PROB', 0.1) if isinstance(blur_cfg, dict) else 0.1
        self.blur_kernels = blur_cfg.get('KERNEL_SIZE', [3, 5]) if isinstance(blur_cfg, dict) else [3, 5]
        
        # Pre-registra mean/std come tensori base (verranno spostati su device al primo uso)
        self._mean_base = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std_base = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._device_cache = None
        self._mean = None
        self._std = None
    
    def _ensure_device(self, device):
        """Sposta mean/std sul device corretto (con caching)."""
        if self._device_cache != device:
            self._mean = self._mean_base.to(device)
            self._std = self._std_base.to(device)
            self._device_cache = device
    
    def __call__(self, image):
        """
        Applica augmentation solo sull'immagine.
        Input: Tensor [C, H, W] normalizzato (può essere su GPU)
        Output: Tensor [C, H, W] con augmentation (stesso device dell'input)
        """
        device = image.device
        
        # Sposta mean/std sul device corretto (cached)
        self._ensure_device(device)
        
        # Denormalizza per applicare trasformazioni
        image = image * self._std + self._mean
        image = image.clamp(0, 1)
        
        # Color jitter (richiede PIL, quindi sposta su CPU temporaneamente)
        if self.use_color_jitter and random.random() < 0.8:
            image_cpu = image.cpu()
            image_pil = TF.to_pil_image(image_cpu)
            image_pil = self.color_jitter(image_pil)
            image = TF.to_tensor(image_pil).to(device)
        
        # Random grayscale
        if self.gray_prob > 0 and random.random() < self.gray_prob:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        
        # Gaussian blur
        if self.use_blur and random.random() < self.blur_prob:
            kernel_size = random.choice(self.blur_kernels)
            image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])
        
        # Rinormalizza (usa mean/std già sul device corretto)
        image = (image - self._mean) / self._std
        
        return image


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, default_down):
    """Validazione."""
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
# TRAINING
# =============================================================================

def train_epoch(
    model, train_loader, optimizer, criterion, 
    device, default_down, epoch, 
    grad_clip=1.0, augment_fn=None
):
    """Training di una epoca con multi-scale loss."""
    model.train()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    count_s1_meter = AverageMeter()
    count_s2_meter = AverageMeter()
    count_s4_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Stage2 V8 [Ep {epoch}]")
    
    for images, densities, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        # Applica augmentation addizionale (V8)
        # Nota: images è già su device, augment_fn gestisce il device correttamente
        if augment_fn is not None:
            images = torch.stack([augment_fn(img) for img in images])
        
        _, _, H_in, W_in = images.shape
        
        # Forward
        outputs = model(images)
        pred = outputs['p2r_density']
        log_scale = outputs.get('log_scale', None)
        
        # Canonicalize
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Multi-scale loss
        losses = criterion(pred, points_list, cell_area, H_in, W_in, log_scale)
        loss = losses['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Metriche
        mae = np.mean([abs(p - g) for p, g in zip(losses['pred_counts'], losses['gt_counts'])])
        
        loss_meter.update(loss.item())
        mae_meter.update(mae)
        
        # Loss per scala
        if 'count_s1' in losses:
            count_s1_meter.update(losses['count_s1'])
        if 'count_s2' in losses:
            count_s2_meter.update(losses['count_s2'])
        if 'count_s4' in losses:
            count_s4_meter.update(losses['count_s4'])
        
        pbar.set_postfix({
            'L': f"{loss_meter.avg:.3f}",
            'MAE': f"{mae_meter.avg:.1f}",
            's1': f"{count_s1_meter.avg:.1f}" if count_s1_meter.count > 0 else "-",
        })
    
    return {
        'loss': loss_meter.avg, 
        'mae': mae_meter.avg,
        'count_s1': count_s1_meter.avg,
        'count_s2': count_s2_meter.avg,
        'count_s4': count_s4_meter.avg,
    }


def get_lr(optimizer):
    """Ottieni LR corrente."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Carica config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    # Parametri
    data_cfg = cfg['DATA']
    p2r_cfg = cfg['OPTIM_P2R']
    loss_cfg = cfg['P2R_LOSS']
    
    epochs = p2r_cfg.get('EPOCHS', 5000)
    patience = p2r_cfg.get('EARLY_STOPPING_PATIENCE', 200)
    warmup_epochs = p2r_cfg.get('WARMUP_EPOCHS', 50)
    grad_clip = p2r_cfg.get('GRAD_CLIP', 1.0)
    lr = float(p2r_cfg.get('LR', 5e-5))
    lr_backbone = float(p2r_cfg.get('LR_BACKBONE', 1e-6))
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    # Multi-scale config
    use_multi_scale = loss_cfg.get('USE_MULTI_SCALE', True)
    scales = loss_cfg.get('SCALES', [1, 2, 4])
    scale_weights = loss_cfg.get('SCALE_WEIGHTS', [1.0, 0.5, 0.25])
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip_v8')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V8.1 - Multi-Scale Loss + Augmentation (FIXED)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Warmup: {warmup_epochs}")
    print(f"LR: {lr}, LR backbone: {lr_backbone}")
    print(f"Patience: {patience}")
    print(f"Grad clip: {grad_clip}")
    print(f"Multi-scale: {use_multi_scale}")
    if use_multi_scale:
        print(f"  Scales: {scales}")
        print(f"  Weights: {scale_weights}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Dataset
    from datasets.transforms import build_transforms
    
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    DatasetClass = get_dataset(cfg["DATASET"])
    train_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf
    )
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=p2r_cfg.get('BATCH_SIZE', 8),
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
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Augmentation addizionale V8
    augment_fn = EnhancedTransformsV8(cfg)
    print(f"\nV8 Augmentation:")
    print(f"  Color Jitter: {augment_fn.use_color_jitter}")
    print(f"  Grayscale prob: {augment_fn.gray_prob}")
    print(f"  Blur: {augment_fn.use_blur} (prob={augment_fn.blur_prob})")
    
    # Modello
    bin_config = cfg["BINS_CONFIG"][cfg["DATASET"]]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=False,
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
    # FIX V8.1: Carica Stage 1 checkpoint con priorità corretta
    # =========================================================================
    # Cerca checkpoint in ordine di priorità:
    # 1. best_model.pth (output di Stage 1)
    # 2. stage1_best.pth (nome alternativo)
    # 3. Prova anche nella cartella V7 se non trovato
    
    stage1_loaded = False
    stage1_candidates = [
        os.path.join(output_dir, "best_model.pth"),           # Nome corretto Stage 1
        os.path.join(output_dir, "stage1_best.pth"),          # Nome alternativo
        os.path.join(output_dir, "last.pth"),                 # Last checkpoint
    ]
    
    # Aggiungi anche path V7 come fallback
    v7_dir = os.path.join(cfg["EXP"]["OUT_DIR"], "shha_target60_v7")
    stage1_candidates.extend([
        os.path.join(v7_dir, "best_model.pth"),
        os.path.join(v7_dir, "stage1_best.pth"),
    ])
    
    for stage1_path in stage1_candidates:
        if os.path.isfile(stage1_path):
            try:
                state = torch.load(stage1_path, map_location=device)
                if "model" in state:
                    state = state["model"]
                model.load_state_dict(state, strict=False)
                print(f"✅ Caricato Stage 1 da: {stage1_path}")
                stage1_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Errore caricamento {stage1_path}: {e}")
                continue
    
    if not stage1_loaded:
        print("⚠️ Stage 1 non trovato, training da zero")
        print(f"   Cercato in: {stage1_candidates[:3]}")
    
    # Setup optimizer
    print("\n🔧 Setup parametri:")
    param_groups = []
    
    # Backbone
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
        print("   Backbone: FROZEN")
    
    # P2R head
    p2r_params = [p for n, p in model.named_parameters() if 'p2r_head' in n]
    if p2r_params:
        param_groups.append({
            'params': p2r_params,
            'lr': lr,
            'name': 'p2r_head'
        })
        print(f"   P2R head: LR={lr}")
    
    # ZIP head (congelato in Stage 2)
    for n, p in model.named_parameters():
        if 'zip_head' in n:
            p.requires_grad = False
    print("   ZIP head: FROZEN")
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n   Trainabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=p2r_cfg.get('WEIGHT_DECAY', 1e-4))
    
    # Scheduler con warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Multi-scale loss
    criterion = MultiScaleP2RLoss(
        scales=scales if use_multi_scale else [1],
        scale_weights=scale_weights if use_multi_scale else [1.0],
        count_weight=loss_cfg.get('COUNT_WEIGHT', 2.0),
        spatial_weight=loss_cfg.get('SPATIAL_WEIGHT', 0.15),
        scale_loss_weight=loss_cfg.get('SCALE_WEIGHT', 0.5),
        min_radius=loss_cfg.get('MIN_RADIUS', 8.0),
        max_radius=loss_cfg.get('MAX_RADIUS', 64.0),
    )
    
    # Resume se richiesto
    start_epoch = 1
    best_mae = float('inf')
    no_improve_count = 0
    
    checkpoint_path = os.path.join(output_dir, "stage2_last.pth")
    if args.resume and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        no_improve_count = ckpt.get('no_improve_count', 0)
        print(f"✅ Resumed from epoch {start_epoch-1}, best MAE: {best_mae:.2f}")
    
    # Valutazione iniziale
    print("\n📋 Valutazione iniziale:")
    val_results = validate(model, val_loader, device, default_down)
    print(f"   MAE: {val_results['mae']:.2f}, RMSE: {val_results['rmse']:.2f}")
    
    if val_results['mae'] < best_mae:
        best_mae = val_results['mae']
    
    # Training loop
    print(f"\n🚀 START Training: {epochs} epochs")
    print(f"   Baseline MAE: {best_mae:.2f}")
    print(f"   Target: MAE < 65.0")
    print()
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion,
            device, default_down, epoch, grad_clip,
            augment_fn=augment_fn
        )
        
        scheduler.step()
        
        # Validate
        if epoch % p2r_cfg.get('VAL_INTERVAL', 5) == 0 or epoch == 1:
            val_results = validate(model, val_loader, device, default_down)
            
            current_lr = get_lr(optimizer)
            mae = val_results['mae']
            
            # Log
            improved = mae < best_mae
            status = "✅ NEW BEST" if improved else ""
            
            print(f"Epoch {epoch:4d} | "
                  f"Train MAE: {train_results['mae']:.1f} | "
                  f"Val MAE: {mae:.2f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Best: {best_mae:.2f} {status}")
            
            if improved:
                best_mae = mae
                no_improve_count = 0
                
                # Salva best
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'mae': mae,
                    'rmse': val_results['rmse'],
                }, os.path.join(output_dir, "stage2_best.pth"))
            else:
                no_improve_count += p2r_cfg.get('VAL_INTERVAL', 5)
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Salva checkpoint periodico
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'no_improve_count': no_improve_count,
            }, os.path.join(output_dir, "stage2_last.pth"))
    
    # Risultati finali
    print("\n" + "=" * 60)
    print("🏁 STAGE 2 V8.1 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_best.pth")
    
    if best_mae < 65:
        print(f"   🎉 OBIETTIVO RAGGIUNTO! MAE < 65")
    elif best_mae < 68:
        print(f"   ✅ Miglioramento rispetto a V7 (68.97)")
    else:
        print(f"   ⚠️ Performance simile a V7")
    
    print("=" * 60)
    print(f"\n📌 Prossimo step: python train_stage3.py per bias correction")


if __name__ == "__main__":
    main()