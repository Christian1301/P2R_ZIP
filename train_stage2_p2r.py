#!/usr/bin/env python3
"""
Stage 2 V8 - P2R Training con Augmentation Potenziata

MIGLIORAMENTI DA V7:
1. 5000 epoche (era 3000)
2. Data augmentation aggressiva:
   - Scale range più ampio [0.4, 1.0]
   - Color jitter
   - Random grayscale
   - Gaussian blur
3. Warmup epochs
4. Gradient clipping
5. Patience aumentata (200)

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

# Torchvision per augmentation
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


class EnhancedTransforms:
    """
    Augmentation potenziate per Stage 2 V8.
    
    Include:
    - Random crop con scala variabile
    - Color jitter
    - Horizontal flip
    - Random grayscale
    - Gaussian blur (opzionale)
    """
    
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.cfg = cfg
        
        data_cfg = cfg.get('DATA', {})
        
        # Parametri crop
        self.crop_size = data_cfg.get('CROP_SIZE', 384)
        self.crop_scale = data_cfg.get('CROP_SCALE', [0.4, 1.0])
        
        # Color jitter
        cj_cfg = data_cfg.get('COLOR_JITTER', {})
        self.use_color_jitter = cj_cfg.get('ENABLED', True) if isinstance(cj_cfg, dict) else bool(cj_cfg)
        if self.use_color_jitter:
            self.color_jitter = T.ColorJitter(
                brightness=cj_cfg.get('BRIGHTNESS', 0.2) if isinstance(cj_cfg, dict) else 0.2,
                contrast=cj_cfg.get('CONTRAST', 0.2) if isinstance(cj_cfg, dict) else 0.2,
                saturation=cj_cfg.get('SATURATION', 0.2) if isinstance(cj_cfg, dict) else 0.2,
                hue=cj_cfg.get('HUE', 0.1) if isinstance(cj_cfg, dict) else 0.1,
            )
        
        # Horizontal flip
        self.use_hflip = data_cfg.get('HORIZONTAL_FLIP', True)
        
        # Random grayscale
        self.gray_prob = data_cfg.get('RANDOM_GRAY_PROB', 0.1)
        
        # Gaussian blur
        blur_cfg = data_cfg.get('GAUSSIAN_BLUR', {})
        self.use_blur = blur_cfg.get('ENABLED', False) if isinstance(blur_cfg, dict) else False
        self.blur_prob = blur_cfg.get('PROB', 0.1) if isinstance(blur_cfg, dict) else 0.1
        self.blur_kernel = blur_cfg.get('KERNEL_SIZE', [3, 5]) if isinstance(blur_cfg, dict) else [3, 5]
        
        # Normalizzazione
        self.norm_mean = data_cfg.get('NORM_MEAN', [0.485, 0.456, 0.406])
        self.norm_std = data_cfg.get('NORM_STD', [0.229, 0.224, 0.225])
        self.normalize = T.Normalize(mean=self.norm_mean, std=self.norm_std)
    
    def __call__(self, image, density, points):
        """
        Applica trasformazioni a immagine, density map e punti.
        
        Args:
            image: PIL Image o Tensor
            density: Tensor density map
            points: Tensor [N, 2] coordinate punti (x, y)
        
        Returns:
            image, density, points trasformati
        """
        # Converti a tensor se necessario
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        
        if self.is_train:
            # 1. Random crop
            image, density, points = self._random_crop(image, density, points)
            
            # 2. Color jitter (solo su immagine)
            if self.use_color_jitter and random.random() < 0.8:
                # Converti a PIL per color jitter
                image_pil = TF.to_pil_image(image)
                image_pil = self.color_jitter(image_pil)
                image = TF.to_tensor(image_pil)
            
            # 3. Horizontal flip
            if self.use_hflip and random.random() < 0.5:
                image, density, points = self._hflip(image, density, points)
            
            # 4. Random grayscale
            if random.random() < self.gray_prob:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
            
            # 5. Gaussian blur
            if self.use_blur and random.random() < self.blur_prob:
                kernel_size = random.choice(self.blur_kernel)
                image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])
        
        # Normalizza
        image = self.normalize(image)
        
        return image, density, points
    
    def _random_crop(self, image, density, points):
        """Random crop con scala variabile."""
        _, H, W = image.shape
        
        # Scala random
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_h = int(self.crop_size * scale)
        crop_w = int(self.crop_size * scale)
        
        # Limita alle dimensioni immagine
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        
        # Posizione random
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        
        # Crop immagine
        image = image[:, top:top+crop_h, left:left+crop_w]
        
        # Crop density (potrebbe avere dimensioni diverse)
        dH, dW = density.shape[-2:]
        scale_h = dH / H
        scale_w = dW / W
        d_top = int(top * scale_h)
        d_left = int(left * scale_w)
        d_crop_h = int(crop_h * scale_h)
        d_crop_w = int(crop_w * scale_w)
        density = density[..., d_top:d_top+d_crop_h, d_left:d_left+d_crop_w]
        
        # Filtra e trasla punti
        if len(points) > 0:
            # Filtra punti nel crop
            mask = (
                (points[:, 0] >= left) & (points[:, 0] < left + crop_w) &
                (points[:, 1] >= top) & (points[:, 1] < top + crop_h)
            )
            points = points[mask]
            
            # Trasla
            if len(points) > 0:
                points = points.clone()
                points[:, 0] -= left
                points[:, 1] -= top
        
        # Resize a crop_size
        image = F.interpolate(
            image.unsqueeze(0), 
            size=(self.crop_size, self.crop_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        if density.dim() == 2:
            density = density.unsqueeze(0).unsqueeze(0)
        elif density.dim() == 3:
            density = density.unsqueeze(0)
        
        # Scala density per preservare count
        old_sum = density.sum()
        density = F.interpolate(
            density,
            size=(self.crop_size // 8, self.crop_size // 8),  # P2R downsample
            mode='bilinear',
            align_corners=False
        )
        if old_sum > 0:
            density = density * (old_sum / (density.sum() + 1e-8))
        density = density.squeeze()
        
        # Scala punti
        if len(points) > 0:
            points = points.clone().float()
            points[:, 0] = points[:, 0] * (self.crop_size / crop_w)
            points[:, 1] = points[:, 1] * (self.crop_size / crop_h)
        
        return image, density, points
    
    def _hflip(self, image, density, points):
        """Horizontal flip."""
        image = TF.hflip(image)
        
        if density.dim() == 2:
            density = density.flip(-1)
        else:
            density = density.flip(-1)
        
        if len(points) > 0:
            _, _, W = image.shape
            points = points.clone()
            points[:, 0] = W - 1 - points[:, 0]
        
        return image, density, points


class P2RLossV8(nn.Module):
    """
    Loss P2R per Stage 2 V8.
    
    Componenti:
    1. Count Loss: MAE sul conteggio totale
    2. Spatial Loss: localizzazione predizioni
    3. Scale Loss: regolarizzazione log_scale
    """
    
    def __init__(
        self,
        count_weight=2.0,
        spatial_weight=0.15,
        scale_weight=0.5,
        min_radius=8.0,
        max_radius=64.0,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def forward(self, pred, points_list, cell_area, log_scale=None):
        """
        Args:
            pred: [B, 1, H, W] density predictions
            points_list: lista di tensori [N_i, 2] con coordinate GT
            cell_area: area della cella per scaling count
            log_scale: parametro scala (opzionale)
        """
        B = pred.shape[0]
        device = pred.device
        
        losses = {}
        
        # Count loss
        count_losses = []
        gt_counts = []
        pred_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            pred_count = pred[i].sum() / cell_area
            count_losses.append(torch.abs(pred_count - gt))
            gt_counts.append(gt)
            pred_counts.append(pred_count.item())
        
        losses['count'] = torch.stack(count_losses).mean()
        
        # Spatial loss (semplificata)
        H, W = pred.shape[-2:]
        spatial_losses = []
        
        for i, pts in enumerate(points_list):
            if len(pts) == 0:
                spatial_losses.append(torch.tensor(0.0, device=device))
                continue
            
            # Crea target gaussiano
            target = torch.zeros(H, W, device=device)
            for pt in pts:
                x = int((pt[0] / cell_area).clamp(0, W-1).item())
                y = int((pt[1] / cell_area).clamp(0, H-1).item())
                # Gaussiana semplice
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
        
        # Scale loss
        if log_scale is not None:
            scale = torch.exp(log_scale)
            scale_penalty = F.relu(self.min_radius - scale) + F.relu(scale - self.max_radius)
            losses['scale'] = scale_penalty.mean()
        else:
            losses['scale'] = torch.tensor(0.0, device=device)
        
        # Total
        total = (
            self.count_weight * losses['count'] +
            self.spatial_weight * losses['spatial'] +
            self.scale_weight * losses['scale']
        )
        
        losses['total'] = total
        losses['gt_counts'] = gt_counts
        losses['pred_counts'] = pred_counts
        
        return losses


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


def train_epoch(model, train_loader, optimizer, criterion, device, default_down, epoch, grad_clip=1.0):
    """Training di una epoca."""
    model.train()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Stage2 V8 [Ep {epoch}]")
    
    for images, densities, points in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        _, _, H_in, W_in = images.shape
        
        # Forward
        outputs = model(images)
        pred = outputs['p2r_density']
        log_scale = outputs.get('log_scale', None)
        
        # Canonicalize
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Loss
        losses = criterion(pred, points_list, cell_area, log_scale)
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
        
        pbar.set_postfix({
            'L': f"{loss_meter.avg:.3f}",
            'MAE': f"{mae_meter.avg:.1f}"
        })
    
    return {'loss': loss_meter.avg, 'mae': mae_meter.avg}


def get_lr(optimizer):
    """Ottieni LR corrente."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
    
    run_name = cfg.get('RUN_NAME', 'p2r_zip_v8')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Stage 2 V8 - P2R Training Potenziato")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Warmup: {warmup_epochs}")
    print(f"LR: {lr}, LR backbone: {lr_backbone}")
    print(f"Patience: {patience}")
    print(f"Grad clip: {grad_clip}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Dataset con augmentation potenziate
    # Nota: usiamo le transform standard del progetto, le augmentation
    # sono configurate nel config
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
    
    # Carica Stage 1 checkpoint
    stage1_path = os.path.join(output_dir, "best_model.pth")
    
    # Prova anche nella cartella V7 se V8 non ha Stage 1
    if not os.path.isfile(stage1_path):
        v7_path = os.path.join(cfg["EXP"]["OUT_DIR"], "shha_target60_v7", "stage1_best.pth")
        if os.path.isfile(v7_path):
            stage1_path = v7_path
            print(f"⚠️ Usando Stage 1 da V7: {v7_path}")
    
    if os.path.isfile(stage1_path):
        state = torch.load(stage1_path, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"✅ Caricato Stage 1 da {stage1_path}")
    else:
        print(f"⚠️ Stage 1 non trovato, training da zero")
    
    # Setup optimizer
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
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss
    criterion = P2RLossV8(
        count_weight=loss_cfg.get('COUNT_WEIGHT', 2.0),
        spatial_weight=loss_cfg.get('SPATIAL_WEIGHT', 0.15),
        scale_weight=loss_cfg.get('SCALE_WEIGHT', 0.5),
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
    print(f"   Target: MAE < 65.0")
    print()
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion,
            device, default_down, epoch, grad_clip
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
    print("🏁 STAGE 2 V8 COMPLETATO")
    print("=" * 60)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_best.pth")
    
    if best_mae < 65:
        print(f"   🎉 OBIETTIVO RAGGIUNTO! MAE < 65")
    elif best_mae < 68:
        print(f"   ✅ Buon miglioramento rispetto a V7 (68.97)")
    else:
        print(f"   ⚠️ Margine di miglioramento limitato")
    
    print("=" * 60)


if __name__ == "__main__":
    main()