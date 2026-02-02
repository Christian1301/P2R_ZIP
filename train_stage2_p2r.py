#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Stage 2 BYPASS - P2R con ZIP come Feature

ARCHITETTURA:
- Backbone: FROZEN (pesi da Stage 1)
- ZIP Head: FROZEN (pesi da Stage 1, produce π e λ)
- P2R Head: TRAINING (input 514 canali = 512 features + π + λ)

DIFFERENZA DA STAGE 2 ORIGINALE:
- NO hard gating (features × mask)
- π e λ concatenati alle features come canali extra
- P2R impara SE e COME usare le info ZIP

TARGET:
- MAE < 60 su ShanghaiTech Part A
- Baseline garantita: ~69 (P2R standalone)

USO:
    python train_stage2_bypass.py --config config.yaml
    
    # Oppure con checkpoint Stage 1 custom:
    python train_stage2_bypass.py --config config.yaml --stage1-ckpt exp/shha_v15/best_model.pth
"""

import os
import sys
import yaml
import json
import argparse
import random
import math
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
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Import moduli progetto
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn, init_seeds, canonicalize_p2r_grid


# =============================================================================
# DATA AUGMENTATION AGGRESSIVA
# =============================================================================

class AggressiveTransform:
    """
    Data augmentation aggressiva per crowd counting.
    Trasforma sia immagine che punti annotations.
    """
    
    def __init__(
        self,
        crop_size=(384, 384),
        scale_range=(0.8, 1.2),
        rotation_range=15,
        flip_prob=0.5,
        color_jitter=True,
        random_erasing=True,
        erasing_prob=0.3,
    ):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing
        self.erasing_prob = erasing_prob
        
        if color_jitter:
            self.color_transform = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1,
            )
        
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image, points):
        """
        Args:
            image: PIL Image
            points: numpy array [N, 2] con coordinate (x, y)
        Returns:
            image_tensor, points_tensor
        """
        W_orig, H_orig = image.size
        points = np.array(points) if len(points) > 0 else np.zeros((0, 2))
        
        # 1. Random Scale
        scale = random.uniform(*self.scale_range)
        new_W, new_H = int(W_orig * scale), int(H_orig * scale)
        image = image.resize((new_W, new_H), Image.BILINEAR)
        if len(points) > 0:
            points = points * scale
        
        # 2. Random Rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle, expand=True, fill=0)
            
            if len(points) > 0:
                cx, cy = new_W / 2, new_H / 2
                new_W_rot, new_H_rot = image.size
                cx_new, cy_new = new_W_rot / 2, new_H_rot / 2
                
                rad = math.radians(-angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                
                pts_centered = points - np.array([cx, cy])
                pts_rotated = np.zeros_like(pts_centered)
                pts_rotated[:, 0] = pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a
                pts_rotated[:, 1] = pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a
                points = pts_rotated + np.array([cx_new, cy_new])
                new_W, new_H = new_W_rot, new_H_rot
        
        # 3. Random Crop
        crop_h, crop_w = self.crop_size
        pad_h, pad_w = max(0, crop_h - new_H), max(0, crop_w - new_W)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            new_W += pad_w
            new_H += pad_h
        
        top = random.randint(0, new_H - crop_h)
        left = random.randint(0, new_W - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        
        if len(points) > 0:
            points = points - np.array([left, top])
            valid_mask = (
                (points[:, 0] >= 0) & (points[:, 0] < crop_w) &
                (points[:, 1] >= 0) & (points[:, 1] < crop_h)
            )
            points = points[valid_mask]
        
        # 4. Random Horizontal Flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            if len(points) > 0:
                points[:, 0] = crop_w - 1 - points[:, 0]
        
        # 5. Color Jitter
        if self.color_jitter:
            image = self.color_transform(image)
        
        # 6. Random Grayscale
        if random.random() < 0.1:
            image = TF.to_grayscale(image, num_output_channels=3)
        
        # 7. To Tensor + Normalize
        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        
        # 8. Random Erasing
        if self.random_erasing and random.random() < self.erasing_prob:
            erase_h = random.randint(crop_h // 8, crop_h // 4)
            erase_w = random.randint(crop_w // 8, crop_w // 4)
            erase_top = random.randint(0, crop_h - erase_h)
            erase_left = random.randint(0, crop_w - erase_w)
            
            noise = torch.randn(3, erase_h, erase_w)
            image_tensor[:, erase_top:erase_top+erase_h, erase_left:erase_left+erase_w] = noise
            
            if len(points) > 0:
                erase_mask = ~(
                    (points[:, 0] >= erase_left) & (points[:, 0] < erase_left + erase_w) &
                    (points[:, 1] >= erase_top) & (points[:, 1] < erase_top + erase_h)
                )
                points = points[erase_mask]
        
        points_tensor = torch.tensor(points, dtype=torch.float32) if len(points) > 0 else torch.zeros((0, 2), dtype=torch.float32)
        return image_tensor, points_tensor


class SimpleValTransform:
    """Transform semplice per validation (no augmentation)."""
    
    def __init__(self):
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image, points):
        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        points_tensor = torch.tensor(np.array(points), dtype=torch.float32) if len(points) > 0 else torch.zeros((0, 2), dtype=torch.float32)
        return image_tensor, points_tensor


class AugmentedDatasetWrapper:
    """Wrapper che applica transforms custom al dataset."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        
        if isinstance(data, dict):
            image = data.get('image')
            points = data.get('points', [])
        else:
            image = data[0]
            points = data[2] if len(data) > 2 else []
        
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = TF.to_pil_image(image)
        
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        elif isinstance(points, list):
            points = np.array(points) if len(points) > 0 else np.zeros((0, 2))
        
        image_tensor, points_tensor = self.transform(image, points)
        gt_density = torch.zeros(1, 1, 1)  # Dummy, non usato
        
        return image_tensor, gt_density, points_tensor


def collate_fn_augmented(batch):
    """Collate function per dataset augmentato."""
    images = torch.stack([b[0] for b in batch])
    gt_densities = torch.stack([b[1] for b in batch])
    points_list = [b[2] for b in batch]
    return images, gt_densities, points_list


# =============================================================================
# CONFIG DEFAULTS
# =============================================================================

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
    'jhu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'nwpu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
}

DATASET_ALIASES = {
    'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
    'shhb': 'shhb', 'shanghaitechb': 'shhb',
    'ucf': 'ucf', 'ucfqnrf': 'ucf',
    'jhu': 'jhu', 'nwpu': 'nwpu',
}


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class BypassP2RLoss(nn.Module):
    """
    Loss per P2R con ZIP come feature.
    
    Combina:
    1. Count Loss (L1): supervisionato sui conteggi
    2. Spatial Loss: MSE sulla distribuzione normalizzata
    3. Scale Loss: penalizza errori relativi
    """
    
    def __init__(
        self,
        count_weight: float = 2.0,
        spatial_weight: float = 0.15,
        scale_weight: float = 0.5,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.spatial_weight = spatial_weight
        self.scale_weight = scale_weight
    
    def forward(
        self,
        pred_density: torch.Tensor,
        points_list: list,
        cell_area: float,
    ) -> dict:
        """
        Args:
            pred_density: [B, 1, H, W] predicted density
            points_list: Lista di tensori [N_i, 2] con coordinate punti
            cell_area: Area per cella per normalizzazione count
            
        Returns:
            dict con total_loss e metriche
        """
        B = pred_density.shape[0]
        device = pred_density.device
        
        count_losses = []
        scale_losses = []
        spatial_losses = []
        
        pred_counts = []
        gt_counts = []
        
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = pred_density[i].sum() / cell_area
            
            gt_counts.append(gt)
            pred_counts.append(pred_count.item())
            
            # 1. Count Loss (L1)
            count_loss = torch.abs(pred_count - gt)
            count_losses.append(count_loss)
            
            # 2. Scale Loss (errore relativo)
            if gt > 1:
                scale_loss = torch.abs(pred_count - gt) / (gt + 1e-6)
                scale_losses.append(scale_loss)
            else:
                scale_losses.append(torch.zeros(1, device=device).squeeze())
            
            # 3. Spatial Loss (per immagini non vuote)
            if gt > 0 and len(pts) > 0:
                H, W = pred_density.shape[2], pred_density.shape[3]
                gt_density = self._points_to_density(pts, H, W, cell_area, device)
                
                # Normalizza per confrontare forma
                pred_norm = pred_density[i] / (pred_density[i].sum() + 1e-8)
                gt_norm = gt_density / (gt_density.sum() + 1e-8)
                
                spatial_loss = F.mse_loss(pred_norm, gt_norm)
                spatial_losses.append(spatial_loss)
            else:
                # Immagini vuote: penalizza predizioni non-zero
                spatial_losses.append(pred_density[i].mean())
        
        # Aggrega
        count_loss = torch.stack(count_losses).mean()
        scale_loss = torch.stack(scale_losses).mean()
        spatial_loss = torch.stack(spatial_losses).mean()
        
        total_loss = (
            self.count_weight * count_loss +
            self.scale_weight * scale_loss +
            self.spatial_weight * spatial_loss
        )
        
        # MAE per monitoring
        mae = np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)])
        bias = sum(pred_counts) / max(sum(gt_counts), 1)
        
        return {
            'total_loss': total_loss,
            'count_loss': count_loss,
            'scale_loss': scale_loss,
            'spatial_loss': spatial_loss,
            'mae': mae,
            'bias': bias,
        }
    
    def _points_to_density(self, points, H, W, cell_area, device):
        """Genera GT density dai punti."""
        gt_density = torch.zeros(1, H, W, device=device)
        
        # Calcola downscale
        downscale = int(np.sqrt(cell_area))
        
        for pt in points:
            x = int(pt[0].item() / downscale) if torch.is_tensor(pt[0]) else int(pt[0] / downscale)
            y = int(pt[1].item() / downscale) if torch.is_tensor(pt[1]) else int(pt[1] / downscale)
            
            x = min(max(x, 0), W - 1)
            y = min(max(y, 0), H - 1)
            
            gt_density[0, y, x] += 1.0
        
        return gt_density


# =============================================================================
# UTILITIES
# =============================================================================

class AverageMeter:
    """Computa e memorizza media."""
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
    """Early stopping tracker."""
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


def get_lr_scheduler(optimizer, cfg, total_epochs):
    """Crea scheduler con warmup + cosine."""
    warmup_epochs = cfg.get('WARMUP_EPOCHS', 50)
    min_lr = cfg.get('MIN_LR', 1e-7)
    base_lr = cfg.get('LR', 5e-5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(min_lr / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    default_down: int,
    grad_clip: float = 1.0,
    scaler: GradScaler = None,
) -> dict:
    """Training di una epoca."""
    model.train()
    
    losses = AverageMeter()
    mae_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Train [Ep {epoch}]")
    
    for images, gt_density, points_list in pbar:
        images = images.to(device)
        points_list = [p.to(device) for p in points_list]
        
        optimizer.zero_grad()
        
        # Forward
        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            outputs = model(images)
            pred_density = outputs['p2r_density']
            
            # Canonicalize
            _, _, H_in, W_in = images.shape
            pred_density, down_tuple, _ = canonicalize_p2r_grid(
                pred_density, (H_in, W_in), default_down
            )
            cell_area = down_tuple[0] * down_tuple[1]
            
            # Loss
            loss_dict = criterion(pred_density, points_list, cell_area)
            loss = loss_dict['total_loss']
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update meters
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        mae_meter.update(loss_dict['mae'], batch_size)
        
        # Progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'MAE': f'{mae_meter.avg:.2f}',
            'scale': f'{model.p2r_head.get_scale():.2f}',
        })
    
    return {
        'loss': losses.avg,
        'mae': mae_meter.avg,
        'log_scale': model.p2r_head.get_log_scale(),
        'scale': model.p2r_head.get_scale(),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    default_down: int,
    max_size: int = 1536  # Patch size (max VRAM usage)
) -> dict:
    """
    Validazione con Sliding Window e Safe Padding.
    Gestisce immagini giganti e previene crash su patch troppo piccoli.
    """
    model.eval()
    
    all_preds = []
    all_gts = []
    pi_coverages = []
    
    # Fattore di downsample della backbone (VGG16 = 32)
    # Serve per il padding: input deve essere multiplo di questo per evitare errori di arrotondamento/pooling
    divisor = 32 
    
    for images, gt_density, points_list in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        pts = points_list[0]
        gt = len(pts)
        
        B, C, H, W = images.shape
        
        # SLIDING WINDOW LOGIC
        if H > max_size or W > max_size:
            pred_count = 0.0
            pi_cov_temp = []
            
            patch_h, patch_w = max_size, max_size
            
            for y in range(0, H, patch_h):
                for x in range(0, W, patch_w):
                    h_end = min(y + patch_h, H)
                    w_end = min(x + patch_w, W)
                    
                    patch = images[:, :, y:h_end, x:w_end]
                    
                    # --- SAFE PADDING START ---
                    # Calcola dimensioni attuali
                    curr_h, curr_w = patch.shape[2], patch.shape[3]
                    
                    # Calcola quanto padding serve per arrivare al prossimo multiplo di 32
                    # Questo previene il crash "Output size is too small" su strisce sottili
                    pad_h = (divisor - curr_h % divisor) % divisor
                    pad_w = (divisor - curr_w % divisor) % divisor
                    
                    if pad_h > 0 or pad_w > 0:
                        # Pad: (left, right, top, bottom)
                        patch = F.pad(patch, (0, pad_w, 0, pad_h))
                    # --- SAFE PADDING END ---

                    # Forward
                    outputs = model(patch)
                    p_density = outputs['p2r_density']
                    p_pi = outputs['pi_probs']
                    
                    # Canonicalize
                    # Qui è importante: usiamo le dimensioni PADDATE per calcolare l'area corretta
                    # perché la densità predetta copre anche l'area di padding.
                    # Poiché il padding è nero, la densità lì sarà ~0, quindi sommare tutto è sicuro.
                    _, down_tuple, _ = canonicalize_p2r_grid(
                        p_density, (curr_h + pad_h, curr_w + pad_w), default_down
                    )
                    cell_area = down_tuple[0] * down_tuple[1]
                    
                    # Accumula conteggio
                    pred_count += (p_density.sum() / cell_area).item()
                    
                    # Stats
                    pi_cov_temp.append((p_pi > 0.5).float().mean().item())
            
            pred = pred_count
            coverage = np.mean(pi_cov_temp) * 100
            
        else:
            # Immagine intera (se piccola) - Applichiamo lo stesso padding per sicurezza
            curr_h, curr_w = H, W
            pad_h = (divisor - curr_h % divisor) % divisor
            pad_w = (divisor - curr_w % divisor) % divisor
            
            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h))
                
            outputs = model(images)
            pred_density = outputs['p2r_density']
            pi_probs = outputs['pi_probs']
            
            # Canonicalize su dimensioni paddate
            _, down_tuple, _ = canonicalize_p2r_grid(
                pred_density, (curr_h + pad_h, curr_w + pad_w), default_down
            )
            cell_area = down_tuple[0] * down_tuple[1]
            
            pred = (pred_density.sum() / cell_area).item()
            coverage = (pi_probs > 0.5).float().mean().item() * 100

        all_gts.append(gt)
        all_preds.append(pred)
        pi_coverages.append(coverage)
    
    # Calcolo metriche finali
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    mae = np.mean(np.abs(all_preds - all_gts))
    rmse = np.sqrt(np.mean((all_preds - all_gts) ** 2))
    
    valid_mask = all_gts > 0
    bias = np.mean(all_preds[valid_mask] / all_gts[valid_mask]) if valid_mask.sum() > 0 else 1.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pi_coverage': np.mean(pi_coverages),
        'log_scale': model.p2r_head.get_log_scale(),
        'scale': model.p2r_head.get_scale(),
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_mae, output_dir, is_best=False, early_stopping=None):
    """Salva checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_mae': best_mae,
        'mae': metrics['mae'],
        'log_scale': model.p2r_head.get_log_scale(),
        'early_stopping': {
            'counter': early_stopping.counter,
            'best_score': early_stopping.best_score,
        } if early_stopping else None,
    }
    
    torch.save(checkpoint, os.path.join(output_dir, 'stage2_bypass_last.pth'))
    
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'stage2_bypass_best.pth'))
        print(f"💾 Best: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 BYPASS")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--stage1-ckpt', type=str, default=None,
                        help='Path esplicito al checkpoint Stage 1')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate nei layer intermedi P2R')
    parser.add_argument('--final-dropout', type=float, default=0.5,
                        help='Dropout rate nel layer finale P2R')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (default: usa config)')
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                        help='Usa data augmentation aggressiva')
    parser.add_argument('--no-augmentation', action='store_false', dest='use_augmentation',
                        help='Disabilita data augmentation aggressiva')
    args = parser.parse_args()
    
    # Carica config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(config.get('DEVICE', 'cuda') if torch.cuda.is_available() else 'cpu')
    init_seeds(config.get('SEED', 42))
    
    # Parsing dataset config
    dataset_section = config.get('DATASET', 'shha')
    data_section = config.get('DATA', {})
    
    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = {}
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
        data_cfg.update(dataset_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}
    
    normalized = ''.join(c for c in str(dataset_name_raw).lower() if c.isalnum())
    dataset_name = DATASET_ALIASES.get(normalized, str(dataset_name_raw).lower())
    
    # Defaults
    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]
    
    # Output directory
    run_name = config.get('RUN_NAME', 'bypass_run')
    exp_dir = config.get('EXP', {}).get('OUT_DIR', config.get('EXPERIMENT_DIR', 'exp'))
    output_dir = os.path.join(exp_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Bins config
    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"No bins config for dataset '{dataset_name}'")
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    print("=" * 70)
    print("🚀 STAGE 2 BYPASS - P2R con ZIP come Feature")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print("\n📐 Architettura:")
    print("   Backbone [512ch] ─┬─→ ZIP Head → π, λ ─┐")
    print("                     │                     │ concat")
    print("                     └─────────────────────┴─→ [514ch] → P2R → density")
    print("\n   ✓ Backbone: FROZEN")
    print("   ✓ ZIP Head: FROZEN")
    print("   ✓ P2R Head: TRAINING (514 input channels)")
    print("=" * 70)
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    zip_head_cfg = config.get('ZIP_HEAD', {})
    p2r_loss_cfg = config.get('P2R_LOSS', {})
    
    model = P2R_ZIP_Model(
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=config.get('MODEL', {}).get('BACKBONE', 'vgg16_bn'),
        freeze_backbone=True,
        freeze_zip=True,
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
        p2r_head_kwargs={
            'fea_channel': config.get('MODEL', {}).get('P2R_FEA_CHANNEL', 256),
            'log_scale_init': p2r_loss_cfg.get('LOG_SCALE_INIT', 4.0),
            'log_scale_clamp': tuple(p2r_loss_cfg.get('LOG_SCALE_CLAMP', [-2.0, 10.0])),
            'dropout_rate': args.dropout,
            'final_dropout_rate': args.final_dropout,
        },
    ).to(device)
    
    print(f"\n🛡️ Regularizzazione:")
    print(f"   Dropout: {args.dropout}")
    print(f"   Final Dropout: {args.final_dropout}")
    print(f"   Data Augmentation: {'ON' if args.use_augmentation else 'OFF'}")
    
    # =========================================================================
    # LOAD STAGE 1 CHECKPOINT
    # =========================================================================
    
    if args.stage1_ckpt:
        stage1_path = args.stage1_ckpt
    else:
        # Cerca automaticamente
        candidates = [
            os.path.join(output_dir, 'best_model.pth'),
            os.path.join(output_dir, 'last.pth'),
        ]
        stage1_path = None
        for c in candidates:
            if os.path.isfile(c):
                stage1_path = c
                break
    
    if stage1_path and os.path.isfile(stage1_path):
        print(f"\n📂 Caricamento Stage 1: {stage1_path}")
        model.load_stage1_checkpoint(stage1_path, device)
    else:
        print("\n⚠️ ATTENZIONE: Nessun checkpoint Stage 1 trovato!")
        print("   Il modello partirà con pesi random per backbone e ZIP head.")
        print("   Questo è sconsigliato - esegui prima Stage 1.")
    
    print(f"\n📊 P2R Head:")
    print(f"   Input channels: {model.p2r_head.in_channel}")
    print(f"   GroupNorm groups: {model.p2r_head._norm_groups}")
    print(f"   log_scale init: {model.p2r_head.get_log_scale():.4f}")
    print(f"   scale init: {model.p2r_head.get_scale():.4f}")
    
    # =========================================================================
    # DATASET
    # =========================================================================
    
    # =========================================================================
    # DATASET - con Data Augmentation Aggressiva se richiesto
    # =========================================================================
    
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    DatasetClass = get_dataset(dataset_name)
    
    if args.use_augmentation:
        # Dataset con augmentation aggressiva
        print("\n🔄 Usando Data Augmentation Aggressiva")
        
        # Carica dataset base senza transforms
        base_train = DatasetClass(
            root=data_cfg['ROOT'],
            split=data_cfg.get('TRAIN_SPLIT', 'train'),
            block_size=block_size,
            transforms=None,
        )
        base_val = DatasetClass(
            root=data_cfg['ROOT'],
            split=data_cfg.get('VAL_SPLIT', 'val'),
            block_size=block_size,
            transforms=None,
        )
        
        # Wrap con augmentation aggressiva
        train_transform = AggressiveTransform(
            crop_size=(384, 384),
            scale_range=(0.8, 1.2),
            rotation_range=15,
            flip_prob=0.5,
            color_jitter=True,
            random_erasing=True,
            erasing_prob=0.3,
        )
        val_transform = SimpleValTransform()
        
        train_dataset = AugmentedDatasetWrapper(base_train, train_transform)
        val_dataset = AugmentedDatasetWrapper(base_val, val_transform)
        
        # Usa collate_fn per augmented dataset
        train_collate = collate_fn_augmented
        val_collate = collate_fn_augmented
    else:
        # Dataset standard
        train_tf = build_transforms(data_cfg, is_train=True)
        val_tf = build_transforms(data_cfg, is_train=False)
        
        train_dataset = DatasetClass(
            root=data_cfg['ROOT'],
            split=data_cfg.get('TRAIN_SPLIT', 'train'),
            block_size=block_size,
            transforms=train_tf,
        )
        val_dataset = DatasetClass(
            root=data_cfg['ROOT'],
            split=data_cfg.get('VAL_SPLIT', 'val'),
            block_size=block_size,
            transforms=val_tf,
        )
        train_collate = collate_fn
        val_collate = collate_fn
    
    optim_cfg = config.get('OPTIM_P2R', {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg.get('BATCH_SIZE', 8),
        shuffle=True,
        num_workers=optim_cfg.get('NUM_WORKERS', 4),
        collate_fn=train_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate,
        pin_memory=True,
    )
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Block size: {block_size}")
    print(f"   Default downsample: {default_down}")
    if args.use_augmentation:
        print(f"   Augmentation: Scale, Rotation, Flip, ColorJitter, RandomErasing")
    
    # =========================================================================
    # LOSS, OPTIMIZER, SCHEDULER
    # =========================================================================
    
    criterion = BypassP2RLoss(
        count_weight=p2r_loss_cfg.get('COUNT_WEIGHT', 2.0),
        spatial_weight=p2r_loss_cfg.get('SPATIAL_WEIGHT', 0.15),
        scale_weight=p2r_loss_cfg.get('SCALE_WEIGHT', 0.5),
    )
    
    # Solo parametri P2R (backbone e ZIP sono frozen)
    trainable_params = [p for p in model.p2r_head.parameters() if p.requires_grad]
    
    # LR differenziato per log_scale
    log_scale_params = []
    other_params = []
    for name, param in model.p2r_head.named_parameters():
        if param.requires_grad:
            if 'log_scale' in name:
                log_scale_params.append(param)
            else:
                other_params.append(param)
    
    base_lr = float(optim_cfg.get('LR', 5e-5))
    log_scale_lr_mult = float(p2r_loss_cfg.get('LOG_SCALE_LR_MULT', 0.1))
    
    param_groups = [
        {'params': other_params, 'lr': base_lr},
        {'params': log_scale_params, 'lr': base_lr * log_scale_lr_mult},
    ]
    
    optimizer = AdamW(
        param_groups,
        weight_decay=args.weight_decay if args.weight_decay is not None else float(optim_cfg.get('WEIGHT_DECAY', 1e-2)),
    )
    
    # Usa weight_decay effettivo per logging
    effective_weight_decay = args.weight_decay if args.weight_decay is not None else float(optim_cfg.get('WEIGHT_DECAY', 1e-2))
    
    epochs = optim_cfg.get('EPOCHS', 5000)
    scheduler = get_lr_scheduler(optimizer, optim_cfg, epochs)
    
    # Early stopping - patience ridotta per regularizzazione
    default_patience = 500 if args.use_augmentation else 2500
    patience = optim_cfg.get('EARLY_STOPPING_PATIENCE', default_patience)
    early_stopping = EarlyStopping(patience=patience)
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    print(f"\n⚙️ Training Config:")
    print(f"   Epochs: {epochs}")
    print(f"   LR: {base_lr}")
    print(f"   log_scale LR: {base_lr * log_scale_lr_mult}")
    print(f"   Weight Decay: {effective_weight_decay}")
    print(f"   Patience: {patience}")
    print(f"   Grad clip: {optim_cfg.get('GRAD_CLIP', 1.0)}")
    
    # =========================================================================
    # RESUME
    # =========================================================================
    
    start_epoch = 1
    best_mae = float('inf')
    
    resume_path = os.path.join(output_dir, 'stage2_bypass_last.pth')
    if os.path.isfile(resume_path):
        print(f"\n🔄 Resume: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        # Ripristina stato early stopping
        if ckpt.get('early_stopping'):
            early_stopping.counter = ckpt['early_stopping']['counter']
            early_stopping.best_score = ckpt['early_stopping']['best_score']
            print(f"   Early stopping: counter={early_stopping.counter}, best_score={early_stopping.best_score:.2f}")
        print(f"   Epoch {ckpt['epoch']}, Best MAE: {best_mae:.2f}")
    
    # =========================================================================
    # INITIAL VALIDATION
    # =========================================================================
    
    print("\n📋 Validazione iniziale:")
    val_metrics = validate(model, val_loader, device, default_down)
    print(f"   MAE: {val_metrics['mae']:.2f}")
    print(f"   RMSE: {val_metrics['rmse']:.2f}")
    print(f"   Bias: {val_metrics['bias']:.3f}")
    print(f"   π coverage: {val_metrics['pi_coverage']:.1f}%")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print(f"\n🚀 Training: {start_epoch} → {epochs}")
    
    val_interval = optim_cfg.get('VAL_INTERVAL', 5)
    history = []
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, default_down,
            grad_clip=optim_cfg.get('GRAD_CLIP', 1.0),
            scaler=scaler,
        )
        
        scheduler.step()
        
        # Validate
        if epoch % val_interval == 0:
            val_metrics = validate(model, val_loader, device, default_down)
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"   Train MAE: {train_metrics['mae']:.2f}")
            print(f"   Val MAE: {val_metrics['mae']:.2f}")
            print(f"   Val RMSE: {val_metrics['rmse']:.2f}")
            print(f"   Bias: {val_metrics['bias']:.3f}")
            print(f"   π coverage: {val_metrics['pi_coverage']:.1f}%")
            print(f"   log_scale: {val_metrics['log_scale']:.4f} (scale={val_metrics['scale']:.2f})")
            print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # History
            history.append({
                'epoch': epoch,
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'log_scale': val_metrics['log_scale'],
            })
            
            # Best model
            is_best = val_metrics['mae'] < best_mae
            if is_best:
                best_mae = val_metrics['mae']
                early_stopping.counter = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, best_mae, output_dir, is_best,
                early_stopping=early_stopping
            )
            
            # Save history
            with open(os.path.join(output_dir, 'stage2_bypass_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            
            # Early stopping
            if early_stopping(val_metrics['mae']):
                print(f"\n⛔ Early stopping @ epoch {epoch}")
                break
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETATO")
    print("=" * 70)
    print(f"   Best MAE: {best_mae:.2f}")
    print(f"   Final log_scale: {model.p2r_head.get_log_scale():.4f}")
    print(f"   Final scale: {model.p2r_head.get_scale():.2f}")
    print(f"   Checkpoint: {output_dir}/stage2_bypass_best.pth")
    
    if best_mae < 60:
        print("\n   🎯 TARGET RAGGIUNTO! MAE < 60")
    elif best_mae < 65:
        print("\n   ✅ Ottimo risultato! MAE < 65")
    elif best_mae < 70:
        print("\n   ✅ Buon risultato, simile a baseline P2R standalone")
    else:
        print("\n   ⚠️ Performance sotto le attese")
    
    print("=" * 70)
    
    return best_mae


if __name__ == '__main__':
    main()