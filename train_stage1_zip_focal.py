#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 V6 - ZIP Pre-training per P2R-ZIP Model V2

Supporta sia il modello V1 (P2R_ZIP_Model) che V2 (P2R_ZIP_Model_V2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import sys
import numpy as np

from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    collate_fn,
)


# ============================================================
# FACTORY: Carica il modello corretto
# ============================================================

def create_model(config, device):
    """
    Crea il modello in base alla configurazione.
    Supporta sia V1 che V2.
    """
    model_cfg = config.get("MODEL", {})
    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    zip_head_cfg = config.get("ZIP_HEAD", {})
    
    # Parametri comuni
    backbone_name = model_cfg.get("BACKBONE", "vgg16_bn")
    pi_thresh = model_cfg.get("ZIP_PI_THRESH", 0.3)
    use_ste = model_cfg.get("USE_STE_MASK", True)
    
    # Determina quale modello usare
    model_version = config.get("MODEL_VERSION", "v1").lower()
    
    if model_version == "v2":
        from models.p2r_zip_model import P2R_ZIP_Model_V2
        
        model = P2R_ZIP_Model_V2(
            backbone_name=backbone_name,
            fusion_channels=256,
            pi_thresh=pi_thresh,
            use_ste_mask=use_ste,
            zip_head_kwargs={
                "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
                "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
                "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
                "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
            },
        )
        print(f"🔧 Modello V2 creato (P2R head binario)")
    else:
        from models.p2r_zip_model import P2R_ZIP_Model
        
        model = P2R_ZIP_Model(
            backbone_name=backbone_name,
            pi_thresh=pi_thresh,
            bins=bin_cfg["bins"],
            bin_centers=bin_cfg["bin_centers"],
            upsample_to_input=False,
            use_ste_mask=use_ste,
            zip_head_kwargs={
                "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
                "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
                "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
                "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
            },
        )
        print(f"🔧 Modello V1 creato")
    
    print(f"   USE_STE_MASK = {use_ste}")
    return model.to(device)


# ============================================================
# LOSS IBRIDA (BCE + COUNT + FOCAL POLARIZATION)
# ============================================================

class HybridZIPLoss(nn.Module):
    def __init__(
        self,
        pos_weight_bce: float = 1.5,
        count_weight: float = 0.1,
        lambda_reg_weight: float = 0.01,
        polarization_weight: float = 5.0,
        block_size: int = 16,
        occupancy_threshold: float = 0.5,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.lambda_reg_weight = lambda_reg_weight
        self.polarization_weight = polarization_weight 
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_bce]),
            reduction='mean'
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    
    def forward(self, predictions, gt_density):
        logit_pi = predictions["logit_pi_maps"][:, 1:2] 
        lambda_maps = predictions["lambda_maps"]         
        
        # 1. Preparazione Ground Truth
        gt_counts = F.avg_pool2d(gt_density, self.block_size, stride=self.block_size) * (self.block_size ** 2)
        if gt_counts.shape[-2:] != logit_pi.shape[-2:]:
            gt_counts = F.interpolate(gt_counts, size=logit_pi.shape[-2:], mode='nearest')
            
        gt_occupancy = (gt_counts > self.occupancy_threshold).float()
        
        # 2. Loss Classificazione (BCE)
        if self.bce.pos_weight.device != logit_pi.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pi.device)
            
        loss_bce = self.bce(logit_pi, gt_occupancy)
        
        # 3. Loss Conteggio
        pi_prob = torch.sigmoid(logit_pi)
        pred_count = pi_prob * lambda_maps
        mask_pos = (gt_counts > 0.5).float()
        
        loss_count_pos = self.smooth_l1(pred_count * mask_pos, gt_counts * mask_pos)
        loss_count_neg = (pred_count * (1 - mask_pos)).mean() * 0.1
        loss_count = loss_count_pos + loss_count_neg
        
        # 4. Regolarizzazione Lambda
        loss_reg = F.relu(lambda_maps - 10.0).mean()

        # 5. FOCAL-STYLE POLARIZATION LOSS
        distance_from_extreme = 1.0 - (2 * pi_prob - 1).abs()
        loss_polar = (distance_from_extreme ** 2).mean()

        # Totale
        total_loss = loss_bce + \
                     self.count_weight * loss_count + \
                     self.lambda_reg_weight * loss_reg + \
                     self.polarization_weight * loss_polar
        
        # 6. Metriche
        with torch.no_grad():
            pred_bin = (pi_prob > 0.5).float()
            acc = (pred_bin == gt_occupancy).float().mean() * 100
            
            tp = (pred_bin * gt_occupancy).sum()
            fp = (pred_bin * (1 - gt_occupancy)).sum()
            fn = ((1 - pred_bin) * gt_occupancy).sum()
            
            recall = (tp / (tp + fn + 1e-6)) * 100
            precision = (tp / (tp + fp + 1e-6)) * 100
            coverage = pred_bin.mean() * 100
            
            gray_zone = ((pi_prob > 0.2) & (pi_prob < 0.8)).float().mean() * 100

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_polar": loss_polar.item(),
            "loss_count": loss_count.item(),
            "accuracy": acc.item(),
            "recall": recall.item(),
            "precision": precision.item(),
            "coverage": coverage.item(),
            "gray_zone": gray_zone.item(),
        }
        
        return total_loss, metrics


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    metrics_accum = {}
    
    pbar = tqdm(dataloader, desc=f"Stage 1 [Ep {epoch}]")
    for images, gt_density, _ in pbar:
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        
        loss, batch_metrics = criterion(predictions, gt_density)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in batch_metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            
        pbar.set_postfix({
            'L': f"{loss.item():.2f}",
            'Pol': f"{batch_metrics['loss_polar']:.3f}",
            'Gray': f"{batch_metrics['gray_zone']:.1f}%",
            'Acc': f"{batch_metrics['accuracy']:.1f}"
        })

    if scheduler:
        scheduler.step()
    
    n = len(dataloader)
    return total_loss / n, {k: v/n for k,v in metrics_accum.items()}


def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    metrics_accum = {}
    
    with torch.no_grad():
        for images, gt_density, _ in tqdm(dataloader, desc="Validating", leave=False):
            images, gt_density = images.to(device), gt_density.to(device)
            predictions = model(images)
            loss, batch_metrics = criterion(predictions, gt_density)
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    n = len(dataloader)
    return total_loss / n, {k: v/n for k,v in metrics_accum.items()}


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config non trovato: {args.config}")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get("DEVICE", "cuda"))
    init_seeds(config.get("SEED", 2025))

    # Dataset
    dataset_name = config["DATASET"]
    data_cfg = config["DATA"]
    
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(dataset_name)
    
    train_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        transforms=train_tf
    )
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        transforms=val_tf
    )
    
    optim_cfg = config["OPTIM_ZIP"]
    
    train_loader = DataLoader(
        train_ds,
        batch_size=optim_cfg.get("BATCH_SIZE", 6),
        shuffle=True,
        num_workers=optim_cfg.get("NUM_WORKERS", 4),
        collate_fn=collate_fn,
        drop_last=True,
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
    
    print(f"Dataset train: Trovate {len(train_ds)} immagini.")
    print(f"Dataset test: Trovate {len(val_ds)} immagini.")

    # Modello (V1 o V2 in base a config)
    model = create_model(config, device)
    
    # Freeze P2R head (addestreremo solo ZIP in Stage 1)
    for p in model.p2r_head.parameters():
        p.requires_grad = False
    print("   P2R head: FROZEN")
    
    # Loss
    zip_cfg = config.get("ZIP_LOSS", {})
    
    pos_weight = float(zip_cfg.get("POS_WEIGHT_BCE", 1.5))
    pol_weight = float(zip_cfg.get("POLARIZATION_WEIGHT", 5.0))
    cnt_weight = float(zip_cfg.get("COUNT_WEIGHT", 0.1))
    
    criterion = HybridZIPLoss(
        pos_weight_bce=pos_weight,
        count_weight=cnt_weight,
        block_size=data_cfg.get("ZIP_BLOCK_SIZE", 16),
        polarization_weight=pol_weight,
    ).to(device)
    
    print(f"📊 Loss: POS_WEIGHT={pos_weight}, POLAR_WEIGHT={pol_weight}, COUNT_WEIGHT={cnt_weight}")

    # Optimizer
    optimizer = get_optimizer([
        {"params": model.backbone.parameters(), "lr": optim_cfg.get("LR_BACKBONE", 4e-5)},
        {"params": model.zip_head.parameters(), "lr": optim_cfg.get("BASE_LR", 8e-5)},
    ], optim_cfg)
    
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg.get("EPOCHS", 2000))

    # Output directory
    run_name = config.get("RUN_NAME", "p2r_zip")
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    # State
    best_loss = float('inf')
    best_acc = 0.0
    best_gray = 100.0
    start_epoch = 1

    # Resume
    checkpoint_path = os.path.join(out_dir, "stage1_last.pth")
    resume_enabled = optim_cfg.get("RESUME_LAST", False) and not args.no_resume
    
    if resume_enabled and os.path.isfile(checkpoint_path):
        print(f"🔄 Trovato checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            best_acc = ckpt.get('best_acc', 0.0)
            best_gray = ckpt.get('best_gray', 100.0)
            print(f"✅ Resume da epoca {start_epoch}")
        except Exception as e:
            print(f"⚠️ Errore resume: {e} - Training da zero")
    else:
        print("🆕 Training da zero")

    print(f"\n🚀 Avvio Stage 1 Training")
    print(f"   Epochs: {optim_cfg.get('EPOCHS', 2000)}")
    print(f"   Output: {out_dir}")
    print()
    sys.stdout.flush()

    # Training loop
    epochs = optim_cfg.get("EPOCHS", 2000)
    val_interval = optim_cfg.get("VAL_INTERVAL", 5)
    
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_met = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, device, epoch
        )
        
        if epoch % val_interval == 0:
            val_loss, val_met = validate(model, criterion, val_loader, device)
            
            gray = val_met['gray_zone']
            
            print(f"   Val -> Loss: {val_loss:.4f} | "
                  f"Pol: {val_met['loss_polar']:.3f} | "
                  f"Gray: {gray:.1f}% | "
                  f"Acc: {val_met['accuracy']:.1f}%")
            sys.stdout.flush()
            
            # Salvataggio Best Loss
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, best_loss, out_dir, is_best=True)
            
            # Salvataggio Best Accuracy
            if val_met['accuracy'] > best_acc:
                best_acc = val_met['accuracy']
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "accuracy": best_acc},
                    os.path.join(out_dir, "stage1_best_acc.pth")
                )
                print(f"   💾 New best accuracy: {best_acc:.1f}%")
            
            # Salvataggio Best Gray Zone
            if gray < best_gray:
                best_gray = gray
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "gray_zone": best_gray},
                    os.path.join(out_dir, "stage1_best_polar.pth")
                )

        # Checkpoint periodico
        torch.save({
            'epoch': epoch, 
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'best_loss': best_loss, 
            'best_acc': best_acc,
            'best_gray': best_gray,
        }, os.path.join(out_dir, "stage1_last.pth"))

    # Fine
    print("\n" + "=" * 60)
    print("🏁 STAGE 1 COMPLETATO")
    print("=" * 60)
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Best Accuracy: {best_acc:.1f}%")
    print(f"   Best Gray Zone: {best_gray:.1f}%")
    print(f"   Checkpoint: {out_dir}/stage1_best_acc.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()