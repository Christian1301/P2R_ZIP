# train_stage1_zip.py
# -*- coding: utf-8 -*-
"""
Stage 1 V3 - ZIP Pre-training con STE e POLARIZATION LOSS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
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
# LOSS IBRIDA (BCE + COUNT + POLARIZATION)
# ============================================================

class HybridZIPLoss(nn.Module):
    """
    Calcola Loss e Metriche. INCLUDE POLARIZATION LOSS.
    """
    
    def __init__(
        self,
        pos_weight_bce: float = 5.0,
        count_weight: float = 0.5,
        lambda_reg_weight: float = 0.01,
        polarization_weight: float = 1.5, # <--- IMPORTANTE: Peso per spingere a 0 o 1
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

        # 5. --- POLARIZATION LOSS (FIX CRITICA) ---
        # Spinge le probabilità verso gli estremi (0 o 1)
        loss_polar = (pi_prob * (1 - pi_prob)).mean()

        # Totale
        total_loss = loss_bce + \
                     self.count_weight * loss_count + \
                     self.lambda_reg_weight * loss_reg + \
                     self.polarization_weight * loss_polar  # <--- AGGIUNTO
        
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

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_polar": loss_polar.item(), # Monitorami!
            "loss_count": loss_count.item(),
            "accuracy": acc.item(),
            "recall": recall.item(),
            "precision": precision.item(),
            "coverage": coverage.item()
        }
        
        return total_loss, metrics


# ============================================================
# TRAINING LOOP (Identico alla tua versione, solo pulizia)
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
            'Pol': f"{batch_metrics['loss_polar']:.3f}", # Visualizza Polar
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("❌ Config mancante")
        return

    with open(args.config) as f: config = yaml.safe_load(f)
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    dataset_name = config["DATASET"]
    train_tf = build_transforms(config["DATA"], is_train=True)
    val_tf = build_transforms(config["DATA"], is_train=False)
    DatasetClass = get_dataset(dataset_name)
    
    train_loader = DataLoader(
        DatasetClass(config["DATA"]["ROOT"], config["DATA"]["TRAIN_SPLIT"], transforms=train_tf),
        batch_size=config["OPTIM_ZIP"]["BATCH_SIZE"], shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        DatasetClass(config["DATA"]["ROOT"], config["DATA"]["VAL_SPLIT"], transforms=val_tf),
        batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Configurazione Modello
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    use_ste = config["MODEL"].get("USE_STE_MASK", True)
    
    model = P2R_ZIP_Model(
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"], 
        bin_centers=bin_cfg["bin_centers"],
        upsample_to_input=False,
        use_ste_mask=use_ste, # <--- Corretto
    ).to(device)
    
    print(f"🔧 Modello creato con USE_STE_MASK = {use_ste}")

    for p in model.p2r_head.parameters(): p.requires_grad = False
    
    # Loss con Polarization Weight
    zip_cfg = config["ZIP_LOSS"]
    criterion = HybridZIPLoss(
        pos_weight_bce=zip_cfg.get("POS_WEIGHT_BCE", 5.0),
        count_weight=zip_cfg.get("COUNT_WEIGHT", 0.5),
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
        polarization_weight=1.5 # Hardcoded per sicurezza o leggi da config se vuoi
    ).to(device)

    optim_cfg = config["OPTIM_ZIP"]
    optimizer = get_optimizer([
        {"params": model.backbone.parameters(), "lr": optim_cfg["LR_BACKBONE"]},
        {"params": model.zip_head.parameters(), "lr": optim_cfg["BASE_LR"]},
    ], optim_cfg)
    
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)

    best_loss = float('inf')
    best_acc = 0.0
    start_epoch = 1

    checkpoint_path = os.path.join(out_dir, "stage1_last.pth")
    if not args.no_resume and os.path.isfile(checkpoint_path):
        print(f"🔄 Trovato checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"✅ Resume da epoca {start_epoch}")

    print(f"🚀 Avvio Stage 1 (USE_STE={use_ste}, PolarLoss=ATTIVA)")

    for epoch in range(start_epoch, optim_cfg["EPOCHS"] + 1):
        train_loss, train_met = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device, epoch)
        
        if epoch % optim_cfg["VAL_INTERVAL"] == 0:
            val_loss, val_met = validate(model, criterion, val_loader, device)
            
            print(f"   Val -> Loss: {val_loss:.4f} (Pol: {val_met['loss_polar']:.3f}) | Acc: {val_met['accuracy']:.2f}%")
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, best_loss, out_dir, is_best=True)
            
            if val_met['accuracy'] > best_acc:
                best_acc = val_met['accuracy']
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "accuracy": best_acc},
                    os.path.join(out_dir, "stage1_best_acc.pth")
                )

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'best_loss': best_loss, 'best_acc': best_acc,
        }, os.path.join(out_dir, "stage1_last.pth"))

if __name__ == "__main__":
    main()