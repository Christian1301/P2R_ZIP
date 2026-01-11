# train_stage1_zip.py
# -*- coding: utf-8 -*-
"""
Stage 1 V-Final - ZIP Pre-training
Obiettivo: Addestrare il backbone a distinguere Background vs Foreground.

MODIFICHE RICHIESTE:
1. Calcolo metriche complete (Accuracy, Precision, Recall).
2. Salvataggio DOPPIO checkpoint:
   - 'best_model.pth': basato sulla Loss totale (bilanciamento).
   - 'stage1_best_acc.pth': basato sulla pura Accuracy (migliore separazione BG/FG).
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
    resume_if_exists,
    save_checkpoint,
    collate_fn,
)


# ============================================================
# LOSS IBRIDA (BCE + COUNT)
# ============================================================

class HybridZIPLoss(nn.Module):
    """
    Calcola Loss e Metriche: Accuracy, Precision, Recall.
    """
    
    def __init__(
        self,
        pos_weight_bce: float = 5.0,
        count_weight: float = 0.5,
        lambda_reg_weight: float = 0.01,
        block_size: int = 16,
        occupancy_threshold: float = 0.5,
    ):
        super().__init__()
        self.count_weight = count_weight
        self.lambda_reg_weight = lambda_reg_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        # BCE con pos_weight per sbilanciamento classi (molto più background che folla)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_bce]),
            reduction='mean'
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    
    def forward(self, predictions, gt_density):
        logit_pi = predictions["logit_pi_maps"][:, 1:2]  # [B, 1, H, W]
        lambda_maps = predictions["lambda_maps"]         # [B, 1, H, W]
        
        # 1. Preparazione Ground Truth per blocchi
        # Downsample densità per ottenere conteggio nel blocco
        gt_counts = F.avg_pool2d(gt_density, self.block_size, stride=self.block_size) * (self.block_size ** 2)
        
        # Allinea dimensioni se necessario (gestione padding/rounding)
        if gt_counts.shape[-2:] != logit_pi.shape[-2:]:
            gt_counts = F.interpolate(gt_counts, size=logit_pi.shape[-2:], mode='nearest')
            
        # GT Occupancy: 1 se c'è almeno mezza persona, 0 altrimenti
        gt_occupancy = (gt_counts > self.occupancy_threshold).float()
        
        # 2. Loss Classificazione (BCE)
        if self.bce.pos_weight.device != logit_pi.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pi.device)
            
        loss_bce = self.bce(logit_pi, gt_occupancy)
        
        # 3. Loss Conteggio (solo su blocchi attivi e leggermente su vuoti)
        pi_prob = torch.sigmoid(logit_pi)
        pred_count = pi_prob * lambda_maps
        
        # Maschera blocchi pieni
        mask_pos = (gt_counts > 0.5).float()
        
        # Loss conteggio sui positivi
        loss_count_pos = self.smooth_l1(pred_count * mask_pos, gt_counts * mask_pos)
        # Penalità sui negativi (devono tendere a 0)
        loss_count_neg = (pred_count * (1 - mask_pos)).mean() * 0.1
        
        loss_count = loss_count_pos + loss_count_neg
        
        # 4. Regolarizzazione Lambda
        loss_reg = F.relu(lambda_maps - 10.0).mean() # Penalizza valori esplosivi
        
        # Totale
        total_loss = loss_bce + self.count_weight * loss_count + self.lambda_reg_weight * loss_reg
        
        # 5. Calcolo Metriche Extra
        with torch.no_grad():
            pred_bin = (pi_prob > 0.5).float()
            
            # Accuracy
            acc = (pred_bin == gt_occupancy).float().mean() * 100
            
            # Precision & Recall
            tp = (pred_bin * gt_occupancy).sum()
            fp = (pred_bin * (1 - gt_occupancy)).sum()
            fn = ((1 - pred_bin) * gt_occupancy).sum()
            
            recall = (tp / (tp + fn + 1e-6)) * 100
            precision = (tp / (tp + fp + 1e-6)) * 100
            
            # Coverage (percentuale area predetta come piena)
            coverage = pred_bin.mean() * 100

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_count": loss_count.item(),
            "accuracy": acc.item(),
            "recall": recall.item(),
            "precision": precision.item(),
            "coverage": coverage.item()
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
        
        # Accumula metriche
        for k, v in batch_metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            
        pbar.set_postfix({
            'L': f"{loss.item():.2f}",
            'Acc': f"{batch_metrics['accuracy']:.1f}",
            'Rec': f"{batch_metrics['recall']:.1f}"
        })

    if scheduler:
        scheduler.step()
    
    # Media epoca
    n = len(dataloader)
    metrics_avg = {k: v/n for k,v in metrics_accum.items()}
    return total_loss / n, metrics_avg


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
    metrics_avg = {k: v/n for k,v in metrics_accum.items()}
    return total_loss / n, metrics_avg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no-resume', action='store_true', help='Disabilita resume automatico')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("❌ Config mancante")
        return

    with open(args.config) as f: config = yaml.safe_load(f)
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    # 1. Dataset e Dataloader
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

    # 2. Modello
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    model = P2R_ZIP_Model(
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"], bin_centers=bin_cfg["bin_centers"],
        upsample_to_input=False
    ).to(device)

    # Congela P2R head per Stage 1
    for p in model.p2r_head.parameters(): p.requires_grad = False
    
    # 3. Loss e Optimizer
    zip_cfg = config["ZIP_LOSS"]
    criterion = HybridZIPLoss(
        pos_weight_bce=zip_cfg.get("POS_WEIGHT_BCE", 5.0),
        count_weight=zip_cfg.get("COUNT_WEIGHT", 0.5),
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"]
    ).to(device)

    optim_cfg = config["OPTIM_ZIP"]
    optimizer = get_optimizer([
        {"params": model.backbone.parameters(), "lr": optim_cfg["LR_BACKBONE"]},
        {"params": model.zip_head.parameters(), "lr": optim_cfg["BASE_LR"]},
    ], optim_cfg)
    
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # 4. Training Loop
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)

    best_loss = float('inf')
    best_acc = 0.0  # NUOVA METRICA
    start_epoch = 1

    # Resume automatico
    checkpoint_path = os.path.join(out_dir, "stage1_last.pth")
    if not args.no_resume and os.path.isfile(checkpoint_path):
        print(f"🔄 Trovato checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"✅ Resume da epoca {start_epoch}, best_loss: {best_loss:.4f}, best_acc: {best_acc:.2f}%")

    print(f"🚀 Avvio Stage 1 (Salvataggio: Best Loss & Best Acc)")

    for epoch in range(start_epoch, optim_cfg["EPOCHS"] + 1):
        # Train
        train_loss, train_met = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device, epoch)
        
        # Validation
        if epoch % optim_cfg["VAL_INTERVAL"] == 0:
            val_loss, val_met = validate(model, criterion, val_loader, device)
            
            print(f"   Val Results -> Loss: {val_loss:.4f} | Acc: {val_met['accuracy']:.2f}% | Rec: {val_met['recall']:.2f}% | Prec: {val_met['precision']:.2f}%")
            
            # Salvataggio 1: Miglior Loss (Bilanciamento Generale)
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, best_loss, out_dir, is_best=True)
                print("   🏆 New Best Loss!")
            
            # Salvataggio 2: Miglior Accuracy (Miglior Maschera) - RICHIESTO DALL'UTENTE
            if val_met['accuracy'] > best_acc:
                best_acc = val_met['accuracy']
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "accuracy": best_acc},
                    os.path.join(out_dir, "stage1_best_acc.pth")
                )
                print(f"   ⭐ New Best Accuracy: {best_acc:.2f}%")

        # Salvataggio checkpoint per resume (ogni epoca)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'best_loss': best_loss,
            'best_acc': best_acc,
        }, os.path.join(out_dir, "stage1_last.pth"))

if __name__ == "__main__":
    main()