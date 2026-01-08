#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V-Final - TRUE JOINT TRAINING
Obiettivo: Ottimizzazione congiunta ZIP + P2R
Loss: L_total = (1 - alpha) * L_ZIP + alpha * L_P2R

Sostituisce completamente la vecchia logica di 'Bias Correction'.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid
from losses.zip_nll import zip_nll

# Definiamo la Loss P2R localmente per renderla indipendente
class MultiScaleP2RLossLocal(nn.Module):
    def __init__(self, scales=[1, 2, 4], weights=[1.0, 0.5, 0.25], count_weight=2.0):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.count_weight = count_weight
        # Normalizza pesi scale
        tot = sum(weights)
        self.weights = [w/tot for w in weights]

    def forward(self, pred, points_list, cell_area):
        losses = []
        for s, w in zip(self.scales, self.weights):
            if s == 1:
                p_s = pred
            else:
                p_s = F.avg_pool2d(pred, kernel_size=s, stride=s) * (s**2)
            
            # Count loss per scala
            batch_loss = 0
            for i, pts in enumerate(points_list):
                gt = len(pts)
                curr_count = p_s[i].sum() / cell_area
                batch_loss += torch.abs(curr_count - gt)
            losses.append(batch_loss / len(points_list) * w)
            
        return sum(losses) * self.count_weight

class JointLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.p2r_criterion = MultiScaleP2RLossLocal()

    def forward(self, outputs, gt_density, points_list, cell_area, dims):
        H_in, W_in = dims
        
        # --- 1. ZIP LOSS (NLL) ---
        pi = torch.sigmoid(outputs['logit_pi_maps'][:, 1:2])
        lam = outputs['lambda_maps']
        
        # Downsample GT per ZIP
        block_h, block_w = pi.shape[-2:]
        factor_h, factor_w = H_in / block_h, W_in / block_w
        gt_counts_block = F.adaptive_avg_pool2d(gt_density, (block_h, block_w)) * (factor_h * factor_w)
        
        # ZIP NLL Loss
        loss_zip = zip_nll(pi, lam, gt_counts_block)
        
        # --- 2. P2R LOSS ---
        pred_p2r = outputs['p2r_density']
        # Canonicalizzazione per gestire dimensioni output diverse
        pred_p2r, _, _ = canonicalize_p2r_grid(pred_p2r, (H_in, W_in), 8.0)
        
        loss_p2r = self.p2r_criterion(pred_p2r, points_list, cell_area)
        
        # --- 3. COMBINED LOSS ---
        # Bilanciamento dinamico
        loss_total = (1 - self.alpha) * loss_zip + self.alpha * loss_p2r
        
        return loss_total, {
            "loss_zip": loss_zip.item(),
            "loss_p2r": loss_p2r.item()
        }

def train_one_epoch(model, criterion, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    metrics_acc = {"loss_zip": 0, "loss_p2r": 0}
    
    pbar = tqdm(loader, desc=f"Stage 3 Joint [Ep {epoch}]")
    for images, gt_density, points in pbar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        outputs = model(images)
        
        _, _, H, W = images.shape
        # Assumiamo downsample 8 standard per calcolo area in P2R
        cell_area = 64.0 
        
        loss, metrics = criterion(outputs, gt_density, points, cell_area, (H, W))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_acc[k] += v
            
        pbar.set_postfix({"L": f"{loss.item():.2f}", "Zip": f"{metrics['loss_zip']:.2f}", "P2R": f"{metrics['loss_p2r']:.2f}"})
        
    n = len(loader)
    return {k: v/n for k,v in metrics_acc.items()}, total_loss/n

def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml mancante")
        return

    with open("config.yaml") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    
    # 1. Dataset
    # Usiamo trasformazioni di training anche qui per robustezza
    train_tf = build_transforms(cfg['DATA'], is_train=True)
    val_tf = build_transforms(cfg['DATA'], is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    
    train_loader = DataLoader(
        DatasetClass(cfg["DATA"]["ROOT"], cfg["DATA"]["TRAIN_SPLIT"], transforms=train_tf),
        batch_size=cfg["OPTIM_JOINT"]["BATCH_SIZE"], 
        shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        DatasetClass(cfg["DATA"]["ROOT"], cfg["DATA"]["VAL_SPLIT"], transforms=val_tf),
        batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # 2. Modello
    bin_cfg = cfg["BINS_CONFIG"][cfg["DATASET"]]
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"], bin_centers=bin_cfg["bin_centers"],
        upsample_to_input=False
    ).to(device)
    
    # 3. Caricamento Checkpoint Stage 2 (CRUCIALE)
    out_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    ckpt_path = os.path.join(out_dir, "stage2_best.pth")
    if os.path.exists(ckpt_path):
        print(f"✅ Caricamento pesi Stage 2: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    else:
        print("⚠️  Stage 2 non trovato! Assicurati di aver completato lo stage 2.")

    # 4. Optimizer Differenziato
    # Backbone molto lento (fine-tuning), Heads normali
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-6}, # Backbone sbloccato ma lento
        {'params': model.p2r_head.parameters(), 'lr': 1e-5},
        {'params': model.zip_head.parameters(), 'lr': 1e-5}
    ], weight_decay=1e-4)
    
    # 5. Configurazione Loss Combinata
    alpha = cfg["JOINT_LOSS"].get("SOFT_WEIGHT_ALPHA", 0.3)
    criterion = JointLoss(alpha=alpha).to(device)
    
    print(f"🚀 Avvio Joint Training (Epoche: {cfg['OPTIM_JOINT']['EPOCHS']}, Alpha: {alpha})")
    
    best_mae = float('inf')
    
    for epoch in range(1, cfg["OPTIM_JOINT"]["EPOCHS"] + 1):
        # Train
        train_metrics, _ = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        
        # Validation semplice
        model.eval()
        abs_errs = []
        with torch.no_grad():
            for imgs, _, points in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                # Calcolo conteggio pred (usa p2r_density + eventuale mask se voluta, qui raw per semplicità metriche)
                pred = out['p2r_density']
                _, _, h_in, w_in = imgs.shape
                pred, (dh, dw), _ = canonicalize_p2r_grid(pred, (h_in, w_in), 8.0)
                pred_count = pred.sum() / (dh*dw)
                
                gt = len(points[0])
                abs_errs.append(abs(pred_count.item() - gt))
        
        val_mae = np.mean(abs_errs)
        print(f"   Epoch {epoch} | Val MAE: {val_mae:.2f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mae": best_mae}, 
                os.path.join(out_dir, "stage3_best.pth")
            )
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mae": best_mae}, 
                os.path.join(out_dir, "best_model.pth")
            )
            print("   🏆 New Best Model Saved!")

if __name__ == "__main__":
    main()