#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V-Final FIXED - Joint Training con Pesi Fissi e Backbone Unfreeze
Obiettivo: Fine-tuning collaborativo ZIP + P2R

MODIFICHE CRITICHE:
1. Rimossa JointLossUncertainty (instabile). Sostituita con pesi fissi (P2R=1.0, ZIP=0.1).
2. Backbone parzialmente scongelato (ultimi blocchi) per adattamento features.
3. Soft Gating implicito (gestito dal modello con USE_STE_MASK=False).
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

# === LOSS P2R LOCALE ===
class MultiScaleP2RLossLocal(nn.Module):
    def __init__(self, scales=[1, 2, 4], weights=[1.0, 0.5, 0.25], count_weight=2.0):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.count_weight = count_weight
        tot = sum(weights)
        self.weights = [w/tot for w in weights]

    def forward(self, pred, points_list, cell_area):
        losses = []
        for s, w in zip(self.scales, self.weights):
            if s == 1:
                p_s = pred
            else:
                p_s = F.avg_pool2d(pred, kernel_size=s, stride=s) * (s**2)
            
            # L1 Count Loss per scala
            batch_loss = 0
            for i, pts in enumerate(points_list):
                gt = len(pts)
                curr_count = p_s[i].sum() / cell_area
                batch_loss += torch.abs(curr_count - gt)
            losses.append(batch_loss / len(points_list) * w)
            
        return sum(losses) * self.count_weight

# === TRAINING LOOP ===
def train_one_epoch(model, p2r_criterion, optimizer, loader, device, epoch, config_weights):
    model.train()
    total_loss = 0
    metrics_acc = {"loss_zip": 0, "loss_p2r": 0}
    
    # Pesi fissi dal config o default
    w_p2r = config_weights.get("FIXED_P2R_WEIGHT", 1.0)
    w_zip = config_weights.get("FIXED_ZIP_WEIGHT", 0.1)

    pbar = tqdm(loader, desc=f"Stage 3 Joint [Ep {epoch}]")
    for images, gt_density, points in pbar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        outputs = model(images)

        _, _, H, W = images.shape
        cell_area = 64.0 # Assumendo default downsample 8 (8*8)

        # 1. Calcolo Loss ZIP (per mantenere struttura)
        pi = torch.sigmoid(outputs['logit_pi_maps'][:, 1:2])
        lam = outputs['lambda_maps']
        block_h, block_w = pi.shape[-2:]
        factor_h, factor_w = H / block_h, W / block_w
        gt_counts_block = F.adaptive_avg_pool2d(gt_density, (block_h, block_w)) * (factor_h * factor_w)
        loss_zip = zip_nll(pi, lam, gt_counts_block)

        # 2. Calcolo Loss P2R (per precisione conteggio)
        pred_p2r = outputs['p2r_density']
        # Canonicalize serve per gestire eventuali discrepanze di size
        pred_p2r, _, _ = canonicalize_p2r_grid(pred_p2r, (H, W), 8.0)
        loss_p2r = p2r_criterion(pred_p2r, points, cell_area)

        # 3. Loss Totale Pesata
        loss = (w_p2r * loss_p2r) + (w_zip * loss_zip)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        metrics_acc["loss_zip"] += loss_zip.item()
        metrics_acc["loss_p2r"] += loss_p2r.item()

        pbar.set_postfix({
            "L": f"{loss.item():.2f}",
            "Zip": f"{loss_zip.item():.2f}",
            "P2R": f"{loss_p2r.item():.2f}"
        })

    n = len(loader)
    return {k: v/n for k,v in metrics_acc.items()}, total_loss/n

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no-resume', action='store_true', help='Disabilita resume automatico')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("❌ config.yaml mancante")
        return

    with open(args.config) as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    
    # 1. Dataset
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
    
    # Forziamo use_ste_mask=False da config o default per Soft Gating in training
    use_ste = cfg["MODEL"].get("USE_STE_MASK", False) 
    print(f"⚙️  Model Config: USE_STE_MASK={use_ste} (Deve essere False per Soft Gating)")

    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"], bin_centers=bin_cfg["bin_centers"],
        upsample_to_input=False,
        use_ste_mask=use_ste 
    ).to(device)
    
    # 3. Caricamento Checkpoint Stage 2
    out_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    ckpt_path = os.path.join(out_dir, "stage2_best.pth")
    if os.path.exists(ckpt_path):
        print(f"✅ Caricamento pesi Stage 2: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    else:
        print("⚠️  Stage 2 non trovato! Training potrebbe essere instabile.")

    # 4. Optimizer & Unfreeze Parziale
    # Congela tutto inizialmente
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Scongela ultimi blocchi del backbone (VGG features 34-43)
    # Questo permette alle features condivise di adattarsi al task congiunto
    print("🔓 Scongelamento ultimi layer backbone (body[34:])...")
    trainable_backbone_params = []
    # Nota: la struttura interna dipende da torchvision.models.vgg16_bn
    # model.backbone.body è un nn.Sequential
    for i, layer in enumerate(model.backbone.body):
        if i >= 34: 
            for param in layer.parameters():
                param.requires_grad = True
                trainable_backbone_params.append(param)
    
    lr_backbone = float(cfg["OPTIM_JOINT"].get("LR_BACKBONE", 1e-6))
    lr_heads = float(cfg["OPTIM_JOINT"].get("LR_HEADS", 1e-5))

    optimizer = torch.optim.AdamW([
        {'params': trainable_backbone_params, 'lr': lr_backbone},
        {'params': model.p2r_head.parameters(), 'lr': lr_heads},
        {'params': model.zip_head.parameters(), 'lr': lr_heads},
    ], weight_decay=1e-4)

    p2r_criterion = MultiScaleP2RLossLocal().to(device)
    
    # Scheduler
    epochs = cfg["OPTIM_JOINT"]["EPOCHS"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Resume logic
    best_mae = float('inf')
    start_epoch = 1
    
    # ... (Codice resume omesso per brevità, identico a prima ma senza criterion load) ...

    print(f"🚀 Avvio Joint Training (Fixed Weights): P2R=1.0, ZIP=0.1")
    
    for epoch in range(start_epoch, epochs + 1):
        # Training
        train_metrics, _ = train_one_epoch(
            model, p2r_criterion, optimizer, train_loader, device, epoch, cfg["JOINT_LOSS"]
        )

        # Validation
        model.eval()
        abs_errs = []
        with torch.no_grad():
            for imgs, _, points in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                # In eval, il modello applica automaticamente output masking rigido
                pred = out['p2r_density']
                _, _, h_in, w_in = imgs.shape
                pred, (dh, dw), _ = canonicalize_p2r_grid(pred, (h_in, w_in), 8.0)
                pred_count = pred.sum() / (dh*dw)
                gt = len(points[0])
                abs_errs.append(abs(pred_count.item() - gt))

        val_mae = np.mean(abs_errs)
        scheduler.step(val_mae)

        print(f"   Epoch {epoch} | Val MAE: {val_mae:.2f} | ZIP Loss: {train_metrics['loss_zip']:.3f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "mae": best_mae
            }, os.path.join(out_dir, "stage3_best.pth"))
            print("   🏆 New Best Model Saved!")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mae': best_mae,
        }, os.path.join(out_dir, "stage3_last.pth"))

if __name__ == "__main__":
    main()