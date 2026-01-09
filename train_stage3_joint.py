#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 V-Final - TRUE JOINT TRAINING con EMA-Normalized Uncertainty Weighting
Obiettivo: Ottimizzazione congiunta ZIP + P2R con bilanciamento automatico

Approccio:
1. Normalizza le loss con EMA (Exponential Moving Average) per portarle sulla stessa scala
2. Applica Uncertainty Weighting (Kendall et al. 2018) alle loss normalizzate

Questo garantisce che i pesi learned riflettano la vera importanza dei task,
non le differenze di scala tra le loss.
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

class JointLossUncertainty(nn.Module):
    """
    Joint Loss con EMA-Normalized Uncertainty Weighting

    1. Normalizza le loss usando EMA (Exponential Moving Average)
       - Porta entrambe le loss sulla stessa scala (~1.0)
    2. Applica Uncertainty Weighting (Kendall et al. 2018) alle loss normalizzate
       - I pesi learned riflettono la vera importanza, non la scala

    Formula: L = L_zip_norm * exp(-s_zip) + s_zip + L_p2r_norm * exp(-s_p2r) + s_p2r
    dove L_i_norm = L_i / EMA(L_i)
    """
    def __init__(self, ema_decay=0.99):
        super().__init__()
        self.p2r_criterion = MultiScaleP2RLossLocal()
        self.ema_decay = ema_decay

        # EMA buffers per normalizzazione (non-learnable, persistenti)
        # Inizializzati a valori tipici per evitare divisione per zero
        self.register_buffer('zip_ema', torch.tensor([1.0]))
        self.register_buffer('p2r_ema', torch.tensor([100.0]))  # P2R è tipicamente ~100

        # Log-varianze learnable (inizializzate a 0 = peso uguale)
        self.log_var_zip = nn.Parameter(torch.zeros(1))
        self.log_var_p2r = nn.Parameter(torch.zeros(1))

    def forward(self, outputs, gt_density, points_list, cell_area, dims):
        H_in, W_in = dims

        # --- 1. ZIP LOSS (NLL) ---
        pi = torch.sigmoid(outputs['logit_pi_maps'][:, 1:2])
        lam = outputs['lambda_maps']

        block_h, block_w = pi.shape[-2:]
        factor_h, factor_w = H_in / block_h, W_in / block_w
        gt_counts_block = F.adaptive_avg_pool2d(gt_density, (block_h, block_w)) * (factor_h * factor_w)

        loss_zip = zip_nll(pi, lam, gt_counts_block)

        # --- 2. P2R LOSS ---
        pred_p2r = outputs['p2r_density']
        pred_p2r, _, _ = canonicalize_p2r_grid(pred_p2r, (H_in, W_in), 8.0)
        loss_p2r = self.p2r_criterion(pred_p2r, points_list, cell_area)

        # --- 3. AGGIORNA EMA (solo durante training) ---
        if self.training:
            with torch.no_grad():
                self.zip_ema = self.ema_decay * self.zip_ema + (1 - self.ema_decay) * loss_zip.detach()
                self.p2r_ema = self.ema_decay * self.p2r_ema + (1 - self.ema_decay) * loss_p2r.detach()

        # --- 4. NORMALIZZA LE LOSS (porta entrambe a scala ~1.0) ---
        loss_zip_norm = loss_zip / (self.zip_ema + 1e-8)
        loss_p2r_norm = loss_p2r / (self.p2r_ema + 1e-8)

        # --- 5. UNCERTAINTY WEIGHTED COMBINATION su loss normalizzate ---
        precision_zip = torch.exp(-self.log_var_zip)
        precision_p2r = torch.exp(-self.log_var_p2r)

        loss_zip_weighted = loss_zip_norm * precision_zip + self.log_var_zip
        loss_p2r_weighted = loss_p2r_norm * precision_p2r + self.log_var_p2r

        loss_total = loss_zip_weighted + loss_p2r_weighted

        # Calcola pesi effettivi per logging
        w_zip = precision_zip / (precision_zip + precision_p2r)
        w_p2r = precision_p2r / (precision_zip + precision_p2r)

        return loss_total, {
            "loss_zip": loss_zip.item(),
            "loss_p2r": loss_p2r.item(),
            "loss_zip_norm": loss_zip_norm.item(),
            "loss_p2r_norm": loss_p2r_norm.item(),
            "w_zip": w_zip.item(),
            "w_p2r": w_p2r.item(),
            "zip_ema": self.zip_ema.item(),
            "p2r_ema": self.p2r_ema.item()
        }

def train_one_epoch(model, criterion, optimizer, loader, device, epoch):
    model.train()
    criterion.train()  # Importante per i parametri learnable della loss
    total_loss = 0
    metrics_acc = {"loss_zip": 0, "loss_p2r": 0, "w_zip": 0, "w_p2r": 0}

    pbar = tqdm(loader, desc=f"Stage 3 Joint [Ep {epoch}]")
    for images, gt_density, points in pbar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        outputs = model(images)

        _, _, H, W = images.shape
        cell_area = 64.0

        loss, metrics = criterion(outputs, gt_density, points, cell_area, (H, W))

        loss.backward()
        # Clip gradients per model e criterion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in ["loss_zip", "loss_p2r", "w_zip", "w_p2r"]:
            metrics_acc[k] += metrics[k]

        # Mostra loss e pesi learned
        pbar.set_postfix({
            "L": f"{loss.item():.2f}",
            "Zip": f"{metrics['loss_zip']:.2f}",
            "P2R": f"{metrics['loss_p2r']:.2f}",
            "wZ": f"{metrics['w_zip']:.2f}",
            "wP": f"{metrics['w_p2r']:.2f}"
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

    # 4. Loss con Uncertainty Weighting (Kendall et al. 2018)
    # Crea la loss PRIMA dell'optimizer per includere i suoi parametri learnable
    criterion = JointLossUncertainty().to(device)

    # 5. Optimizer CONSERVATIVO con ReduceLROnPlateau
    # Backbone COMPLETAMENTE congelato
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Heads + parametri della loss (log_var_zip, log_var_p2r)
    initial_lr = 1e-5  # LR leggermente più alto per uncertainty weighting
    optimizer = torch.optim.AdamW([
        {'params': model.p2r_head.parameters(), 'lr': initial_lr},
        {'params': model.zip_head.parameters(), 'lr': initial_lr},
        {'params': criterion.parameters(), 'lr': initial_lr * 10}  # LR più alto per log_var
    ], weight_decay=1e-5)

    print("   Backbone: FROZEN")
    print(f"   Heads LR: {initial_lr}")
    print(f"   Loss weights LR: {initial_lr * 10} (Uncertainty Weighting)")

    # Scheduler ReduceLROnPlateau
    epochs = cfg["OPTIM_JOINT"]["EPOCHS"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=True
    )

    # Resume automatico
    best_mae = float('inf')
    start_epoch = 1
    no_improve_count = 0
    patience = 50  # Early stopping dopo 50 epoche senza miglioramento

    checkpoint_path = os.path.join(out_dir, "stage3_last.pth")
    if not args.no_resume and os.path.isfile(checkpoint_path):
        print(f"🔄 Trovato checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'criterion' in ckpt:
            criterion.load_state_dict(ckpt['criterion'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', float('inf'))
        no_improve_count = ckpt.get('no_improve_count', 0)
        print(f"✅ Resume da epoca {start_epoch}, best MAE: {best_mae:.2f}")

    print(f"🚀 Avvio Joint Training con EMA-Normalized Uncertainty Weighting:")
    print(f"   Epoche: {epochs}")
    print(f"   Early Stopping Patience: {patience}")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
    print(f"   Loss: EMA normalization + Kendall et al. 2018")
    print(f"   EMA decay: 0.99 (normalizza loss a scala ~1.0)")

    for epoch in range(start_epoch, epochs + 1):
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

        # Scheduler step (ReduceLROnPlateau richiede la metrica)
        scheduler.step(val_mae)

        # Mostra LR corrente e pesi learned
        current_lr = optimizer.param_groups[0]['lr']
        w_zip = train_metrics.get('w_zip', 0.5)
        w_p2r = train_metrics.get('w_p2r', 0.5)
        print(f"   Epoch {epoch} | Val MAE: {val_mae:.2f} | LR: {current_lr:.2e} | Weights: ZIP={w_zip:.1%} P2R={w_p2r:.1%}")

        if val_mae < best_mae:
            best_mae = val_mae
            no_improve_count = 0
            torch.save({
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "mae": best_mae
            }, os.path.join(out_dir, "stage3_best.pth"))
            torch.save({
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "mae": best_mae
            }, os.path.join(out_dir, "best_model.pth"))
            print("   🏆 New Best Model Saved!")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\n⛔ Early stopping @ epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Salva checkpoint per resume (ogni epoca)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mae': best_mae,
            'no_improve_count': no_improve_count,
        }, os.path.join(out_dir, "stage3_last.pth"))

    print(f"\n🏁 Training completato! Best MAE: {best_mae:.2f}")

if __name__ == "__main__":
    main()