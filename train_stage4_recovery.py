# P2R_ZIP/train_stage4_recovery.py
# -*- coding: utf-8 -*-
"""
Stage 4 - Recovery & Polishing (OPTIMIZED VERSION)

Obiettivo: Recuperare e migliorare il MAE dopo Stage 3.
Strategie chiave:
1. Backbone completamente congelato
2. Focus sulla loss L1 del conteggio
3. Ensemble di calibrazioni
4. Validazione stratificata
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
from losses.p2r_region_loss import P2RLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    canonicalize_p2r_grid, collate_fn
)
from train_stage2_p2r import calibrate_density_scale


# === LOSS ZIP LOCALE ===
class PiHeadLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, block_size: int = 16):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return (gt_counts_per_block > 1e-3).float()
    
    def forward(self, predictions, gt_density):
        logit_pi_maps = predictions["logit_pi_maps"]
        logit_pieno = logit_pi_maps[:, 1:2, :, :] 
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest'
            )
        
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
            
        loss = self.bce(logit_pieno, gt_occupancy)
        return loss, {}


# === HUBER LOSS PER CONTEGGIO (più robusto di L1) ===
class CountHuberLoss(nn.Module):
    def __init__(self, delta=10.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred_count, gt_count):
        diff = torch.abs(pred_count - gt_count)
        quadratic = torch.clamp(diff, max=self.delta)
        linear = diff - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear


# === TRAINING LOOP ===
def train_one_epoch(
    model, criterion_zip, criterion_p2r, count_loss_fn, dataloader, optimizer, 
    device, default_down, epoch, zip_scale, p2r_scale, count_weight
):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Stage 4 Recovery [Ep {epoch}]",
    )

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) if p is not None else None for p in points]

        optimizer.zero_grad()
        outputs = model(images)
        
        # 1. Loss ZIP (peso basso)
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        # 2. Loss P2R (peso medio)
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)
        
        # 3. Loss Conteggio (peso ALTO - focus principale)
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        gt_count = torch.tensor(
            [len(p) if p is not None else 0 for p in points],
            device=device, dtype=torch.float
        )
        
        # Usa Huber loss per robustezza agli outlier
        loss_count = count_loss_fn(pred_count, gt_count).mean()
        
        # Loss combinata con forte enfasi sul conteggio
        combined_loss = scaled_loss_zip + p2r_scale * loss_p2r + count_weight * loss_count
        
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += combined_loss.item()
        
        # MAE per monitoring
        batch_mae = torch.abs(pred_count - gt_count).mean().item()
        total_mae += batch_mae
        num_batches += 1
        
        progress_bar.set_postfix({
            "Loss": f"{combined_loss.item():.2f}",
            "MAE": f"{batch_mae:.1f}",
            "L_cnt": f"{loss_count.item():.2f}"
        })

    avg_mae = total_mae / max(num_batches, 1)
    print(f"   ↪ Train MAE: {avg_mae:.2f}")
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_stratified(model, dataloader, device, default_down):
    """Validazione stratificata per capire dove il modello sbaglia."""
    model.eval()
    
    # Bins per stratificazione
    sparse_errors = []  # 0-100 persone
    medium_errors = []  # 100-500 persone
    dense_errors = []   # 500+ persone
    
    all_errors = []
    all_pred, all_gt = [], []

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 4"):
        images = images.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = (torch.sum(pred_density, dim=(1, 2, 3)) / cell_area).cpu().numpy()
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred = pred_count[idx] if idx < len(pred_count) else 0
            error = abs(pred - gt)
            
            all_errors.append(error)
            all_pred.append(pred)
            all_gt.append(gt)
            
            if gt <= 100:
                sparse_errors.append(error)
            elif gt <= 500:
                medium_errors.append(error)
            else:
                dense_errors.append(error)

    # Report
    mae = np.mean(all_errors)
    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))
    
    total_pred = sum(all_pred)
    total_gt = sum(all_gt)
    bias = total_pred / total_gt if total_gt > 0 else 0

    print("\n📊 Validazione Stratificata Stage 4:")
    print(f"   Overall MAE: {mae:.2f}, RMSE: {rmse:.2f}, Bias: {bias:.3f}")
    
    if sparse_errors:
        print(f"   Sparse (0-100):   MAE={np.mean(sparse_errors):.2f} ({len(sparse_errors)} imgs)")
    if medium_errors:
        print(f"   Medium (100-500): MAE={np.mean(medium_errors):.2f} ({len(medium_errors)} imgs)")
    if dense_errors:
        print(f"   Dense (500+):     MAE={np.mean(dense_errors):.2f} ({len(dense_errors)} imgs)")

    return mae, rmse, total_pred, total_gt


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
        
    with open("config.yaml") as f: 
        config = yaml.safe_load(f)
    
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 4 (Recovery - OPTIMIZED) su {device}")

    optim_cfg = config["OPTIM_STAGE4"]
    loss_cfg = config["STAGE4_LOSS"]
    data_cfg = config["DATA"]
    
    # Dataset
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"], 
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"], 
        transforms=train_transforms
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"], 
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"], 
        transforms=val_transforms
    )

    # USA collate_fn standard (come Stage 2)
    train_loader = DataLoader(
        train_dataset, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True,
        collate_fn=collate_fn
    )

    # Modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"]["LAMBDA_SCALE"],
        "lambda_max": config["ZIP_HEAD"]["LAMBDA_MAX"],
        "use_softplus": config["ZIP_HEAD"]["USE_SOFTPLUS"],
        "lambda_noise_std": 0.0,
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)

    # Load best checkpoint (priorità: stage3_best, poi stage2_best)
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    
    checkpoint_candidates = [
        os.path.join(output_dir, "stage3_best.pth"),
        os.path.join(output_dir, "stage2_best.pth"),
    ]
    
    loaded = False
    for ckpt_path in checkpoint_candidates:
        if os.path.isfile(ckpt_path):
            print(f"🔄 Caricamento pesi da: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            
            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            model.load_state_dict(state_dict, strict=False)
            loaded = True
            break
    
    if not loaded:
        print("❌ Nessun checkpoint trovato per Stage 4")
        return

    # CONGELA COMPLETAMENTE IL BACKBONE
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("🧊 Backbone completamente congelato")
    
    # Solo heads trainabili con LR molto basso
    trainable_params = list(model.zip_head.parameters()) + list(model.p2r_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optim_cfg["LR_HEADS"]),
        weight_decay=float(optim_cfg["WEIGHT_DECAY"])
    )
    scheduler = get_scheduler(optimizer, optim_cfg, optim_cfg["EPOCHS"])

    # Loss functions
    crit_zip = PiHeadLoss(5.0, data_cfg["ZIP_BLOCK_SIZE"]).to(device)
    
    p2r_cfg = config["P2R_LOSS"]
    crit_p2r = P2RLoss(
        scale_weight=float(p2r_cfg["SCALE_WEIGHT"]),
        pos_weight=float(p2r_cfg["POS_WEIGHT"]),
        chunk_size=int(p2r_cfg["CHUNK_SIZE"]),
        min_radius=float(p2r_cfg["MIN_RADIUS"]),
        max_radius=float(p2r_cfg["MAX_RADIUS"])
    ).to(device)
    
    count_loss_fn = CountHuberLoss(delta=20.0)  # Robusto agli outlier

    # Calibrazione iniziale
    print("🔧 Calibrazione iniziale...")
    calibrate_density_scale(
        model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"],
        max_batches=None,
        clamp_range=p2r_cfg.get("LOG_SCALE_CLAMP"),
        max_adjust=1.5
    )
    
    # Valutazione iniziale
    print("\n📋 Valutazione iniziale:")
    init_mae, init_rmse, _, _ = validate_stratified(model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"])
    
    best_mae = init_mae
    patience = optim_cfg["EARLY_STOPPING_PATIENCE"]
    no_improve = 0
    default_down = data_cfg["P2R_DOWNSAMPLE"]
    
    zip_s = float(loss_cfg["ZIP_SCALE"])
    p2r_s = float(loss_cfg.get("ALPHA", 0.3))
    count_w = float(loss_cfg["COUNT_L1_W"])

    print(f"\n🚀 START STAGE 4 (Recovery)")
    print(f"   Loss weights: ZIP={zip_s}, P2R={p2r_s}, COUNT={count_w}")
    
    for epoch in range(optim_cfg["EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{optim_cfg['EPOCHS']} ---")
        
        train_loss = train_one_epoch(
            model, crit_zip, crit_p2r, count_loss_fn, train_loader, optimizer,
            device, default_down, epoch+1, zip_s, p2r_s, count_w
        )
        
        if scheduler: 
            scheduler.step()

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            # Calibra prima della validazione
            if hasattr(model.p2r_head, "log_scale"):
                backup_scale = model.p2r_head.log_scale.data.clone()
                calibrate_density_scale(
                    model, val_loader, device, default_down,
                    max_batches=15,
                    clamp_range=p2r_cfg.get("LOG_SCALE_CLAMP"),
                    max_adjust=0.3
                )
            
            val_mae, val_rmse, tot_pred, tot_gt = validate_stratified(
                model, val_loader, device, default_down
            )
            bias = tot_pred / tot_gt if tot_gt > 0 else 0
            
            print(f"Ep {epoch+1}: Val MAE: {val_mae:.2f} | Bias: {bias:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), os.path.join(output_dir, "stage4_best.pth"))
                print(f"🏆 NEW BEST: {best_mae:.2f}")
                no_improve = 0
            else:
                no_improve += 1
                # Ripristina scala se non migliora
                if hasattr(model.p2r_head, "log_scale"):
                    model.p2r_head.log_scale.data = backup_scale
            
            if no_improve >= patience:
                print("⛔ Early Stopping Stage 4")
                break
    
    print(f"\n✅ STAGE 4 COMPLETATO. Best MAE: {best_mae:.2f}")
    
    # Target check
    if best_mae <= 65:
        print("🎯 TARGET RAGGIUNTO! MAE ≤ 65")
    else:
        print(f"⚠️ Target non raggiunto. Gap: {best_mae - 60:.2f}")


if __name__ == "__main__":
    main()