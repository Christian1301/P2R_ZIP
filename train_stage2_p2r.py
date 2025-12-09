# train_stage2_p2r_fixed.py
# -*- coding: utf-8 -*-
"""
Stage 2 P2R Training - VERSIONE CORRETTA

Correzioni principali:
1. Backbone COMPLETAMENTE congelato (no fine-tuning)
2. Scale weight ridotto drasticamente
3. Calibrazione migliorata con analisi per-immagine
4. Debug più dettagliato per identificare problemi
5. Early stopping basato su MAE, non su loss
"""

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms

from losses.p2r_region_loss import P2RLossFixed
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    setup_experiment,
    collate_fn,
    canonicalize_p2r_grid,
    calibrate_density_scale_v2,
    debug_batch_predictions,
    save_checkpoint,
)


def calibrate_density_scale(*args, **kwargs):
    """Compatibilità con altri stage: alias verso la versione aggiornate."""
    return calibrate_density_scale_v2(*args, **kwargs)


@torch.no_grad()
def evaluate_p2r_detailed(model, loader, loss_fn, device, default_down):
    """
    Valutazione dettagliata con breakdown per densità.
    """
    model.eval()
    
    all_pred, all_gt = [], []
    sparse_errors, medium_errors, dense_errors = [], [], []
    total_loss = 0.0

    for images, _, points in tqdm(loader, desc="[Eval P2R]"):
        images = images.to(device)
        points_list = [p.to(device) for p in points]

        out = model(images)
        pred_density = out.get("p2r_density")

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        
        loss = loss_fn(pred_density, points_list, down=down_tuple)
        total_loss += loss.item()

        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            pred = (pred_density[i].sum() / cell_area).item()
            
            all_pred.append(pred)
            all_gt.append(gt)
            
            error = abs(pred - gt)
            if gt <= 100:
                sparse_errors.append(error)
            elif gt <= 500:
                medium_errors.append(error)
            else:
                dense_errors.append(error)

    # Statistiche
    pred_np = np.array(all_pred)
    gt_np = np.array(all_gt)
    
    mae = np.mean(np.abs(pred_np - gt_np))
    rmse = np.sqrt(np.mean((pred_np - gt_np) ** 2))
    bias = pred_np.sum() / gt_np.sum() if gt_np.sum() > 0 else 0
    
    print("\n" + "="*50)
    print("📊 RISULTATI STAGE 2 DETTAGLIATI")
    print("="*50)
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Bias: {bias:.3f}")
    print(f"   Loss: {total_loss/len(loader):.4f}")
    print("-"*50)
    
    if sparse_errors:
        print(f"   Sparse (0-100):   MAE={np.mean(sparse_errors):.2f} ({len(sparse_errors)} imgs)")
    if medium_errors:
        print(f"   Medium (100-500): MAE={np.mean(medium_errors):.2f} ({len(medium_errors)} imgs)")
    if dense_errors:
        print(f"   Dense (500+):     MAE={np.mean(dense_errors):.2f} ({len(dense_errors)} imgs)")
    
    # Analisi outlier
    ratios = pred_np / np.maximum(gt_np, 1)
    worst_idx = np.argmax(np.abs(pred_np - gt_np))
    print("-"*50)
    print(f"   Worst case: gt={gt_np[worst_idx]:.0f}, pred={pred_np[worst_idx]:.0f}")
    print(f"   Ratio range: [{ratios.min():.3f}, {ratios.max():.3f}]")
    print("="*50 + "\n")
    
    return total_loss/len(loader), mae, rmse, pred_np.sum(), gt_np.sum()


def train_one_epoch(model, loader, loss_fn, optimizer, device, default_down, epoch, clamp_cfg):
    """Training epoch con monitoring migliorato."""
    model.train()
    total_loss = 0.0
    batch_pred, batch_gt = [], []
    
    pbar = tqdm(loader, desc=f"[P2R Train] Epoch {epoch}")
    
    for batch_idx, (images, _, points) in enumerate(pbar):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        optimizer.zero_grad()
        
        out = model(images)
        pred_density = out.get("p2r_density")
        
        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        
        loss = loss_fn(pred_density, points_list, down=down_tuple)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Clamp log_scale
        if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
            with torch.no_grad():
                min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
                model.p2r_head.log_scale.data.clamp_(min_val, max_val)
        
        total_loss += loss.item()
        
        # Collect per debug
        with torch.no_grad():
            down_h, down_w = down_tuple
            cell_area = down_h * down_w
            for i, pts in enumerate(points_list):
                pred = (pred_density[i].sum() / cell_area).item()
                batch_pred.append(pred)
                batch_gt.append(len(pts))
        
        # Update progress bar
        if hasattr(model.p2r_head, "log_scale"):
            scale = torch.exp(model.p2r_head.log_scale).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", scale=f"{scale:.4f}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Debug ogni 10 epoche
    if epoch % 10 == 0:
        debug_batch_predictions(batch_pred, batch_gt, prefix=f"Epoch {epoch} ")
    
    return total_loss / len(loader)


def main():
    # Carica config
    config_path = "config.yaml"
    if os.path.exists("/home/claude/p2r_fixes/config_fixed.yaml"):
        config_path = "/home/claude/p2r_fixes/config_fixed.yaml"
        print(f"⚠️ Usando config corretto: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Stage 2 P2R Training (FIXED) su {device}")
    
    # Config
    optim_cfg = cfg["OPTIM_P2R"]
    data_cfg = cfg["DATA"]
    p2r_cfg = cfg["P2R_LOSS"]
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    # Dataset
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
        train_ds, batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True, num_workers=optim_cfg["NUM_WORKERS"],
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )
    
    # Modello
    bin_config = cfg["BINS_CONFIG"][cfg["DATASET"]]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=False,  # IMPORTANTE
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs={
            "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
            "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
            "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
            "lambda_noise_std": 0.0,
        },
    ).to(device)
    
    # Carica Stage 1
    stage1_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    zip_ckpt = os.path.join(stage1_dir, "best_model.pth")
    if os.path.isfile(zip_ckpt):
        state = torch.load(zip_ckpt, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"✅ Caricato Stage 1 da {zip_ckpt}")
    else:
        print(f"⚠️ Stage 1 non trovato: {zip_ckpt}")
    
    # CORREZIONE CRITICA: Congela TUTTO tranne P2R head
    print("🧊 Congelamento backbone e ZIP head...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.zip_head.parameters():
        param.requires_grad = False
    for param in model.p2r_head.parameters():
        param.requires_grad = True
    
    # Inizializza log_scale
    log_scale_init = p2r_cfg.get("LOG_SCALE_INIT", -0.5)
    if hasattr(model.p2r_head, "log_scale"):
        model.p2r_head.log_scale.data.fill_(float(log_scale_init))
        print(f"ℹ️ log_scale inizializzato a {log_scale_init}")
    
    # Optimizer - SOLO P2R head
    p2r_params = [p for p in model.p2r_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        p2r_params,
        lr=float(optim_cfg["LR"]),
        weight_decay=float(optim_cfg["WEIGHT_DECAY"])
    )
    scheduler = get_scheduler(optimizer, optim_cfg, optim_cfg["EPOCHS"])
    
    # Loss CORRETTA
    loss_fn = P2RLossFixed(
        scale_weight=float(p2r_cfg.get("SCALE_WEIGHT", 0.005)),
        pos_weight=float(p2r_cfg.get("POS_WEIGHT", 2.0)),
        chunk_size=int(p2r_cfg.get("CHUNK_SIZE", 2048)),
        min_radius=float(p2r_cfg.get("MIN_RADIUS", 8.0)),
        max_radius=float(p2r_cfg.get("MAX_RADIUS", 64.0)),
        use_soft_target=True,
    ).to(device)
    
    # Output directory
    exp_dir = os.path.join(stage1_dir, "stage2")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Calibrazione iniziale
    print("\n🔧 Calibrazione iniziale...")
    clamp_cfg = p2r_cfg.get("LOG_SCALE_CLAMP", [-1.5, 1.0])
    max_adjust = p2r_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA", 2.0)
    
    calibrate_density_scale_v2(
        model, val_loader, device, default_down,
        max_batches=None,
        clamp_range=clamp_cfg,
        max_adjust=max_adjust,
        verbose=True
    )
    
    # Valutazione iniziale
    print("\n📋 Valutazione iniziale:")
    _, init_mae, _, _, _ = evaluate_p2r_detailed(
        model, val_loader, loss_fn, device, default_down
    )
    
    # Training loop
    best_mae = init_mae
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 200)
    no_improve = 0
    
    print(f"\n🚀 Inizio training Stage 2 per {optim_cfg['EPOCHS']} epoche")
    print(f"   scale_weight={p2r_cfg.get('SCALE_WEIGHT', 0.005)}")
    print(f"   clamp_range={clamp_cfg}")
    
    for epoch in range(1, optim_cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            default_down, epoch, clamp_cfg
        )
        
        if scheduler:
            scheduler.step()
        
        # Validazione
        if epoch % optim_cfg["VAL_INTERVAL"] == 0:
            # Ri-calibra prima della validazione
            calibrate_density_scale_v2(
                model, val_loader, device, default_down,
                max_batches=10,
                clamp_range=clamp_cfg,
                max_adjust=0.5,
                verbose=False
            )
            
            val_loss, mae, rmse, tot_pred, tot_gt = evaluate_p2r_detailed(
                model, val_loader, loss_fn, device, default_down
            )
            
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
                  f"Val MAE {mae:.2f} | Best {best_mae:.2f}")
            
            # Early stopping basato su MAE
            if mae < best_mae:
                best_mae = mae
                no_improve = 0
                
                # Salva best
                best_path = os.path.join(stage1_dir, "stage2_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"💾 Nuovo best salvato: MAE={best_mae:.2f}")
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"⛔ Early stopping dopo {patience} validazioni senza miglioramento")
                break
    
    print(f"\n✅ Stage 2 completato. Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()