# train_stage2_p2r.py
# -*- coding: utf-8 -*-
"""
Stage 2 P2R Training - VERSIONE CORRETTA V2

PROBLEMA RISOLTO:
La versione originale usava BCE loss con normalizzazione, che eliminava
l'informazione sulla magnitudine. Il modello prediceva sempre ~3-4 persone.

SOLUZIONE:
- Loss con supervisione DIRETTA sul conteggio (L1)
- log_scale inizializzato alto e senza clamp restrittivo
- Calibrazione opzionale ma non critica (la loss fa il lavoro)
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
from losses.p2r_region_loss import P2RLoss
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    collate_fn,
    canonicalize_p2r_grid,
    save_checkpoint,
)


@torch.no_grad()
def calibrate_density_scale(
    model,
    loader,
    device,
    default_down,
    max_batches=None,
    clamp_range=None,
    max_adjust=2.0,
    bias_eps=0.1,
    verbose=True,
):
    """
    Calibrazione della scala basata sul bias mediano.
    Con la nuova loss, questa funzione è meno critica.
    """
    if not hasattr(model, "p2r_head") or not hasattr(model.p2r_head, "log_scale"):
        return None

    model.eval()
    pred_counts = []
    gt_counts = []
    ratios = []

    for batch_idx, (images, _, points) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)
        points_list = [p.to(device) for p in points]

        outputs = model(images)
        pred_density = outputs.get("p2r_density", outputs.get("density"))
        if pred_density is None:
            continue

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w

        for i, pts in enumerate(points_list):
            gt = len(pts)
            if gt == 0:
                continue
            
            pred = (pred_density[i].sum() / cell_area).item()
            pred_counts.append(pred)
            gt_counts.append(gt)
            if gt > 0:
                ratios.append(pred / gt)

    if len(gt_counts) == 0:
        if verbose:
            print("ℹ️ Calibrazione saltata: nessun dato valido")
        return None

    ratios_np = np.array(ratios)
    bias_median = np.median(ratios_np) if ratios_np.size > 0 else 1.0
    
    if verbose:
        print(f"\n📊 Statistiche Calibrazione:")
        print(f"   Bias mediano: {bias_median:.3f}")
        print(f"   Ratio range: [{ratios_np.min():.3f}, {ratios_np.max():.3f}]")

    # Se il bias è già vicino a 1, non fare nulla
    if abs(bias_median - 1.0) < bias_eps:
        if verbose:
            print(f"ℹ️ Calibrazione: bias già accettabile ({bias_median:.3f})")
        return bias_median

    # Applica correzione
    prev_log_scale = float(model.p2r_head.log_scale.detach().item())
    raw_adjust = float(np.log(max(bias_median, 0.01)))
    
    if max_adjust is not None:
        adjust = float(np.clip(raw_adjust, -max_adjust, max_adjust))
    else:
        adjust = raw_adjust
    
    model.p2r_head.log_scale.data -= torch.tensor(adjust, device=device)
    
    # Clamp opzionale (range MOLTO più ampio)
    if clamp_range is not None:
        min_val, max_val = float(clamp_range[0]), float(clamp_range[1])
        model.p2r_head.log_scale.data.clamp_(min_val, max_val)
    
    new_log_scale = float(model.p2r_head.log_scale.detach().item())
    new_scale = float(torch.exp(model.p2r_head.log_scale.detach()).item())
    
    if verbose:
        print(f"🔧 Calibrazione: bias={bias_median:.3f} → "
              f"log_scale {prev_log_scale:.2f}→{new_log_scale:.2f} (scala={new_scale:.2f})")
    
    return bias_median


@torch.no_grad()
def evaluate_p2r(model, loader, loss_fn, device, default_down):
    """Valutazione con statistiche dettagliate."""
    model.eval()
    
    all_pred, all_gt = [], []
    sparse_err, medium_err, dense_err = [], [], []
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
            
            err = abs(pred - gt)
            if gt <= 100:
                sparse_err.append(err)
            elif gt <= 500:
                medium_err.append(err)
            else:
                dense_err.append(err)
    
    pred_np = np.array(all_pred)
    gt_np = np.array(all_gt)
    
    mae = np.mean(np.abs(pred_np - gt_np))
    rmse = np.sqrt(np.mean((pred_np - gt_np) ** 2))
    bias = pred_np.sum() / max(gt_np.sum(), 1)
    
    print("\n" + "="*60)
    print("📊 RISULTATI STAGE 2")
    print("="*60)
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Bias: {bias:.3f} (1.0 = perfetto)")
    print(f"   Loss: {total_loss/len(loader):.4f}")
    print("-"*60)
    
    if sparse_err:
        print(f"   Sparse (0-100):   MAE={np.mean(sparse_err):.2f} ({len(sparse_err)} imgs)")
    if medium_err:
        print(f"   Medium (100-500): MAE={np.mean(medium_err):.2f} ({len(medium_err)} imgs)")
    if dense_err:
        print(f"   Dense (500+):     MAE={np.mean(dense_err):.2f} ({len(dense_err)} imgs)")
    
    ratios = pred_np / np.maximum(gt_np, 1)
    print("-"*60)
    print(f"   Ratio pred/gt: mean={ratios.mean():.3f}, std={ratios.std():.3f}")
    print("="*60 + "\n")
    
    return total_loss/len(loader), mae, rmse, pred_np.sum(), gt_np.sum()


def train_one_epoch(model, loader, loss_fn, optimizer, device, default_down, epoch):
    """Training epoch."""
    model.train()
    total_loss = 0.0
    batch_pred, batch_gt = [], []
    
    pbar = tqdm(loader, desc=f"[P2R Train] Epoch {epoch}")
    
    for images, _, points in pbar:
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
        
        total_loss += loss.item()
        
        # Stats
        with torch.no_grad():
            down_h, down_w = down_tuple
            cell_area = down_h * down_w
            for i, pts in enumerate(points_list):
                pred = (pred_density[i].sum() / cell_area).item()
                batch_pred.append(pred)
                batch_gt.append(len(pts))
        
        # Progress bar
        if hasattr(model.p2r_head, "log_scale"):
            scale = torch.exp(model.p2r_head.log_scale).item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", scale=f"{scale:.1f}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    
    # Stats fine epoca
    if batch_pred and epoch % 10 == 0:
        pred_np = np.array(batch_pred)
        gt_np = np.array(batch_gt)
        train_mae = np.mean(np.abs(pred_np - gt_np))
        train_ratio = pred_np.sum() / max(gt_np.sum(), 1)
        print(f"   Train MAE: {train_mae:.2f}, Ratio: {train_ratio:.3f}")
    
    return total_loss / len(loader)


def main():
    # Config
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Stage 2 P2R Training su {device}")
    
    optim_cfg = cfg["OPTIM_P2R"]
    data_cfg = cfg["DATA"]
    p2r_cfg = cfg.get("P2R_LOSS", {})
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
        upsample_to_input=False,
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
    
    # Congela backbone e ZIP head
    print("🧊 Congelamento backbone e ZIP head...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.zip_head.parameters():
        param.requires_grad = False
    for param in model.p2r_head.parameters():
        param.requires_grad = True
    
    # Il nuovo p2r_head.py ha già log_scale=4.0
    # Ma se usiamo un checkpoint vecchio, forziamo il reset
    if hasattr(model.p2r_head, "log_scale"):
        current_scale = model.p2r_head.log_scale.item()
        if current_scale < 2.0:  # Se è ancora il vecchio valore
            model.p2r_head.log_scale.data.fill_(4.0)
            print(f"🔧 Reset log_scale: {current_scale:.2f} → 4.0")
        print(f"   log_scale: {model.p2r_head.log_scale.item():.2f} "
              f"(scala: {torch.exp(model.p2r_head.log_scale).item():.1f})")
    
    # Optimizer
    p2r_params = [p for p in model.p2r_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        p2r_params,
        lr=float(optim_cfg.get("LR", 8e-5)),
        weight_decay=float(optim_cfg.get("WEIGHT_DECAY", 1e-4))
    )
    scheduler = get_scheduler(optimizer, optim_cfg, optim_cfg["EPOCHS"])
    
    # Loss con supervisione diretta sul conteggio
    loss_fn = P2RLoss(
        count_weight=float(p2r_cfg.get("COUNT_WEIGHT", 2.0)),
        scale_weight=float(p2r_cfg.get("SCALE_WEIGHT", 0.5)),
        spatial_weight=float(p2r_cfg.get("SPATIAL_WEIGHT", 0.1)),
        min_radius=float(p2r_cfg.get("MIN_RADIUS", 8.0)),
    ).to(device)
    
    print(f"\n📋 Loss weights: count={loss_fn.count_weight}, "
          f"scale={loss_fn.scale_weight}, spatial={loss_fn.spatial_weight}")
    
    # Valutazione iniziale
    print("\n📋 Valutazione iniziale:")
    _, init_mae, _, _, _ = evaluate_p2r(model, val_loader, loss_fn, device, default_down)
    
    # Training loop
    best_mae = init_mae
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 100)
    no_improve = 0
    
    print(f"\n🚀 Inizio training per {optim_cfg['EPOCHS']} epoche")
    
    for epoch in range(1, optim_cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            default_down, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        # Validazione
        if epoch % optim_cfg.get("VAL_INTERVAL", 5) == 0:
            val_loss, mae, rmse, tot_pred, tot_gt = evaluate_p2r(
                model, val_loader, loss_fn, device, default_down
            )
            
            scale = model.p2r_head.log_scale.item() if hasattr(model.p2r_head, "log_scale") else 0
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
                  f"Val MAE {mae:.2f} | Best {best_mae:.2f} | "
                  f"log_scale {scale:.2f}")
            
            if mae < best_mae:
                best_mae = mae
                no_improve = 0
                
                best_path = os.path.join(stage1_dir, "stage2_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"💾 Nuovo best: MAE={best_mae:.2f}")
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"⛔ Early stopping dopo {patience} validazioni")
                break
    
    print(f"\n✅ Stage 2 completato. Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()