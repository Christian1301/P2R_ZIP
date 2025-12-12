# evaluate_stage3_v2.py
# -*- coding: utf-8 -*-
"""
Valutazione Stage 3 - VERSIONE CORRETTA V2

STRATEGIA:
- Metrica PRINCIPALE: MAE da density RAW (senza maschera π)
- Metrica SECONDARIA: MAE masked (per confronto)
- Il π-head serve per localizzazione, non per il conteggio finale

Questo allinea la valutazione con la nuova strategia di training.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn


def compute_count_raw(pred_density, down_h, down_w):
    """Conteggio dalla density RAW."""
    cell_area = down_h * down_w
    return torch.sum(pred_density, dim=(1, 2, 3)) / cell_area


def compute_count_masked(pred_density, pi_probs, down_h, down_w, threshold=0.5):
    """Conteggio con maschera π."""
    cell_area = down_h * down_w
    
    if pi_probs.shape[-2:] != pred_density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=pred_density.shape[-2:], 
            mode='bilinear', align_corners=False
        )
    
    mask = (pi_probs > threshold).float()
    masked_density = pred_density * mask
    return torch.sum(masked_density, dim=(1, 2, 3)) / cell_area


@torch.no_grad()
def evaluate_stage3(model, dataloader, device, default_down, pi_threshold=0.5):
    """
    Valutazione completa Stage 3.
    
    Mostra:
    - MAE RAW (metrica principale)
    - MAE MASKED (per confronto)
    - Analisi stratificata per densità
    - Metriche π-head (coverage, recall)
    """
    model.eval()
    
    # Accumulatori
    mae_raw, mae_masked = 0.0, 0.0
    mse_raw = 0.0
    total_pred_raw, total_pred_masked, total_gt = 0.0, 0.0, 0.0
    total_coverage, total_recall, total_precision = 0.0, 0.0, 0.0
    n_samples = 0
    
    # Per analisi stratificata
    errors_raw = {"sparse": [], "medium": [], "dense": []}
    errors_masked = {"sparse": [], "medium": [], "dense": []}
    
    # Per analisi bias
    all_pred_raw, all_gt = [], []

    print("\n" + "="*70)
    print("📊 VALUTAZIONE STAGE 3 (V2 - RAW COUNT STRATEGY)")
    print("="*70)

    for images, gt_density, points in tqdm(dataloader, desc="[Eval Stage 3]"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        
        # P2R Density
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        
        # Count RAW (PRINCIPALE)
        pred_count_raw = compute_count_raw(pred_density, down_h, down_w)
        
        # Count MASKED (per confronto)
        pi_probs = outputs.get("pi_probs")
        if pi_probs is None:
            logit_pi = outputs["logit_pi_maps"]
            pi_probs = logit_pi.softmax(dim=1)[:, 1:2]
        
        pred_count_masked = compute_count_masked(
            pred_density, pi_probs, down_h, down_w, threshold=pi_threshold
        )
        
        # π-head metrics
        gt_occupancy = (F.avg_pool2d(gt_density, 16, 16) * 256 > 0.5).float()
        pred_occupancy = (pi_probs > 0.5).float()
        
        if pred_occupancy.shape[-2:] != gt_occupancy.shape[-2:]:
            pred_occupancy_resized = F.interpolate(
                pred_occupancy, gt_occupancy.shape[-2:], mode='nearest'
            )
        else:
            pred_occupancy_resized = pred_occupancy
        
        coverage = pred_occupancy.mean().item() * 100
        
        if gt_occupancy.sum() > 0:
            tp = (pred_occupancy_resized * gt_occupancy).sum()
            recall = (tp / gt_occupancy.sum()).item() * 100
        else:
            recall = 100.0
        
        if pred_occupancy.sum() > 0:
            precision = (tp / pred_occupancy.sum()).item() * 100 if gt_occupancy.sum() > 0 else 0
        else:
            precision = 100.0
        
        # Per-sample metrics
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_r = pred_count_raw[idx].item()
            pred_m = pred_count_masked[idx].item()
            
            err_r = abs(pred_r - gt)
            err_m = abs(pred_m - gt)
            
            mae_raw += err_r
            mae_masked += err_m
            mse_raw += err_r ** 2
            total_pred_raw += pred_r
            total_pred_masked += pred_m
            total_gt += gt
            total_coverage += coverage
            total_recall += recall
            total_precision += precision
            n_samples += 1
            
            all_pred_raw.append(pred_r)
            all_gt.append(gt)
            
            # Stratificazione
            if gt <= 100:
                errors_raw["sparse"].append(err_r)
                errors_masked["sparse"].append(err_m)
            elif gt <= 500:
                errors_raw["medium"].append(err_r)
                errors_masked["medium"].append(err_m)
            else:
                errors_raw["dense"].append(err_r)
                errors_masked["dense"].append(err_m)

    # Calcola metriche finali
    mae_raw /= n_samples
    mae_masked /= n_samples
    rmse = np.sqrt(mse_raw / n_samples)
    bias_raw = total_pred_raw / total_gt if total_gt > 0 else 0
    bias_masked = total_pred_masked / total_gt if total_gt > 0 else 0
    coverage = total_coverage / n_samples
    recall = total_recall / n_samples
    precision = total_precision / n_samples

    # === REPORT ===
    print(f"\n{'─'*70}")
    print(f"🎯 METRICHE PRINCIPALI (Density RAW)")
    print(f"{'─'*70}")
    print(f"   MAE:  {mae_raw:.2f}  ← USA QUESTA PER CONFRONTI")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Bias: {bias_raw:.3f} (1.0 = perfetto)")
    
    print(f"\n{'─'*70}")
    print(f"📈 CONFRONTO RAW vs MASKED")
    print(f"{'─'*70}")
    print(f"   MAE RAW:    {mae_raw:.2f}")
    print(f"   MAE MASKED: {mae_masked:.2f}")
    
    if mae_raw < mae_masked:
        delta = mae_masked - mae_raw
        print(f"   ✅ RAW è MEGLIO di {delta:.1f} punti")
        print(f"   → Usa density RAW per inference")
    else:
        delta = mae_raw - mae_masked
        print(f"   ✅ MASKED è meglio di {delta:.1f} punti")
        print(f"   → Il π-head aiuta, usa masked per inference")
    
    print(f"\n{'─'*70}")
    print(f"🎭 METRICHE π-HEAD")
    print(f"{'─'*70}")
    print(f"   Coverage:  {coverage:.1f}% (% blocchi attivi)")
    print(f"   Recall:    {recall:.1f}% (% persone non mascherate)")
    print(f"   Precision: {precision:.1f}% (% blocchi attivi corretti)")
    
    print(f"\n{'─'*70}")
    print(f"📊 ANALISI STRATIFICATA")
    print(f"{'─'*70}")
    
    for density_range, (name_raw, name_masked) in [
        ("sparse", ("Sparse (0-100)", "Sparse")),
        ("medium", ("Medium (100-500)", "Medium")),
        ("dense", ("Dense (500+)", "Dense")),
    ]:
        errs_r = errors_raw[density_range]
        errs_m = errors_masked[density_range]
        if errs_r:
            mae_r = np.mean(errs_r)
            mae_m = np.mean(errs_m)
            better = "RAW" if mae_r < mae_m else "MASKED"
            print(f"   {name_raw}:")
            print(f"      RAW:    {mae_r:.1f}")
            print(f"      MASKED: {mae_m:.1f}")
            print(f"      Meglio: {better} ({len(errs_r)} imgs)")
    
    # Bias analysis dettagliata
    pred_np = np.array(all_pred_raw)
    gt_np = np.array(all_gt)
    
    print(f"\n{'─'*70}")
    print(f"🔍 ANALISI BIAS DETTAGLIATA")
    print(f"{'─'*70}")
    
    # Sottostima vs Sovrastima
    underestimate = pred_np < gt_np * 0.9
    overestimate = pred_np > gt_np * 1.1
    accurate = ~underestimate & ~overestimate
    
    print(f"   Sottostima (<0.9x): {underestimate.sum()} ({underestimate.mean()*100:.1f}%)")
    print(f"   Accurato (0.9-1.1x): {accurate.sum()} ({accurate.mean()*100:.1f}%)")
    print(f"   Sovrastima (>1.1x): {overestimate.sum()} ({overestimate.mean()*100:.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    return {
        "mae": mae_raw,  # Metrica principale
        "mae_raw": mae_raw,
        "mae_masked": mae_masked,
        "rmse": rmse,
        "bias": bias_raw,
        "coverage": coverage,
        "recall": recall,
        "precision": precision,
    }


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
        
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])

    # Dataset
    data_cfg = cfg["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("OPTIM_JOINT", {}).get("NUM_WORKERS", 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Modello
    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]
    
    zip_head_kwargs = {
        "lambda_scale": cfg["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": cfg["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": cfg["ZIP_HEAD"].get("USE_SOFTPLUS", True),
        "lambda_noise_std": 0.0
    }

    # Controlla se usare STE
    use_ste = cfg.get("MODEL", {}).get("USE_STE_MASK", True)

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"].get("UPSAMPLE_TO_INPUT", False),
        use_ste_mask=use_ste,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Carica checkpoint
    ckpt_path = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"], "stage3_v5_best.pth")
    
    if not os.path.exists(ckpt_path):
        # Prova latest
        ckpt_path = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"], "stage3_v5_latest.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint non trovato: {ckpt_path}")
        return

    print(f"✅ Caricamento {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"⚠️ load_state_dict: missing={len(missing.missing_keys)}, "
              f"unexpected={len(missing.unexpected_keys)}")

    # Valutazione
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    pi_threshold = cfg.get("JOINT_LOSS", {}).get("PI_THRESHOLD", 0.5)
    
    results = evaluate_stage3(model, val_loader, device, default_down, pi_threshold)
    
    # Summary finale
    print("📋 SUMMARY")
    print(f"   Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"   MAE (RAW): {results['mae']:.2f}")
    print(f"   Target: 60-70")
    
    if results['mae'] <= 70:
        print("   ✅ TARGET RAGGIUNTO!")
    else:
        print(f"   ⚠️ Gap dal target: {results['mae'] - 70:.1f}")


if __name__ == "__main__":
    main()