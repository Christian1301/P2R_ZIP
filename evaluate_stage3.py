# evaluate_stage3_fixed.py
# -*- coding: utf-8 -*-
"""
Valutazione Stage 3 - VERSIONE CORRETTA

PROBLEMA RISOLTO:
La versione originale NON applicava la maschera π al conteggio,
risultando in MAE ~2x più alto del valore reale.

CORREZIONI:
1. Usa collate_fn (padding) invece di collate_joint (resize)
2. Applica soft_mask nel calcolo del conteggio (come nel training)
3. Mostra confronto masked vs raw
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


def get_soft_mask(logit_pi_maps, temperature: float = 1.0):
    """
    Crea soft mask dal π-head usando sigmoid.
    IDENTICA alla funzione in train_stage3_joint.py
    """
    logit_pieno = logit_pi_maps[:, 1:2, :, :]  # Canale "pieno"
    soft_mask = torch.sigmoid(logit_pieno / temperature)
    return soft_mask


def compute_masked_count(pred_density, soft_mask, down_h, down_w):
    """
    Calcola il conteggio usando la density mascherata.
    IDENTICA alla funzione in train_stage3_joint.py
    """
    cell_area = down_h * down_w
    
    # Upsample mask se necessario
    if soft_mask.shape[-2:] != pred_density.shape[-2:]:
        soft_mask = F.interpolate(
            soft_mask, 
            size=pred_density.shape[-2:], 
            mode='bilinear',
            align_corners=False
        )
    
    masked_density = pred_density * soft_mask
    count = torch.sum(masked_density, dim=(1, 2, 3)) / cell_area
    
    return count, masked_density


@torch.no_grad()
def evaluate_stage3(model, dataloader, device, default_down, temperature=1.0):
    """
    Valutazione Stage 3 con maschera π applicata.
    
    Mostra sia MAE masked (metrica reale) che MAE raw (riferimento).
    """
    model.eval()
    
    # Metriche
    mae_masked, mae_raw = 0.0, 0.0
    mse_masked = 0.0
    total_pred_masked, total_pred_raw, total_gt = 0.0, 0.0, 0.0
    total_coverage, total_recall = 0.0, 0.0
    n_samples = 0
    
    # Per analisi stratificata
    errors_by_density = {"sparse": [], "medium": [], "dense": []}
    
    print("\n===== VALUTAZIONE STAGE 3 (con maschera π) =====")

    for images, gt_density, points in tqdm(dataloader, desc="[Eval Stage 3]"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        
        # === P2R Density ===
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # === Conteggio RAW (senza maschera) ===
        pred_count_raw = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        # === Conteggio MASKED (con maschera π) - COME NEL TRAINING ===
        logit_pi = outputs["logit_pi_maps"]
        soft_mask = get_soft_mask(logit_pi, temperature=temperature)
        pred_count_masked, _ = compute_masked_count(pred_density, soft_mask, down_h, down_w)
        
        # === Coverage & Recall del π-head ===
        gt_occupancy = (F.avg_pool2d(gt_density, 16, 16) * 256 > 0.5).float()
        pred_occupancy = (soft_mask > 0.5).float()
        
        if pred_occupancy.shape[-2:] != gt_occupancy.shape[-2:]:
            pred_occupancy_resized = F.interpolate(pred_occupancy, gt_occupancy.shape[-2:], mode='nearest')
        else:
            pred_occupancy_resized = pred_occupancy
        
        coverage = pred_occupancy.mean().item() * 100
        if gt_occupancy.sum() > 0:
            recall = (pred_occupancy_resized * gt_occupancy).sum() / gt_occupancy.sum() * 100
        else:
            recall = 100.0
        
        # === Accumula metriche per sample ===
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_m = pred_count_masked[idx].item()
            pred_r = pred_count_raw[idx].item()
            
            err_m = abs(pred_m - gt)
            err_r = abs(pred_r - gt)
            
            mae_masked += err_m
            mae_raw += err_r
            mse_masked += err_m ** 2
            total_pred_masked += pred_m
            total_pred_raw += pred_r
            total_gt += gt
            total_coverage += coverage
            total_recall += recall if isinstance(recall, float) else recall.item()
            n_samples += 1
            
            # Stratificazione
            if gt <= 100:
                errors_by_density["sparse"].append(err_m)
            elif gt <= 500:
                errors_by_density["medium"].append(err_m)
            else:
                errors_by_density["dense"].append(err_m)

    # === Calcola metriche finali ===
    mae_masked /= n_samples
    mae_raw /= n_samples
    rmse = np.sqrt(mse_masked / n_samples)
    bias_masked = total_pred_masked / total_gt if total_gt > 0 else 0
    bias_raw = total_pred_raw / total_gt if total_gt > 0 else 0
    coverage = total_coverage / n_samples
    recall = total_recall / n_samples

    # === Report ===
    print("\n" + "="*60)
    print("📊 RISULTATI STAGE 3")
    print("="*60)
    print(f"   MAE MASKED:  {mae_masked:.2f}  ← METRICA PRINCIPALE")
    print(f"   MAE RAW:     {mae_raw:.2f}  (riferimento, senza maschera)")
    print(f"   RMSE:        {rmse:.2f}")
    print(f"   Bias masked: {bias_masked:.3f}")
    print(f"   Bias raw:    {bias_raw:.3f}")
    print("-"*60)
    print(f"   π coverage:  {coverage:.1f}%")
    print(f"   π recall:    {recall:.1f}%")
    print("-"*60)
    
    # Analisi stratificata
    print("   Per densità (masked):")
    for name, errors in errors_by_density.items():
        if errors:
            print(f"      {name}: MAE={np.mean(errors):.1f} ({len(errors)} imgs)")
    
    # Confronto masked vs raw
    print("-"*60)
    if mae_masked < mae_raw:
        delta = mae_raw - mae_masked
        print(f"   ✅ π-head AIUTA! (masked < raw di {delta:.1f})")
    else:
        delta = mae_masked - mae_raw
        print(f"   ⚠️ π-head peggiora di {delta:.1f}")
    
    print("="*60 + "\n")
    
    return {
        "mae_masked": mae_masked,
        "mae_raw": mae_raw,
        "rmse": rmse,
        "bias_masked": bias_masked,
        "bias_raw": bias_raw,
        "coverage": coverage,
        "recall": recall,
    }


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
        
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])

    # === Dataset ===
    data_cfg = cfg["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    # IMPORTANTE: Usa collate_fn (padding) come nel training!
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["OPTIM_JOINT"]["NUM_WORKERS"],
        collate_fn=collate_fn,  # NON collate_joint!
        pin_memory=True
    )

    # === Modello ===
    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]
    
    zip_head_kwargs = {
        "lambda_scale": cfg["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": cfg["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": cfg["ZIP_HEAD"].get("USE_SOFTPLUS", True),
        "lambda_noise_std": 0.0
    }

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"].get("UPSAMPLE_TO_INPUT", False),
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # === Carica checkpoint ===
    ckpt_path = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"], "stage3_best.pth")
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
        print(f"⚠️ load_state_dict: missing={missing.missing_keys}, unexpected={missing.unexpected_keys}")

    # === Valutazione ===
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    # Usa la stessa temperature del training
    joint_cfg = cfg.get("JOINT_LOSS", {})
    temperature = float(joint_cfg.get("MASK_TEMPERATURE", 1.0))
    
    results = evaluate_stage3(model, val_loader, device, default_down, temperature)
    
    # === Confronto con Stage 4 se disponibile ===
    stage4_path = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"], "stage4_best.pth")
    if os.path.exists(stage4_path):
        print("\n📋 Per confronto, esegui anche: python evaluate_stage4.py")


if __name__ == "__main__":
    main()