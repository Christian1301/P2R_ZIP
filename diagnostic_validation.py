#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNOSTIC SCRIPT - Confronto Metodi di Validazione

Questo script carica stage2_best.pth e valida con DIVERSI metodi
per identificare la causa della discrepanza:

- Stage 2: MAE 87.77, Bias 0.956
- Stage 3: MAE 103.17, Bias 1.106 (stesso checkpoint!)

Possibili cause:
1. cell_area / downsampling diverso
2. upsample_to_input True vs False
3. Transforms di validazione diversi
4. Modo di calcolare il count
"""

import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn


def compute_count_method1(pred_density, down_h, down_w):
    """Metodo Stage 3/4: usa canonicalize cell_area."""
    cell_area = down_h * down_w
    return torch.sum(pred_density, dim=(1, 2, 3)) / cell_area


def compute_count_method2(pred_density):
    """Metodo semplice: somma diretta (assume density già normalizzata)."""
    return torch.sum(pred_density, dim=(1, 2, 3))


def compute_count_method3(pred_density, input_h, input_w):
    """Metodo con upsampling: upsample a input size, poi somma."""
    upsampled = F.interpolate(pred_density, size=(input_h, input_w), mode='bilinear', align_corners=False)
    return torch.sum(upsampled, dim=(1, 2, 3))


def compute_count_method4(pred_density, scale_factor):
    """Metodo con scale factor fisso."""
    return torch.sum(pred_density, dim=(1, 2, 3)) * scale_factor


@torch.no_grad()
def diagnostic_validation(model, dataloader, device, default_down, method_name, count_fn):
    """Validazione con metodo specifico."""
    model.eval()
    
    total_mae = 0.0
    total_pred = 0.0
    total_gt = 0.0
    n_samples = 0
    
    all_preds = []
    all_gts = []
    
    for images, gt_density, points in dataloader:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        pred_density = outputs["p2r_density"]
        
        _, _, h_in, w_in = images.shape
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            
            # Calcola count con il metodo specificato
            pred_count = count_fn(pred_density[idx:idx+1], h_in, w_in, default_down)
            pred = pred_count.item()
            
            total_mae += abs(pred - gt)
            total_pred += pred
            total_gt += gt
            n_samples += 1
            
            all_preds.append(pred)
            all_gts.append(gt)
    
    mae = total_mae / n_samples
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    return {
        "method": method_name,
        "mae": mae,
        "bias": bias,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "preds": all_preds,
        "gts": all_gts,
    }


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    
    print("="*70)
    print("🔍 DIAGNOSTIC - Confronto Metodi di Validazione")
    print("="*70)
    
    # Dataset
    data_cfg = config["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"\n📊 Dataset: {len(val_dataset)} immagini di validazione")
    
    # Output directory
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    print(f"📁 Output dir: {output_dir}")
    print(f"📐 Default downsample: {default_down}")
    
    # =========================================
    # TEST 1: Modello con upsample_to_input=False (come Stage 3/4)
    # =========================================
    print("\n" + "="*70)
    print("TEST 1: upsample_to_input=FALSE (come Stage 3/4)")
    print("="*70)
    
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": config["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": config["ZIP_HEAD"].get("USE_SOFTPLUS", True),
    }
    
    model_no_upsample = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,  # Come Stage 3/4
        use_ste_mask=True,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # Carica stage2_best
    stage2_path = os.path.join(output_dir, "stage2_best.pth")
    if os.path.isfile(stage2_path):
        print(f"\n✅ Caricamento: {stage2_path}")
        state = torch.load(stage2_path, map_location=device)
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        model_no_upsample.load_state_dict(state, strict=False)
    else:
        print(f"❌ {stage2_path} non trovato!")
        return
    
    model_no_upsample.eval()
    
    # Metodi di calcolo count
    methods_no_upsample = []
    
    # Metodo A: canonicalize + cell_area (come Stage 3/4)
    def count_method_A(density, h_in, w_in, default_down):
        density_canon, down_tuple, _ = canonicalize_p2r_grid(density, (h_in, w_in), default_down)
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        return torch.sum(density_canon, dim=(1, 2, 3)) / cell_area
    
    # Metodo B: somma diretta
    def count_method_B(density, h_in, w_in, default_down):
        return torch.sum(density, dim=(1, 2, 3))
    
    # Metodo C: con scale factor = default_down^2
    def count_method_C(density, h_in, w_in, default_down):
        return torch.sum(density, dim=(1, 2, 3)) / (default_down ** 2)
    
    # Metodo D: density.shape based
    def count_method_D(density, h_in, w_in, default_down):
        _, _, h_d, w_d = density.shape
        scale_h = h_in / h_d
        scale_w = w_in / w_d
        cell_area = scale_h * scale_w
        return torch.sum(density, dim=(1, 2, 3)) / cell_area
    
    print("\n🧪 Testing metodi con upsample_to_input=FALSE...")
    
    for name, fn in [
        ("A: canonicalize + cell_area", count_method_A),
        ("B: somma diretta", count_method_B),
        ("C: / default_down²", count_method_C),
        ("D: density.shape based", count_method_D),
    ]:
        result = diagnostic_validation(model_no_upsample, val_loader, device, default_down, name, fn)
        methods_no_upsample.append(result)
        print(f"   {name}: MAE={result['mae']:.2f}, Bias={result['bias']:.3f}")
    
    # =========================================
    # TEST 2: Modello con upsample_to_input=True
    # =========================================
    print("\n" + "="*70)
    print("TEST 2: upsample_to_input=TRUE")
    print("="*70)
    
    model_with_upsample = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=True,  # Diverso!
        use_ste_mask=True,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    model_with_upsample.load_state_dict(state, strict=False)
    model_with_upsample.eval()
    
    # Metodo E: con upsample, somma diretta
    def count_method_E(density, h_in, w_in, default_down):
        # density è già upsampled a (h_in, w_in)
        return torch.sum(density, dim=(1, 2, 3))
    
    # Metodo F: con upsample, / (h*w) per normalizzare
    def count_method_F(density, h_in, w_in, default_down):
        return torch.sum(density, dim=(1, 2, 3)) / (h_in * w_in) * (h_in * w_in / (default_down ** 2))
    
    print("\n🧪 Testing metodi con upsample_to_input=TRUE...")
    
    methods_with_upsample = []
    for name, fn in [
        ("E: upsample + somma diretta", count_method_E),
        ("F: upsample + normalized", count_method_F),
    ]:
        result = diagnostic_validation(model_with_upsample, val_loader, device, default_down, name, fn)
        methods_with_upsample.append(result)
        print(f"   {name}: MAE={result['mae']:.2f}, Bias={result['bias']:.3f}")
    
    # =========================================
    # TEST 3: Verifica dimensioni density
    # =========================================
    print("\n" + "="*70)
    print("TEST 3: Dimensioni Density Map")
    print("="*70)
    
    # Prendi una immagine di test
    test_img, test_gt_density, test_points = next(iter(val_loader))
    test_img = test_img.to(device)
    
    print(f"\n📐 Input image shape: {test_img.shape}")
    
    with torch.no_grad():
        out_no_up = model_no_upsample(test_img)
        out_with_up = model_with_upsample(test_img)
    
    print(f"   Density (upsample=False): {out_no_up['p2r_density'].shape}")
    print(f"   Density (upsample=True):  {out_with_up['p2r_density'].shape}")
    
    _, _, h_in, w_in = test_img.shape
    _, _, h_d, w_d = out_no_up['p2r_density'].shape
    
    print(f"\n📊 Scale factors:")
    print(f"   h_in/h_d = {h_in}/{h_d} = {h_in/h_d:.2f}")
    print(f"   w_in/w_d = {w_in}/{w_d} = {w_in/w_d:.2f}")
    print(f"   cell_area = {(h_in/h_d) * (w_in/w_d):.2f}")
    print(f"   default_down² = {default_down**2}")
    
    # =========================================
    # RIEPILOGO
    # =========================================
    print("\n" + "="*70)
    print("📊 RIEPILOGO RISULTATI")
    print("="*70)
    
    print("\n┌─────────────────────────────────────┬─────────┬─────────┐")
    print("│ Metodo                              │   MAE   │  Bias   │")
    print("├─────────────────────────────────────┼─────────┼─────────┤")
    
    all_results = methods_no_upsample + methods_with_upsample
    
    # Target values
    stage2_target_mae = 87.77
    stage3_observed_mae = 103.17
    
    for r in all_results:
        mae_marker = ""
        if abs(r['mae'] - stage2_target_mae) < 2:
            mae_marker = " ← Stage2!"
        elif abs(r['mae'] - stage3_observed_mae) < 2:
            mae_marker = " ← Stage3!"
        
        print(f"│ {r['method']:<35} │ {r['mae']:>7.2f} │ {r['bias']:>7.3f} │{mae_marker}")
    
    print("└─────────────────────────────────────┴─────────┴─────────┘")
    
    print(f"\n🎯 Target Stage 2: MAE={stage2_target_mae}, Bias=0.956")
    print(f"🎯 Osservato Stage 3: MAE={stage3_observed_mae}, Bias=1.106")
    
    # Trova il metodo più vicino a Stage 2
    best_match = min(all_results, key=lambda x: abs(x['mae'] - stage2_target_mae))
    print(f"\n✅ Metodo più vicino a Stage 2: {best_match['method']}")
    print(f"   MAE={best_match['mae']:.2f}, Bias={best_match['bias']:.3f}")
    
    # =========================================
    # TEST 4: Confronto sample-by-sample
    # =========================================
    print("\n" + "="*70)
    print("TEST 4: Confronto Sample-by-Sample (primi 5)")
    print("="*70)
    
    # Usa metodo A (Stage 3/4 style) e metodo che matcha Stage 2
    method_stage3 = methods_no_upsample[0]  # A: canonicalize
    
    print(f"\n{'Sample':<8} {'GT':<8} {'Pred A':<10} {'Err A':<8}")
    print("-" * 40)
    
    for i in range(min(5, len(method_stage3['gts']))):
        gt = method_stage3['gts'][i]
        pred_a = method_stage3['preds'][i]
        err_a = abs(pred_a - gt)
        print(f"{i:<8} {gt:<8.0f} {pred_a:<10.1f} {err_a:<8.1f}")
    
    # =========================================
    # RACCOMANDAZIONE
    # =========================================
    print("\n" + "="*70)
    print("💡 RACCOMANDAZIONE")
    print("="*70)
    
    if best_match['method'].startswith("E") or best_match['method'].startswith("F"):
        print("""
Il metodo Stage 2 probabilmente usa upsample_to_input=True.
Per allineare Stage 3/4, dovresti:

1. Usare upsample_to_input=True nel modello
2. Oppure adattare il calcolo del count

SOLUZIONE: Modifica P2R_ZIP_Model in Stage 3/4 per usare
           upsample_to_input=True (come Stage 2)
""")
    else:
        print(f"""
Il metodo più vicino è: {best_match['method']}

La discrepanza potrebbe essere dovuta a:
1. Differenza nel cell_area calculation
2. Differenza nelle transforms di validazione
3. Differenza nel modo di caricare i pesi

SOLUZIONE: Usa il metodo '{best_match['method']}' per il count
           nei successivi stage per consistenza.
""")


if __name__ == "__main__":
    main()