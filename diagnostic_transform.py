#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNOSTIC 2 - Verifica Transforms e Confronto con Stage 2

Il diagnostic 1 ha mostrato che il metodo di count è corretto.
La discrepanza (MAE 87 vs 103) deve essere in:
1. Transforms di validazione diverse
2. Checkpoint salvato in momento diverso
3. Modo diverso di processare le immagini

Questo script testa diverse configurazioni di transforms.
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
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn

# Import transforms
import torchvision.transforms as T
from PIL import Image


def create_minimal_transforms(norm_mean, norm_std):
    """Transforms minimali: solo resize e normalize."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])


def create_stage2_style_transforms(norm_mean, norm_std, crop_size=None):
    """Transforms stile Stage 2 (potrebbe non avere crop)."""
    transforms_list = [
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ]
    return T.Compose(transforms_list)


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset semplificato per test transforms."""
    def __init__(self, root, split, transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        
        # Trova immagini e GT
        img_dir = os.path.join(root, split, "images")
        gt_dir = os.path.join(root, split, "ground-truth")
        
        self.samples = []
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(('.jpg', '.png')):
                    img_path = os.path.join(img_dir, fname)
                    # GT file
                    gt_name = "GT_" + fname.replace('.jpg', '.mat').replace('.png', '.mat')
                    gt_path = os.path.join(gt_dir, gt_name)
                    if os.path.isfile(gt_path):
                        self.samples.append((img_path, gt_path))
        
        print(f"   Found {len(self.samples)} samples in {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (W, H)
        
        # Load GT
        import scipy.io as sio
        mat = sio.loadmat(gt_path)
        
        # Try different keys for points
        points = None
        for key in ['image_info', 'annPoints', 'points']:
            if key in mat:
                if key == 'image_info':
                    points = mat[key][0][0][0][0][0]
                else:
                    points = mat[key]
                break
        
        gt_count = len(points) if points is not None else 0
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, gt_count, original_size


def compute_count(pred_density, default_down=8):
    """Calcola count con metodo standard."""
    cell_area = default_down ** 2
    return torch.sum(pred_density, dim=(1, 2, 3)) / cell_area


@torch.no_grad()
def validate_with_transforms(model, dataset, device, default_down, desc=""):
    """Validazione con dataset specifico."""
    model.eval()
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    total_mae = 0.0
    total_pred = 0.0
    total_gt = 0.0
    n = 0
    
    for img, gt_count, orig_size in loader:
        img = img.to(device)
        
        outputs = model(img)
        pred_density = outputs["p2r_density"]
        
        pred_count = compute_count(pred_density, default_down).item()
        
        total_mae += abs(pred_count - gt_count.item())
        total_pred += pred_count
        total_gt += gt_count.item()
        n += 1
    
    mae = total_mae / n
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    return {"mae": mae, "bias": bias, "n": n}


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    
    print("="*70)
    print("🔍 DIAGNOSTIC 2 - Verifica Transforms")
    print("="*70)
    
    data_cfg = config["DATA"]
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    norm_mean = data_cfg["NORM_MEAN"]
    norm_std = data_cfg["NORM_STD"]
    
    print(f"\n📐 Config:")
    print(f"   NORM_MEAN: {norm_mean}")
    print(f"   NORM_STD: {norm_std}")
    print(f"   CROP_SIZE: {data_cfg.get('CROP_SIZE', 'None')}")
    print(f"   P2R_DOWNSAMPLE: {default_down}")
    
    # Crea modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": config["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": config["ZIP_HEAD"].get("USE_SOFTPLUS", True),
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,
        use_ste_mask=True,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # Carica checkpoint
    stage2_path = os.path.join(output_dir, "stage2_best.pth")
    print(f"\n✅ Caricamento: {stage2_path}")
    state = torch.load(stage2_path, map_location=device)
    if "model" in state:
        state = state["model"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    
    # =========================================
    # TEST DIVERSE TRANSFORMS
    # =========================================
    print("\n" + "="*70)
    print("TEST: Diverse Configurazioni Transforms")
    print("="*70)
    
    results = []
    
    # Test 1: Transforms minimali (solo normalize)
    print("\n🧪 Test 1: Transforms minimali...")
    transforms_minimal = create_minimal_transforms(norm_mean, norm_std)
    dataset_minimal = SimpleDataset(
        data_cfg["ROOT"], 
        data_cfg["VAL_SPLIT"],
        transforms=transforms_minimal
    )
    if len(dataset_minimal) > 0:
        r = validate_with_transforms(model, dataset_minimal, device, default_down)
        results.append(("Minimali (no crop)", r))
        print(f"   MAE={r['mae']:.2f}, Bias={r['bias']:.3f}")
    
    # Test 2: Con il dataset standard del progetto
    print("\n🧪 Test 2: Dataset standard (build_transforms)...")
    from datasets.transforms import build_transforms
    DatasetClass = get_dataset(config["DATASET"])
    
    val_transforms = build_transforms(data_cfg, is_train=False)
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms
    )
    
    # Usa la validazione standard
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=0, collate_fn=collate_fn)
    
    total_mae = 0.0
    total_pred = 0.0
    total_gt = 0.0
    n = 0
    
    for images, gt_density, points in val_loader:
        images = images.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]
        
        _, _, h_in, w_in = images.shape
        pred_density_canon, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density_canon, dim=(1, 2, 3)) / cell_area
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred = pred_count[idx].item()
            total_mae += abs(pred - gt)
            total_pred += pred
            total_gt += gt
            n += 1
    
    mae = total_mae / n
    bias = total_pred / total_gt
    results.append(("Standard (build_transforms)", {"mae": mae, "bias": bias, "n": n}))
    print(f"   MAE={mae:.2f}, Bias={bias:.3f}")
    
    # =========================================
    # CHECK: Cosa fa build_transforms per is_train=False?
    # =========================================
    print("\n" + "="*70)
    print("CHECK: Contenuto build_transforms(is_train=False)")
    print("="*70)
    
    print(f"\n   val_transforms = {val_transforms}")
    
    # Prova a ispezionare
    if hasattr(val_transforms, 'transforms'):
        print("\n   Transforms list:")
        for i, t in enumerate(val_transforms.transforms):
            print(f"      {i}: {t}")
    
    # =========================================
    # VERIFICA: Il checkpoint contiene epoca/metrics?
    # =========================================
    print("\n" + "="*70)
    print("CHECK: Contenuto Checkpoint")
    print("="*70)
    
    full_state = torch.load(stage2_path, map_location=device)
    print(f"\n   Keys nel checkpoint: {list(full_state.keys())}")
    
    if "epoch" in full_state:
        print(f"   Epoch salvata: {full_state['epoch']}")
    if "best_mae" in full_state:
        print(f"   Best MAE salvata: {full_state['best_mae']}")
    if "best_val_mae" in full_state:
        print(f"   Best Val MAE: {full_state['best_val_mae']}")
    if "metrics" in full_state:
        print(f"   Metrics: {full_state['metrics']}")
    
    # =========================================
    # VERIFICA: log_scale value
    # =========================================
    print("\n" + "="*70)
    print("CHECK: Parametro log_scale")
    print("="*70)
    
    for name, param in model.p2r_head.named_parameters():
        if "log_scale" in name:
            print(f"   {name}: {param.item():.4f}")
            print(f"   exp(log_scale) = {torch.exp(param).item():.4f}")
    
    # =========================================
    # RIEPILOGO
    # =========================================
    print("\n" + "="*70)
    print("📊 RIEPILOGO")
    print("="*70)
    
    print("\n┌─────────────────────────────────────┬─────────┬─────────┐")
    print("│ Configurazione                      │   MAE   │  Bias   │")
    print("├─────────────────────────────────────┼─────────┼─────────┤")
    for name, r in results:
        print(f"│ {name:<35} │ {r['mae']:>7.2f} │ {r['bias']:>7.3f} │")
    print("├─────────────────────────────────────┼─────────┼─────────┤")
    print(f"│ {'Target Stage 2':<35} │ {87.77:>7.2f} │ {0.956:>7.3f} │")
    print("└─────────────────────────────────────┴─────────┴─────────┘")
    
    print("\n💡 CONCLUSIONI:")
    print("   Se 'Minimali' e 'Standard' danno lo stesso risultato (~103),")
    print("   allora il problema è nel CHECKPOINT, non nelle transforms.")
    print("")
    print("   Possibili cause:")
    print("   1. Stage 2 training salva il checkpoint PRIMA della best validation")
    print("   2. Stage 2 usa un diverso calcolo interno per MAE")
    print("   3. Il checkpoint 'best' non è realmente il migliore")


if __name__ == "__main__":
    main()