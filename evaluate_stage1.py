# evaluate_stage1.py
# -*- coding: utf-8 -*-
"""
Evaluate Stage 1 V-Final
Calcola metriche complete (Accuracy, Recall, Precision) per valutare 
la qualità della maschera ZIP (Background vs Foreground).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import numpy as np
import argparse

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

def compute_zip_metrics(pred_logits, gt_density, block_size, occupancy_threshold=0.3):
    """
    Calcola metriche di classificazione per la maschera ZIP.
    """
    # 1. Ground Truth
    # Downsample densità per ottenere conteggio nel blocco
    gt_counts = F.avg_pool2d(gt_density, block_size, stride=block_size) * (block_size ** 2)
    
    # Allinea dimensioni se necessario (gestione padding/rounding del modello)
    if gt_counts.shape[-2:] != pred_logits.shape[-2:]:
        gt_counts = F.interpolate(gt_counts, size=pred_logits.shape[-2:], mode='nearest')
        
    # GT Occupancy: 1 se c'è gente, 0 altrimenti
    gt_occupancy = (gt_counts > occupancy_threshold).float()
    
    # 2. Predizioni
    pi_prob = torch.sigmoid(pred_logits)
    pred_occupancy = (pi_prob > 0.5).float()
    
    # 3. Metriche (Accumulatori)
    # Calcoliamo su tutto il batch appiattito
    tp = (pred_occupancy * gt_occupancy).sum().item()
    fp = (pred_occupancy * (1 - gt_occupancy)).sum().item()
    fn = ((1 - pred_occupancy) * gt_occupancy).sum().item()
    tn = ((1 - pred_occupancy) * (1 - gt_occupancy)).sum().item()
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_pos": gt_occupancy.sum().item(),
        "total_pixels": gt_occupancy.numel()
    }

@torch.no_grad()
def evaluate_model(model, dataloader, device, config):
    model.eval()
    
    # Accumulatori globali
    stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt_pos": 0, "total_pixels": 0}
    
    block_size = config['DATA']['ZIP_BLOCK_SIZE']
    thresh = config.get("ZIP_LOSS", {}).get("OCCUPANCY_THRESHOLD", 0.3)
    
    print(f"⚙️  Parametri Valutazione: Block={block_size}, Threshold={thresh}")
    
    for images, gt_density, _ in tqdm(dataloader, desc="Evaluating Stage 1"):
        images, gt_density = images.to(device), gt_density.to(device)
        
        outputs = model(images)
        logit_pi = outputs["logit_pi_maps"][:, 1:2] # Canale 1 = Presenza
        
        batch_stats = compute_zip_metrics(logit_pi, gt_density, block_size, thresh)
        
        for k in stats:
            stats[k] += batch_stats[k]
            
    # Calcolo Metriche Finali
    eps = 1e-6
    accuracy = (stats['tp'] + stats['tn']) / (stats['total_pixels'] + eps) * 100
    precision = stats['tp'] / (stats['tp'] + stats['fp'] + eps) * 100
    recall = stats['tp'] / (stats['tp'] + stats['fn'] + eps) * 100
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    
    # Percentuale di area predetta come positiva
    pred_pos_ratio = (stats['tp'] + stats['fp']) / (stats['total_pixels'] + eps) * 100
    gt_pos_ratio = stats['gt_pos'] / (stats['total_pixels'] + eps) * 100

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Predicted Area (%)": pred_pos_ratio,
        "GT Area (%)": gt_pos_ratio
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path specifico al checkpoint (opzionale)")
    args = parser.parse_args()

    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato.")
        return
        
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    # 1. Setup Dataset
    dataset_name = config['DATASET']
    DatasetClass = get_dataset(dataset_name)
    val_tf = build_transforms(config['DATA'], is_train=False)
    
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 2. Setup Modello
    bin_config = config['BINS_CONFIG'][dataset_name]
    model = P2R_ZIP_Model(
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        bins=bin_config['bins'], 
        bin_centers=bin_config['bin_centers'],
        upsample_to_input=config['MODEL']['UPSAMPLE_TO_INPUT'],
    ).to(device)

    # 3. Identifica Checkpoints da testare
    out_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    
    checkpoints = []
    if args.ckpt:
        checkpoints.append(("Custom", args.ckpt))
    else:
        # Cerca i due candidati principali
        p1 = os.path.join(out_dir, "best_model.pth")     # Best Loss
        p2 = os.path.join(out_dir, "stage1_best_acc.pth") # Best Accuracy (se esiste)
        
        if os.path.exists(p1): checkpoints.append(("Best Loss (Standard)", p1))
        if os.path.exists(p2): checkpoints.append(("Best Accuracy", p2))
        
        if not checkpoints:
            # Fallback
            p3 = os.path.join(out_dir, "last.pth")
            if os.path.exists(p3): checkpoints.append(("Last Epoch", p3))

    if not checkpoints:
        print(f"❌ Nessun checkpoint trovato in {out_dir}")
        return

    print(f"📊 Avvio Valutazione Stage 1 su {len(checkpoints)} checkpoint(s)\n")

    for name, path in checkpoints:
        print(f"🔹 Valutando: {name}")
        print(f"   Path: {path}")
        
        try:
            ckpt = torch.load(path, map_location=device)
            state = ckpt['model'] if 'model' in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            
            results = evaluate_model(model, val_loader, device, config)
            
            print(f"   ✅ Risultati:")
            print(f"      Accuracy:  {results['Accuracy']:.2f}%")
            print(f"      Precision: {results['Precision']:.2f}%")
            print(f"      Recall:    {results['Recall']:.2f}%")
            print(f"      F1-Score:  {results['F1-Score']:.2f}%")
            print(f"      GT Area:   {results['GT Area (%)']:.2f}% (Dataset reale)")
            print(f"      Pred Area: {results['Predicted Area (%)']:.2f}% (Maschera predetta)")
            print("-" * 60)
            
        except Exception as e:
            print(f"   ⚠️ Errore caricamento {path}: {e}")
            print("-" * 60)

if __name__ == '__main__':
    main()