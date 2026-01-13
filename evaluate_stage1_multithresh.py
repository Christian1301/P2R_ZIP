# evaluate_stage1_multithresh.py
# -*- coding: utf-8 -*-
"""
Evaluate Stage 1 con MULTI-THRESHOLD
Testa diversi valori di threshold per trovare il miglior trade-off Recall/Precision.
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


def compute_zip_metrics_multi_thresh(pred_logits, gt_density, block_size, occupancy_threshold, pred_thresholds):
    """
    Calcola metriche per MULTIPLI threshold di predizione.
    """
    # 1. Ground Truth
    gt_counts = F.avg_pool2d(gt_density, block_size, stride=block_size) * (block_size ** 2)
    
    if gt_counts.shape[-2:] != pred_logits.shape[-2:]:
        gt_counts = F.interpolate(gt_counts, size=pred_logits.shape[-2:], mode='nearest')
        
    gt_occupancy = (gt_counts > occupancy_threshold).float()
    
    # 2. Predizioni (probabilità)
    pi_prob = torch.sigmoid(pred_logits)
    
    # 3. Calcola stats per ogni threshold
    results = {}
    for thresh in pred_thresholds:
        pred_occupancy = (pi_prob > thresh).float()
        
        tp = (pred_occupancy * gt_occupancy).sum().item()
        fp = (pred_occupancy * (1 - gt_occupancy)).sum().item()
        fn = ((1 - pred_occupancy) * gt_occupancy).sum().item()
        tn = ((1 - pred_occupancy) * (1 - gt_occupancy)).sum().item()
        
        results[thresh] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "gt_pos": gt_occupancy.sum().item(),
            "total_pixels": gt_occupancy.numel()
        }
    
    return results


@torch.no_grad()
def evaluate_model_multi_thresh(model, dataloader, device, config, pred_thresholds):
    model.eval()
    
    # Accumulatori per ogni threshold
    all_stats = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt_pos": 0, "total_pixels": 0} 
                 for t in pred_thresholds}
    
    block_size = config['DATA']['ZIP_BLOCK_SIZE']
    occ_thresh = config.get("ZIP_LOSS", {}).get("OCCUPANCY_THRESHOLD", 0.3)
    
    print(f"⚙️  Block={block_size}, GT Occupancy Threshold={occ_thresh}")
    print(f"⚙️  Testing prediction thresholds: {pred_thresholds}\n")
    
    for images, gt_density, _ in tqdm(dataloader, desc="Evaluating"):
        images, gt_density = images.to(device), gt_density.to(device)
        
        outputs = model(images)
        logit_pi = outputs["logit_pi_maps"][:, 1:2]
        
        batch_results = compute_zip_metrics_multi_thresh(
            logit_pi, gt_density, block_size, occ_thresh, pred_thresholds
        )
        
        for thresh in pred_thresholds:
            for k in all_stats[thresh]:
                all_stats[thresh][k] += batch_results[thresh][k]
    
    # Calcolo metriche finali per ogni threshold
    final_results = {}
    eps = 1e-6
    
    for thresh in pred_thresholds:
        s = all_stats[thresh]
        accuracy = (s['tp'] + s['tn']) / (s['total_pixels'] + eps) * 100
        precision = s['tp'] / (s['tp'] + s['fp'] + eps) * 100
        recall = s['tp'] / (s['tp'] + s['fn'] + eps) * 100
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        pred_area = (s['tp'] + s['fp']) / (s['total_pixels'] + eps) * 100
        gt_area = s['gt_pos'] / (s['total_pixels'] + eps) * 100
        
        final_results[thresh] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Pred Area": pred_area,
            "GT Area": gt_area
        }
    
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path specifico al checkpoint")
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
                        help="Threshold da testare (comma-separated)")
    args = parser.parse_args()

    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato.")
        return
        
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])
    
    # Parse thresholds
    pred_thresholds = [float(t) for t in args.thresholds.split(",")]

    # 1. Dataset
    dataset_name = config['DATASET']
    DatasetClass = get_dataset(dataset_name)
    val_tf = build_transforms(config['DATA'], is_train=False)
    
    val_dataset = DatasetClass(
        root=config['DATA']['ROOT'],
        split=config['DATA']['VAL_SPLIT'],
        block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )
    print(f"Dataset test: Trovate {len(val_dataset)} immagini.")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=collate_fn)

    # 2. Modello
    bin_config = config['BINS_CONFIG'][dataset_name]
    model = P2R_ZIP_Model(
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        bins=bin_config['bins'], 
        bin_centers=bin_config['bin_centers'],
        upsample_to_input=config['MODEL']['UPSAMPLE_TO_INPUT'],
    ).to(device)

    # 3. Checkpoint
    out_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt_name = "Custom"
    else:
        # Default: Best Loss
        ckpt_path = os.path.join(out_dir, "best_model.pth")
        ckpt_name = "Best Loss"
        
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, "stage1_best_acc.pth")
            ckpt_name = "Best Acc"

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint non trovato: {ckpt_path}")
        return

    print(f"\n📊 Valutazione Multi-Threshold")
    print(f"   Checkpoint: {ckpt_name} ({ckpt_path})\n")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    
    # 4. Valutazione
    results = evaluate_model_multi_thresh(model, val_loader, device, config, pred_thresholds)
    
    # 5. Stampa risultati
    print("\n" + "=" * 85)
    print(f"{'Thresh':>7} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>7} | {'Pred Area':>9} | {'Note'}")
    print("=" * 85)
    
    gt_area = list(results.values())[0]["GT Area"]
    best_f1_thresh = max(results.keys(), key=lambda t: results[t]["F1"])
    best_recall_thresh = max(results.keys(), key=lambda t: results[t]["Recall"])
    
    for thresh in sorted(results.keys()):
        r = results[thresh]
        note = ""
        if thresh == best_f1_thresh:
            note = "← Best F1"
        elif thresh == best_recall_thresh and thresh != best_f1_thresh:
            note = "← Best Recall"
        
        # Evidenzia recall > 90%
        recall_str = f"{r['Recall']:>7.2f}%"
        if r['Recall'] >= 90:
            recall_str = f"✓{r['Recall']:>6.2f}%"
        
        print(f"{thresh:>7.2f} | {r['Accuracy']:>7.2f}% | {r['Precision']:>8.2f}% | {recall_str} | {r['F1']:>6.2f}% | {r['Pred Area']:>8.2f}% | {note}")
    
    print("=" * 85)
    print(f"GT Area: {gt_area:.2f}%")
    print("\n💡 Suggerimento: Per Stage 2, scegli il threshold con Recall ≥ 90% e F1 ragionevole.")
    print(f"   Threshold consigliato per config: ZIP_PI_THRESH = {best_f1_thresh}")


if __name__ == '__main__':
    main()