#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Stage 1 - ZIP Pre-training

METRICHE CALCOLATE:
1. Loss (BCE)
2. Accuracy: (TP + TN) / (TP + TN + FP + FN)
3. Precision: TP / (TP + FP)
4. Recall: TP / (TP + FN)
5. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
6. Coverage: % blocchi predetti come "pieni"

dove:
- TP: True Positives (blocchi pieni correttamente identificati)
- TN: True Negatives (blocchi vuoti correttamente identificati)
- FP: False Positives (blocchi vuoti predetti come pieni)
- FN: False Negatives (blocchi pieni predetti come vuoti)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import numpy as np
import json

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

# Default bin configurations for common datasets
DEFAULT_BINS_CONFIG = {
    'shha': {
        'bins': [[0, 0], [1, 3], [4, 6], [7, 10], [11, 15], [16, 22], [23, 32], [33, 9999]],
        'bin_centers': [0.0, 2.0, 5.0, 8.5, 13.0, 19.0, 27.5, 45.0],
    },
    'shhb': {
        'bins': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9999]],
        'bin_centers': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.16],
    },
    'ucf': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'nwpu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'jhu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
}

DATASET_ALIAS_MAP = {
    'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
    'shhb': 'shhb', 'shanghaitechb': 'shhb', 'shanghaitechpartb': 'shhb',
    'ucf': 'ucf', 'ucfqnrf': 'ucf',
    'nwpu': 'nwpu', 'jhu': 'jhu',
}



# =============================================================================
# LOSS (STESSA DEL TRAINING)
# =============================================================================

class PiHeadLoss(nn.Module):
    """BCE Loss per π-head."""
    def __init__(self, pos_weight=5.0, block_size=16):
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
        return loss


# =============================================================================
# VALIDATION CON METRICHE COMPLETE
# =============================================================================

@torch.no_grad()
def validate_with_metrics(model, criterion, dataloader, device, config):
    """
    Validazione Stage 1 con metriche dettagliate.
    
    Returns:
        dict con tutte le metriche
    """
    model.eval()
    
    total_loss = 0.0
    
    # Confusion Matrix
    total_tp = 0  # True Positives
    total_tn = 0  # True Negatives  
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    
    # Statistiche π
    pi_mean_list = []
    pi_std_list = []
    coverage_list = []
    
    # Per MAE indicativo (non affidabile in Stage 1)
    total_mae = 0.0
    total_pred_count = 0.0
    total_gt_count = 0.0
    n_samples = 0
    
    block_size = config.get('DATA', config.get('DATASET', {})).get('ZIP_BLOCK_SIZE', config.get('DATA', config.get('DATASET', {})).get('BLOCK_SIZE', 16))
    
    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage1"):
        images, gt_density = images.to(device), gt_density.to(device)
        
        # Forward
        preds = model(images)
        loss = criterion(preds, gt_density)
        total_loss += loss.item()
        
        # π predictions
        pi_logits = preds["logit_pi_maps"]  # [B, 2, Hb, Wb]
        pi_probs = torch.sigmoid(pi_logits[:, 1:2])  # Prob "pieno"
        
        # Ground truth occupancy
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=block_size,
            stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts_per_block > 0.5).float()
        
        # Allinea dimensioni
        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, 
                size=pi_probs.shape[-2:], 
                mode='nearest'
            )
        
        # Predizioni binarie (threshold 0.5)
        pred_occupancy = (pi_probs > 0.5).float()
        
        # Confusion matrix per batch
        tp = ((pred_occupancy == 1) & (gt_occupancy == 1)).sum().item()
        tn = ((pred_occupancy == 0) & (gt_occupancy == 0)).sum().item()
        fp = ((pred_occupancy == 1) & (gt_occupancy == 0)).sum().item()
        fn = ((pred_occupancy == 0) & (gt_occupancy == 1)).sum().item()
        
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        
        # Statistiche π
        pi_mean_list.append(pi_probs.mean().item())
        pi_std_list.append(pi_probs.std().item())
        coverage_list.append((pi_probs > 0.5).float().mean().item() * 100)
        
        # MAE indicativo (usando λ maps)
        lambda_maps = preds.get("lambda_maps")
        if lambda_maps is not None:
            pred_count_map = pi_probs * lambda_maps
            
            for idx, pts in enumerate(points):
                gt = len(pts) if pts is not None else 0
                pred = pred_count_map[idx].sum().item()
                
                total_mae += abs(pred - gt)
                total_pred_count += pred
                total_gt_count += gt
                n_samples += 1
    
    # =========================================================================
    # CALCOLA METRICHE AGGREGATE
    # =========================================================================
    
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    
    # Classification metrics
    total_samples = total_tp + total_tn + total_fp + total_fn
    
    accuracy = (total_tp + total_tn) / max(total_samples, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
    
    # π statistics
    avg_pi_mean = np.mean(pi_mean_list)
    avg_pi_std = np.mean(pi_std_list)
    avg_coverage = np.mean(coverage_list)
    
    # MAE (solo indicativo)
    mae = total_mae / n_samples if n_samples > 0 else 0
    bias = total_pred_count / total_gt_count if total_gt_count > 0 else 0
    
    # =========================================================================
    # REPORT DETTAGLIATO
    # =========================================================================
    
    print("\n" + "="*70)
    print("📊 STAGE 1 VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"   True Positives (TP):   {total_tp:,}")
    print(f"   True Negatives (TN):   {total_tn:,}")
    print(f"   False Positives (FP):  {total_fp:,}")
    print(f"   False Negatives (FN):  {total_fn:,}")
    print(f"   Total blocks:          {total_samples:,}")
    
    print(f"\n📈 Classification Metrics:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1_score*100:.2f}%")
    
    print(f"\n🎯 π-Head Statistics:")
    print(f"   Loss (BCE):      {avg_loss:.4f}")
    print(f"   π Mean:          {avg_pi_mean:.3f}")
    print(f"   π Std:           {avg_pi_std:.3f}")
    print(f"   Coverage:        {avg_coverage:.1f}%")
    
    print(f"\n📊 Count Metrics (Indicativo):")
    print(f"   MAE (π·λ):       {mae:.2f}")
    print(f"   Bias:            {bias:.3f}")
    print(f"   ⚠️  Non affidabile in Stage 1 - solo per monitoring")
    
    print("\n" + "="*70)
    
    # =========================================================================
    # INTERPRETAZIONE
    # =========================================================================
    
    print("\n💡 Interpretazione:")
    
    if recall < 0.85:
        print(f"   ⚠️  Recall basso ({recall*100:.1f}%)")
        print(f"       → Il modello perde troppi blocchi con persone (FN alto)")
        print(f"       → Soluzione: aumentare pos_weight in BCE")
    elif recall > 0.95:
        print(f"   ✅ Recall alto ({recall*100:.1f}%)")
        print(f"       → Buona copertura dei blocchi occupati")
    
    if precision < 0.70:
        print(f"   ⚠️  Precision bassa ({precision*100:.1f}%)")
        print(f"       → Troppi falsi positivi (FP alto)")
        print(f"       → Il modello predice persone dove non ci sono")
    elif precision > 0.85:
        print(f"   ✅ Precision alta ({precision*100:.1f}%)")
        print(f"       → Poche false detection")
    
    if f1_score > 0.80:
        print(f"   🎯 F1-Score alto ({f1_score*100:.1f}%)")
        print(f"       → Buon bilanciamento precision/recall")
    elif f1_score < 0.70:
        print(f"   ⚠️  F1-Score basso ({f1_score*100:.1f}%)")
        print(f"       → Sbilanciamento tra precision e recall")
    
    if avg_coverage < 10:
        print(f"   ⚠️  Coverage molto bassa ({avg_coverage:.1f}%)")
        print(f"       → Il modello è troppo conservativo")
    elif avg_coverage > 40:
        print(f"   ⚠️  Coverage molto alta ({avg_coverage:.1f}%)")
        print(f"       → Il modello è troppo aggressivo")
    else:
        print(f"   ✅ Coverage ragionevole ({avg_coverage:.1f}%)")
    
    print("\n" + "="*70)
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'tp': total_tp,
            'tn': total_tn,
            'fp': total_fp,
            'fn': total_fn,
        },
        'pi_stats': {
            'mean': avg_pi_mean,
            'std': avg_pi_std,
            'coverage': avg_coverage,
        },
        'count_metrics': {
            'mae': mae,
            'bias': bias,
        }
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
        
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print("="*60)
    print("🔍 EVALUATION STAGE 1 - ZIP Pre-training")
    print("="*60)
    print(f"Device: {device}")
    print("="*60)

    # Setup Modello - gestione robusta config
    dataset_section = config.get('DATASET', {})
    data_section = config.get('DATA')
    
    # Estrai nome dataset
    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = dict(dataset_section)  # Usa DATASET come DATA
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}
    
    # Normalizza nome dataset
    normalized = ''.join(c for c in str(dataset_name_raw).lower() if c.isalnum())
    dataset_name = DATASET_ALIAS_MAP.get(normalized, dataset_name_raw.lower() if isinstance(dataset_name_raw, str) else 'shha')
    
    # Ottieni bins config
    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"No BINS_CONFIG for dataset '{dataset_name}'")
    
    bins, bin_centers = bin_config['bins'], bin_config['bin_centers']
    
    # Defaults per data_cfg
    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]
    if 'ZIP_BLOCK_SIZE' not in data_cfg:
        data_cfg['ZIP_BLOCK_SIZE'] = data_cfg.get('BLOCK_SIZE', 16)
    if 'VAL_SPLIT' not in data_cfg:
        data_cfg['VAL_SPLIT'] = 'val'

    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model_cfg = config.get('MODEL', {})
    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Caricamento Checkpoint
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(
        exp_cfg.get('OUT_DIR', config.get('EXPERIMENT_DIR', 'exp')),
        config.get('RUN_NAME', 'stage1')
    )
    
    # Cerca checkpoint
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(output_dir, "last.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"❌ Nessun checkpoint trovato in {output_dir}")
        return

    print(f"\n✅ Caricamento checkpoint: {checkpoint_path}")
    raw_state = torch.load(checkpoint_path, map_location=device)
    state_dict = raw_state.get('model', raw_state)
    model.load_state_dict(state_dict, strict=False)

    # Setup Loss
    pos_weight = config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0)
    criterion = PiHeadLoss(
        pos_weight=pos_weight,
        block_size=data_cfg['ZIP_BLOCK_SIZE']
    ).to(device)

    # Dataset
    DatasetClass = get_dataset(dataset_name)
    val_tf = build_transforms(data_cfg, is_train=False)
    
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('OPTIM_ZIP', {}).get('NUM_WORKERS', 4),
        collate_fn=collate_fn
    )

    print(f"\nValidation samples: {len(val_dataset)}")

    # Validazione
    results = validate_with_metrics(model, criterion, val_loader, device, config)

    # Salva metriche
    metrics_path = os.path.join(output_dir, "stage1_metrics.json")
    
    # Converti per JSON
    results_json = {
        'loss': float(results['loss']),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'confusion_matrix': results['confusion_matrix'],
        'pi_stats': {k: float(v) for k, v in results['pi_stats'].items()},
        'count_metrics': {k: float(v) for k, v in results['count_metrics'].items()},
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Metriche salvate in: {metrics_path}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Precision: {results['precision']*100:.2f}%")
    print(f"   Recall:    {results['recall']*100:.2f}%")
    print(f"   F1-Score:  {results['f1_score']*100:.2f}%")
    
    if results['f1_score'] > 0.80 and results['recall'] > 0.90:
        print(f"\n   ✅ Stage 1 performance eccellente!")
    elif results['f1_score'] > 0.70:
        print(f"\n   ✅ Stage 1 performance buona")
    else:
        print(f"\n   ⚠️  Stage 1 necessita miglioramenti")


if __name__ == '__main__':
    main()