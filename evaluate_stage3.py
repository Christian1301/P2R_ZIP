#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Stage 3 - Con Soft Weighting

Valuta il modello con diverse modalità:
1. RAW: Density pura senza masking
2. MASKED: Hard masking con threshold τ
3. SOFT: Soft weighting con alpha α

Mostra metriche per fascia di densità.
"""

import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn, canonicalize_p2r_grid


def apply_soft_weighting(density, pi_probs, alpha=0.2):
    """Soft weighting: density × (1 - α + α × π)"""
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
        )
    weights = (1 - alpha) + alpha * pi_probs
    return density * weights


def apply_hard_mask(density, pi_probs, threshold=0.5):
    """Hard masking: density × (π > τ)"""
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
        )
    mask = (pi_probs > threshold).float()
    return density * mask


@torch.no_grad()
def calibrate_log_scale(model, loader, device, default_down, num_batches=15):
    """Calibra log_scale per correggere bias sistematico."""
    model.eval()
    
    pred_counts = []
    gt_counts = []
    
    for batch_idx, (images, _, points) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        _, _, H_in, W_in = images.shape
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        cell_area = down_tuple[0] * down_tuple[1]
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            if gt > 0:
                pred_count = (pred[i].sum() / cell_area).item()
                pred_counts.append(pred_count)
                gt_counts.append(gt)
    
    if len(gt_counts) == 0:
        print("⚠️ Nessun dato per calibrazione")
        return
    
    ratios = [p/g for p, g in zip(pred_counts, gt_counts)]
    bias_global = sum(pred_counts) / sum(gt_counts)
    bias_median = np.median(ratios)
    
    print(f"📊 Statistiche Calibrazione:")
    print(f"   Bias globale (sum): {bias_global:.3f}")
    print(f"   Bias mediano: {bias_median:.3f}")
    print(f"   Std ratio: {np.std(ratios):.3f}")
    print(f"   Range ratio: [{min(ratios):.3f}, {max(ratios):.3f}]")
    
    outlier_high = sum(1 for r in ratios if r > 1.5)
    outlier_low = sum(1 for r in ratios if r < 0.67)
    print(f"   Outlier (>1.5x): {outlier_high}/{len(ratios)}")
    print(f"   Outlier (<0.67x): {outlier_low}/{len(ratios)}")
    
    # Calibra se bias significativo
    if abs(bias_median - 1.0) > 0.05:
        adjust = np.log(bias_median)
        adjust = np.clip(adjust, -1.0, 1.0)
        
        old_scale = model.p2r_head.log_scale.item()
        model.p2r_head.log_scale.data -= torch.tensor(adjust, device=device)
        new_scale = model.p2r_head.log_scale.item()
        
        print(f"🔧 Calibrazione: bias={bias_median:.3f} → log_scale {old_scale:.4f}→{new_scale:.4f} (scala={np.exp(new_scale):.4f})")


@torch.no_grad()
def evaluate(model, dataloader, device, default_down, pi_threshold=0.5, soft_alpha=0.2):
    """
    Valuta con tre modalità: RAW, MASKED (hard), SOFT.
    """
    model.eval()
    
    # Risultati per modalità
    results = {
        'raw': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0},
        'masked': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0},
        'soft': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0},
    }
    
    # Per analisi per densità
    density_bins = {
        'sparse': (0, 100),
        'medium': (100, 500),
        'dense': (500, float('inf')),
    }
    density_results = {
        mode: {bin_name: {'mae': [], 'count': 0} for bin_name in density_bins}
        for mode in ['raw', 'masked', 'soft']
    }
    
    coverages = []
    pi_ratios = []
    
    for images, densities, points in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images)
        raw_density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']
        
        _, _, H_in, W_in = images.shape
        raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        # Resize π
        if pi_probs.shape[-2:] != raw_density.shape[-2:]:
            pi_probs_resized = F.interpolate(
                pi_probs, size=raw_density.shape[-2:], mode='bilinear', align_corners=False
            )
        else:
            pi_probs_resized = pi_probs
        
        # Coverage e π ratio
        coverage = (pi_probs_resized > pi_threshold).float().mean().item() * 100
        coverages.append(coverage)
        pi_ratios.append(pi_probs_resized.mean().item())
        
        # Density variants
        masked_density = apply_hard_mask(raw_density, pi_probs, pi_threshold)
        soft_density = apply_soft_weighting(raw_density, pi_probs, soft_alpha)
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            
            # Counts
            pred_raw = (raw_density[i].sum() / cell_area).item()
            pred_masked = (masked_density[i].sum() / cell_area).item()
            pred_soft = (soft_density[i].sum() / cell_area).item()
            
            # Update results
            for mode, pred in [('raw', pred_raw), ('masked', pred_masked), ('soft', pred_soft)]:
                results[mode]['mae'].append(abs(pred - gt))
                results[mode]['mse'].append((pred - gt) ** 2)
                results[mode]['pred'] += pred
                results[mode]['gt'] += gt
                
                # Per density bin
                for bin_name, (lo, hi) in density_bins.items():
                    if lo <= gt < hi:
                        density_results[mode][bin_name]['mae'].append(abs(pred - gt))
                        density_results[mode][bin_name]['count'] += 1
    
    # Calcola metriche finali
    final_results = {}
    
    for mode in ['raw', 'masked', 'soft']:
        r = results[mode]
        final_results[mode] = {
            'mae': np.mean(r['mae']),
            'rmse': np.sqrt(np.mean(r['mse'])),
            'bias': r['pred'] / r['gt'] if r['gt'] > 0 else 0,
        }
        
        # Per densità
        final_results[mode]['by_density'] = {}
        for bin_name in density_bins:
            dr = density_results[mode][bin_name]
            if dr['mae']:
                final_results[mode]['by_density'][bin_name] = {
                    'mae': np.mean(dr['mae']),
                    'count': dr['count'],
                }
    
    final_results['coverage'] = np.mean(coverages)
    final_results['pi_mean'] = np.mean(pi_ratios)
    
    return final_results


def print_results(results, pi_threshold, soft_alpha):
    """Stampa risultati in formato tabellare."""
    
    print("=" * 70)
    print("📊 RISULTATI VALUTAZIONE")
    print("=" * 70)
    
    # Tabella principale
    print(f"\n{'Metrica':<12} {'RAW':<15} {'MASKED (τ={:.2f})':<20} {'SOFT (α={:.2f})':<15}".format(
        pi_threshold, soft_alpha))
    print("-" * 62)
    
    for metric in ['mae', 'rmse', 'bias']:
        raw = results['raw'][metric]
        masked = results['masked'][metric]
        soft = results['soft'][metric]
        
        if metric == 'bias':
            print(f"{metric.upper():<12} {raw:<15.3f} {masked:<20.3f} {soft:<15.3f}")
        else:
            print(f"{metric.upper():<12} {raw:<15.2f} {masked:<20.2f} {soft:<15.2f}")
    
    print(f"\n📈 Coverage medio (τ={pi_threshold}): {results['coverage']:.1f}%")
    print(f"   π-head active ratio: {results['pi_mean']*100:.1f}%")
    
    # Per densità
    print("\n📊 Per densità:")
    density_labels = {
        'sparse': 'Sparse  (0-100)',
        'medium': 'Medium (100-500)',
        'dense': 'Dense  (500+)',
    }
    
    for bin_name in ['sparse', 'medium', 'dense']:
        raw_data = results['raw']['by_density'].get(bin_name, {})
        masked_data = results['masked']['by_density'].get(bin_name, {})
        soft_data = results['soft']['by_density'].get(bin_name, {})
        
        raw_mae = raw_data.get('mae', 0)
        masked_mae = masked_data.get('mae', 0)
        soft_mae = soft_data.get('mae', 0)
        count = raw_data.get('count', 0)
        
        delta_masked = raw_mae - masked_mae
        delta_soft = raw_mae - soft_mae
        
        print(f"   {density_labels[bin_name]}: RAW={raw_mae:.1f} → MASKED={masked_mae:.1f} (Δ{delta_masked:+.1f}) | SOFT={soft_mae:.1f} (Δ{delta_soft:+.1f}) [{count} imgs]")
    
    # Box finale
    print("\n" + "=" * 70)
    print("🏁 RISULTATO FINALE")
    print("=" * 70)
    
    best_mode = min(['raw', 'masked', 'soft'], key=lambda m: results[m]['mae'])
    best_mae = results[best_mode]['mae']
    
    print(f"   ┌{'─'*50}┐")
    print(f"   │  MAE RAW:        {results['raw']['mae']:<30.2f}│")
    print(f"   │  MAE MASKED:     {results['masked']['mae']:<30.2f}│")
    print(f"   │  MAE SOFT:       {results['soft']['mae']:<30.2f}│")
    print(f"   │{'─'*50}│")
    print(f"   │  MIGLIORE:       {best_mode.upper():<30}│")
    print(f"   │  MAE:            {best_mae:<30.2f}│")
    print(f"   └{'─'*50}┘")
    print("=" * 70)
    
    # Raccomandazione
    print("\n💡 RACCOMANDAZIONE:")
    if results['raw']['mae'] < results['masked']['mae'] and results['raw']['mae'] < results['soft']['mae']:
        print("   → Usa MAE RAW come metrica principale")
        print("   → Il π-head non sta aggiungendo valore, considera:")
        print("     1. Re-train Stage 1 con Focal Loss")
        print("     2. Aumentare pos_weight per migliorare recall")
    elif results['soft']['mae'] < results['masked']['mae']:
        print("   → Soft weighting funziona meglio di hard masking")
        print(f"   → Prova diversi valori di α (attuale: {soft_alpha})")
    else:
        print("   → Hard masking funziona bene")
        print(f"   → Il threshold τ={pi_threshold} è appropriato")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pi-thresh', type=float, default=None)
    parser.add_argument('--soft-alpha', type=float, default=0.2)
    parser.add_argument('--no-calibrate', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['DEVICE'])
    
    data_cfg = config['DATA']
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', 16)
    
    pi_threshold = args.pi_thresh if args.pi_thresh is not None else config['MODEL']['ZIP_PI_THRESH']
    soft_alpha = args.soft_alpha
    
    print(f"✅ Usando π-threshold = {pi_threshold} (da {'args' if args.pi_thresh else 'config'})")
    print(f"✅ Usando soft-alpha = {soft_alpha}")
    
    # Dataset
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config['DATASET'])
    val_ds = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=block_size,
        transforms=val_tf
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    # Model
    bin_config = config['BINS_CONFIG'][config['DATASET']]
    zip_head_cfg = config.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=pi_threshold,
        gate=config['MODEL']['GATE'],
        upsample_to_input=config['MODEL'].get('UPSAMPLE_TO_INPUT', False),
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        use_ste_mask=config['MODEL'].get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Load checkpoint
    run_name = config.get('RUN_NAME', 'shha_v11')
    output_dir = os.path.join(config['EXP']['OUT_DIR'], run_name)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Cerca automaticamente
        for name in ['stage3_best.pth', 'stage3_latest.pth', 'stage2_best.pth', 'best_model.pth']:
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                ckpt_path = path
                break
        else:
            raise FileNotFoundError(f"Nessun checkpoint trovato in {output_dir}")
    
    print(f"🔄 Caricamento pesi da: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    
    # Calibrazione
    if not args.no_calibrate:
        print("🔧 Calibrazione log_scale pre-eval...")
        calibrate_log_scale(model, val_loader, device, default_down)
    
    # Valutazione
    print(f"\n{'='*5} VALUTAZIONE (π-threshold={pi_threshold}, soft-α={soft_alpha}) {'='*5}")
    results = evaluate(model, val_loader, device, default_down, pi_threshold, soft_alpha)
    
    # Stampa risultati
    print_results(results, pi_threshold, soft_alpha)


if __name__ == '__main__':
    main()