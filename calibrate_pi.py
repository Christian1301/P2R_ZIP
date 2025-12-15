#!/usr/bin/env python3
"""
Calibrazione Soglia π Ottimale

Questo script analizza tutto il validation set per trovare:
1. La soglia τ ottimale per hard masking
2. Il valore α ottimale per soft weighting
3. Statistiche per densità (sparse/medium/dense)

Output:
- Tabella MAE per ogni soglia
- Soglia ottimale consigliata
- Grafici di analisi
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


def compute_counts_all_thresholds(
    model, 
    val_loader, 
    device, 
    default_down,
    thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
):
    """
    Calcola count per tutte le soglie e tutti gli alpha su tutto il val set.
    
    Returns:
        results: lista di dict con gt, pred_raw, pred per ogni threshold/alpha
    """
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for images, densities, points in tqdm(val_loader, desc="Analyzing"):
            images = images.to(device)
            points_list = [p.to(device) for p in points]
            
            # Forward
            outputs = model(images)
            pred_density = outputs['p2r_density']
            pi_probs = outputs.get('pi_probs')
            
            if pi_probs is None:
                logit_pi = outputs.get('logit_pi_maps')
                if logit_pi is not None:
                    pi_probs = torch.softmax(logit_pi, dim=1)[:, 1:2]
                else:
                    print("⚠️ π probs non trovate!")
                    continue
            
            _, _, H_in, W_in = images.shape
            pred_density, down_tuple, _ = canonicalize_p2r_grid(
                pred_density, (H_in, W_in), default_down
            )
            
            # Resize pi_probs se necessario
            if pi_probs.shape[-2:] != pred_density.shape[-2:]:
                pi_probs = F.interpolate(
                    pi_probs,
                    size=pred_density.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            down_h, down_w = down_tuple
            cell_area = down_h * down_w
            
            for i, pts in enumerate(points_list):
                gt = len(pts)
                
                # Count RAW (senza maschera)
                pred_raw = (pred_density[i].sum() / cell_area).item()
                
                # π statistics
                pi_mean = pi_probs[i].mean().item()
                pi_std = pi_probs[i].std().item()
                pi_min = pi_probs[i].min().item()
                pi_max = pi_probs[i].max().item()
                
                result = {
                    'gt': gt,
                    'pred_raw': pred_raw,
                    'pi_mean': pi_mean,
                    'pi_std': pi_std,
                    'pi_min': pi_min,
                    'pi_max': pi_max,
                    'density_class': 'sparse' if gt < 100 else ('medium' if gt < 500 else 'dense'),
                }
                
                # Hard masking per ogni soglia
                for tau in thresholds:
                    mask = (pi_probs[i] > tau).float()
                    masked_density = pred_density[i] * mask
                    pred_masked = (masked_density.sum() / cell_area).item()
                    result[f'pred_tau_{tau}'] = pred_masked
                    result[f'coverage_tau_{tau}'] = mask.mean().item()
                
                # Soft weighting per ogni alpha
                for alpha in alphas:
                    # Formula: pred * (1 - alpha + alpha * π)
                    weight = (1 - alpha) + alpha * pi_probs[i]
                    weighted_density = pred_density[i] * weight
                    pred_weighted = (weighted_density.sum() / cell_area).item()
                    result[f'pred_alpha_{alpha}'] = pred_weighted
                
                all_results.append(result)
    
    return all_results


def analyze_results(results, thresholds, alphas):
    """Analizza i risultati e trova parametri ottimali."""
    
    print("\n" + "=" * 70)
    print("📊 ANALISI CALIBRAZIONE π")
    print("=" * 70)
    
    n_samples = len(results)
    print(f"\nCampioni analizzati: {n_samples}")
    
    # Statistiche π
    pi_means = [r['pi_mean'] for r in results]
    print(f"\n📈 Statistiche π:")
    print(f"   Media globale: {np.mean(pi_means):.3f}")
    print(f"   Std: {np.std(pi_means):.3f}")
    print(f"   Range: [{np.min(pi_means):.3f}, {np.max(pi_means):.3f}]")
    
    # =========================================================================
    # HARD MASKING ANALYSIS
    # =========================================================================
    print("\n" + "-" * 70)
    print("🎯 HARD MASKING (pred × mask)")
    print("-" * 70)
    
    # MAE per ogni soglia
    print(f"\n{'Soglia τ':<12} {'MAE':<10} {'RMSE':<10} {'Bias':<10} {'Coverage':<10}")
    print("-" * 52)
    
    # RAW (no masking)
    mae_raw = np.mean([abs(r['pred_raw'] - r['gt']) for r in results])
    rmse_raw = np.sqrt(np.mean([(r['pred_raw'] - r['gt'])**2 for r in results]))
    bias_raw = np.mean([r['pred_raw'] / (r['gt'] + 1e-6) for r in results if r['gt'] > 0])
    print(f"{'RAW':<12} {mae_raw:<10.2f} {rmse_raw:<10.2f} {bias_raw:<10.3f} {'100%':<10}")
    
    best_tau = None
    best_mae_tau = float('inf')
    tau_results = {}
    
    for tau in thresholds:
        maes = [abs(r[f'pred_tau_{tau}'] - r['gt']) for r in results]
        mae = np.mean(maes)
        rmse = np.sqrt(np.mean([(r[f'pred_tau_{tau}'] - r['gt'])**2 for r in results]))
        
        # Bias (evita divisione per zero)
        biases = [r[f'pred_tau_{tau}'] / (r['gt'] + 1e-6) for r in results if r['gt'] > 0]
        bias = np.mean(biases) if biases else 0
        
        coverage = np.mean([r[f'coverage_tau_{tau}'] for r in results]) * 100
        
        tau_results[tau] = {'mae': mae, 'rmse': rmse, 'bias': bias, 'coverage': coverage}
        
        marker = " ← BEST" if mae < best_mae_tau else ""
        if mae < best_mae_tau:
            best_mae_tau = mae
            best_tau = tau
        
        print(f"{tau:<12.1f} {mae:<10.2f} {rmse:<10.2f} {bias:<10.3f} {coverage:<10.1f}%{marker}")
    
    improvement_tau = mae_raw - best_mae_tau
    print(f"\n✅ Soglia ottimale τ = {best_tau}")
    print(f"   MAE: {mae_raw:.2f} → {best_mae_tau:.2f} (miglioramento: {improvement_tau:+.2f})")
    
    # =========================================================================
    # SOFT WEIGHTING ANALYSIS
    # =========================================================================
    print("\n" + "-" * 70)
    print("🎯 SOFT WEIGHTING (pred × (1-α + α×π))")
    print("-" * 70)
    
    print(f"\n{'Alpha α':<12} {'MAE':<10} {'RMSE':<10} {'Bias':<10}")
    print("-" * 42)
    
    best_alpha = None
    best_mae_alpha = float('inf')
    alpha_results = {}
    
    for alpha in alphas:
        maes = [abs(r[f'pred_alpha_{alpha}'] - r['gt']) for r in results]
        mae = np.mean(maes)
        rmse = np.sqrt(np.mean([(r[f'pred_alpha_{alpha}'] - r['gt'])**2 for r in results]))
        
        biases = [r[f'pred_alpha_{alpha}'] / (r['gt'] + 1e-6) for r in results if r['gt'] > 0]
        bias = np.mean(biases) if biases else 0
        
        alpha_results[alpha] = {'mae': mae, 'rmse': rmse, 'bias': bias}
        
        marker = " ← BEST" if mae < best_mae_alpha else ""
        if mae < best_mae_alpha:
            best_mae_alpha = mae
            best_alpha = alpha
        
        print(f"{alpha:<12.1f} {mae:<10.2f} {rmse:<10.2f} {bias:<10.3f}{marker}")
    
    improvement_alpha = mae_raw - best_mae_alpha
    print(f"\n✅ Alpha ottimale α = {best_alpha}")
    print(f"   MAE: {mae_raw:.2f} → {best_mae_alpha:.2f} (miglioramento: {improvement_alpha:+.2f})")
    
    # =========================================================================
    # ANALYSIS PER DENSITY CLASS
    # =========================================================================
    print("\n" + "-" * 70)
    print("📊 ANALISI PER DENSITÀ")
    print("-" * 70)
    
    for density_class in ['sparse', 'medium', 'dense']:
        class_results = [r for r in results if r['density_class'] == density_class]
        if not class_results:
            continue
        
        n = len(class_results)
        mae_raw_class = np.mean([abs(r['pred_raw'] - r['gt']) for r in class_results])
        mae_best_tau = np.mean([abs(r[f'pred_tau_{best_tau}'] - r['gt']) for r in class_results])
        mae_best_alpha = np.mean([abs(r[f'pred_alpha_{best_alpha}'] - r['gt']) for r in class_results])
        
        print(f"\n{density_class.upper()} ({n} imgs):")
        print(f"   RAW:           MAE = {mae_raw_class:.2f}")
        print(f"   τ = {best_tau}:       MAE = {mae_best_tau:.2f} ({mae_best_tau - mae_raw_class:+.2f})")
        print(f"   α = {best_alpha}:       MAE = {mae_best_alpha:.2f} ({mae_best_alpha - mae_raw_class:+.2f})")
    
    # =========================================================================
    # RACCOMANDAZIONI
    # =========================================================================
    print("\n" + "=" * 70)
    print("💡 RACCOMANDAZIONI")
    print("=" * 70)
    
    print(f"""
Per il tuo config.yaml, aggiorna:

1. HARD MASKING (se vuoi usare maschera binaria):
   MODEL:
     ZIP_PI_THRESH: {best_tau}

2. SOFT WEIGHTING (raccomandato - più smooth):
   STAGE3_LOSS:
     SOFT_WEIGHT_ALPHA: {best_alpha}
   
   O in JOINT_LOSS:
     SOFT_WEIGHT_ALPHA: {best_alpha}

Confronto metodi:
- RAW (no π):        MAE = {mae_raw:.2f}
- Hard τ={best_tau}:       MAE = {best_mae_tau:.2f} ({improvement_tau:+.2f})
- Soft α={best_alpha}:       MAE = {best_mae_alpha:.2f} ({improvement_alpha:+.2f})

{'🎯 SOFT WEIGHTING è migliore!' if best_mae_alpha < best_mae_tau else '🎯 HARD MASKING è migliore!'}
""")
    
    return {
        'best_tau': best_tau,
        'best_alpha': best_alpha,
        'mae_raw': mae_raw,
        'mae_best_tau': best_mae_tau,
        'mae_best_alpha': best_mae_alpha,
        'tau_results': tau_results,
        'alpha_results': alpha_results,
    }


def plot_analysis(results, analysis, output_path='visualizations/calibration_analysis.png'):
    """Genera grafici di analisi."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE vs Threshold
    ax1 = axes[0, 0]
    taus = sorted(analysis['tau_results'].keys())
    maes = [analysis['tau_results'][t]['mae'] for t in taus]
    ax1.plot(taus, maes, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=analysis['mae_raw'], color='r', linestyle='--', label=f'RAW: {analysis["mae_raw"]:.2f}')
    ax1.axvline(x=analysis['best_tau'], color='g', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Soglia τ', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('Hard Masking: MAE vs Soglia τ', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE vs Alpha
    ax2 = axes[0, 1]
    alphas = sorted(analysis['alpha_results'].keys())
    maes_alpha = [analysis['alpha_results'][a]['mae'] for a in alphas]
    ax2.plot(alphas, maes_alpha, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=analysis['mae_raw'], color='r', linestyle='--', label=f'RAW: {analysis["mae_raw"]:.2f}')
    ax2.axvline(x=analysis['best_alpha'], color='b', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Alpha α', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Soft Weighting: MAE vs Alpha α', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias vs Threshold
    ax3 = axes[1, 0]
    biases = [analysis['tau_results'][t]['bias'] for t in taus]
    ax3.plot(taus, biases, 'b-o', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='g', linestyle='--', label='Bias ideale = 1.0')
    ax3.set_xlabel('Soglia τ', fontsize=12)
    ax3.set_ylabel('Bias (pred/gt)', fontsize=12)
    ax3.set_title('Hard Masking: Bias vs Soglia τ', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Coverage vs Threshold
    ax4 = axes[1, 1]
    coverages = [analysis['tau_results'][t]['coverage'] for t in taus]
    ax4.plot(taus, coverages, 'purple', linewidth=2, marker='o', markersize=8)
    ax4.set_xlabel('Soglia τ', fontsize=12)
    ax4.set_ylabel('Coverage (%)', fontsize=12)
    ax4.set_title('Coverage della Maschera vs Soglia τ', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Grafici salvati in: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint (default: auto-detect)')
    parser.add_argument('--output', type=str, default='visualizations/calibration_analysis.png')
    args = parser.parse_args()
    
    # Carica config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    print("=" * 60)
    print("🔍 CALIBRAZIONE SOGLIA π OTTIMALE")
    print("=" * 60)
    
    # Dataset
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_ds)}")
    
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
            "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
            "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
            "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
            "lambda_noise_std": 0.0,
        },
    ).to(device)
    
    # Carica checkpoint
    run_name = cfg.get('RUN_NAME', 'p2r_zip')
    output_dir = os.path.join(cfg["EXP"]["OUT_DIR"], run_name)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-detect: prova stage3 → stage2
        for ckpt_name in ['stage3_best.pth', 'stage2_best.pth', 'best_model.pth']:
            ckpt_path = os.path.join(output_dir, ckpt_name)
            if os.path.isfile(ckpt_path):
                break
    
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if 'model' in state:
            state = state['model']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=False)
        print(f"✅ Caricato: {ckpt_path}")
    else:
        print(f"❌ Checkpoint non trovato: {ckpt_path}")
        return
    
    # Analisi
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = compute_counts_all_thresholds(
        model, val_loader, device, default_down,
        thresholds=thresholds,
        alphas=alphas
    )
    
    # Analisi e report
    analysis = analyze_results(results, thresholds, alphas)
    
    # Plot
    plot_analysis(results, analysis, args.output)
    
    print("\n✅ Calibrazione completata!")


if __name__ == "__main__":
    main()