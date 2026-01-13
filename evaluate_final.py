#!/usr/bin/env python3
"""
Valutazione Finale con Soglia π Calibrata

Valuta il modello sul validation/test set usando la soglia τ ottimale
trovata dalla calibrazione.

Output:
- MAE, RMSE, Bias dettagliati
- Analisi per densità
- Confronto RAW vs MASKED
- Salvataggio risultati
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from datetime import datetime

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


@torch.no_grad()
def evaluate_with_threshold(
    model, 
    data_loader, 
    device, 
    default_down,
    tau=0.2,
    verbose=True
):
    """
    Valutazione completa con soglia π specificata.
    
    Args:
        model: modello P2R-ZIP
        data_loader: DataLoader
        device: device
        default_down: downsample factor
        tau: soglia π per masking
        verbose: stampa dettagli per immagine
    
    Returns:
        dict con metriche dettagliate
    """
    model.eval()
    
    results = {
        'per_image': [],
        'by_density': {'sparse': [], 'medium': [], 'dense': []},
    }
    
    all_mae_raw = []
    all_mae_masked = []
    all_mse_raw = []
    all_mse_masked = []
    all_bias_raw = []
    all_bias_masked = []
    
    for batch_idx, (images, densities, points) in enumerate(tqdm(data_loader, desc=f"Evaluating (τ={tau})")):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        # Forward
        outputs = model(images)
        pred_density = outputs['p2r_density']
        
        # Get π probs
        pi_probs = outputs.get('pi_probs')
        if pi_probs is None:
            logit_pi = outputs.get('logit_pi_maps')
            if logit_pi is not None:
                pi_probs = torch.softmax(logit_pi, dim=1)[:, 1:2]
            else:
                print("⚠️ π probs non trovate, uso maschera = 1")
                pi_probs = torch.ones_like(pred_density)
        
        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        
        # Resize pi_probs se necessario
        if pi_probs.shape[-2:] != pred_density.shape[-2:]:
            pi_probs = F.interpolate(
                pi_probs,
                size=pred_density.shape[-2:],
                mode='nearest',  # nearest preserva valori binari, bilinear li ammorbidisce
            )
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            
            # Count RAW
            pred_raw = (pred_density[i].sum() / cell_area).item()
            
            # Count MASKED con soglia τ
            mask = (pi_probs[i] > tau).float()
            masked_density = pred_density[i] * mask
            pred_masked = (masked_density.sum() / cell_area).item()
            
            # Coverage
            coverage = mask.mean().item() * 100
            
            # π statistics
            pi_mean = pi_probs[i].mean().item()
            
            # Errori
            err_raw = abs(pred_raw - gt)
            err_masked = abs(pred_masked - gt)
            
            # Bias
            bias_raw = pred_raw / (gt + 1e-6) if gt > 0 else 0
            bias_masked = pred_masked / (gt + 1e-6) if gt > 0 else 0
            
            # Density class
            if gt < 100:
                density_class = 'sparse'
            elif gt < 500:
                density_class = 'medium'
            else:
                density_class = 'dense'
            
            # Salva risultati
            img_result = {
                'idx': batch_idx,
                'gt': gt,
                'pred_raw': pred_raw,
                'pred_masked': pred_masked,
                'err_raw': err_raw,
                'err_masked': err_masked,
                'bias_raw': bias_raw,
                'bias_masked': bias_masked,
                'coverage': coverage,
                'pi_mean': pi_mean,
                'density_class': density_class,
                'improved': err_masked < err_raw,
            }
            
            results['per_image'].append(img_result)
            results['by_density'][density_class].append(img_result)
            
            all_mae_raw.append(err_raw)
            all_mae_masked.append(err_masked)
            all_mse_raw.append(err_raw ** 2)
            all_mse_masked.append(err_masked ** 2)
            if gt > 0:
                all_bias_raw.append(bias_raw)
                all_bias_masked.append(bias_masked)
    
    # Calcola metriche aggregate
    n = len(all_mae_raw)
    
    results['summary'] = {
        'n_samples': n,
        'tau': tau,
        'mae_raw': np.mean(all_mae_raw),
        'mae_masked': np.mean(all_mae_masked),
        'rmse_raw': np.sqrt(np.mean(all_mse_raw)),
        'rmse_masked': np.sqrt(np.mean(all_mse_masked)),
        'bias_raw': np.mean(all_bias_raw),
        'bias_masked': np.mean(all_bias_masked),
        'improvement': np.mean(all_mae_raw) - np.mean(all_mae_masked),
        'n_improved': sum(1 for r in results['per_image'] if r['improved']),
        'pct_improved': sum(1 for r in results['per_image'] if r['improved']) / n * 100,
    }
    
    # Metriche per densità
    results['by_density_summary'] = {}
    for density_class in ['sparse', 'medium', 'dense']:
        class_results = results['by_density'][density_class]
        if class_results:
            results['by_density_summary'][density_class] = {
                'n': len(class_results),
                'mae_raw': np.mean([r['err_raw'] for r in class_results]),
                'mae_masked': np.mean([r['err_masked'] for r in class_results]),
                'improvement': np.mean([r['err_raw'] for r in class_results]) - np.mean([r['err_masked'] for r in class_results]),
            }
    
    return results


def print_results(results):
    """Stampa risultati formattati."""
    
    summary = results['summary']
    
    print("\n" + "=" * 70)
    print("📊 RISULTATI VALUTAZIONE FINALE")
    print("=" * 70)
    
    print(f"\nCampioni: {summary['n_samples']}")
    print(f"Soglia τ: {summary['tau']}")
    
    print("\n" + "-" * 70)
    print("🎯 METRICHE GLOBALI")
    print("-" * 70)
    
    print(f"\n{'Metrica':<20} {'RAW':<15} {'MASKED (τ={:.1f})':<15} {'Δ':<10}".format(summary['tau']))
    print("-" * 60)
    print(f"{'MAE':<20} {summary['mae_raw']:<15.2f} {summary['mae_masked']:<15.2f} {summary['improvement']:+.2f}")
    print(f"{'RMSE':<20} {summary['rmse_raw']:<15.2f} {summary['rmse_masked']:<15.2f} {summary['rmse_raw'] - summary['rmse_masked']:+.2f}")
    print(f"{'Bias':<20} {summary['bias_raw']:<15.3f} {summary['bias_masked']:<15.3f} {summary['bias_raw'] - summary['bias_masked']:+.3f}")
    
    print(f"\n📈 Miglioramento: {summary['improvement']:+.2f} MAE")
    print(f"   Immagini migliorate: {summary['n_improved']}/{summary['n_samples']} ({summary['pct_improved']:.1f}%)")
    
    print("\n" + "-" * 70)
    print("📊 PER CLASSE DI DENSITÀ")
    print("-" * 70)
    
    for density_class in ['sparse', 'medium', 'dense']:
        if density_class in results['by_density_summary']:
            d = results['by_density_summary'][density_class]
            status = "✅" if d['improvement'] > 0 else "⚠️"
            print(f"\n{density_class.upper()} ({d['n']} imgs):")
            print(f"   RAW:    MAE = {d['mae_raw']:.2f}")
            print(f"   MASKED: MAE = {d['mae_masked']:.2f} ({d['improvement']:+.2f}) {status}")
    
    # Top 5 miglioramenti e peggioramenti
    print("\n" + "-" * 70)
    print("🔍 ANALISI DETTAGLIATA")
    print("-" * 70)
    
    per_image = results['per_image']
    
    # Sort by improvement (err_raw - err_masked)
    sorted_by_improvement = sorted(per_image, key=lambda x: x['err_raw'] - x['err_masked'], reverse=True)
    
    print("\n📈 Top 5 MIGLIORAMENTI:")
    for r in sorted_by_improvement[:5]:
        imp = r['err_raw'] - r['err_masked']
        print(f"   GT={r['gt']:3d} | RAW={r['pred_raw']:.0f} ({r['err_raw']:.1f}) → MASKED={r['pred_masked']:.0f} ({r['err_masked']:.1f}) | Δ={imp:+.1f}")
    
    print("\n📉 Top 5 PEGGIORAMENTI:")
    for r in sorted_by_improvement[-5:]:
        imp = r['err_raw'] - r['err_masked']
        print(f"   GT={r['gt']:3d} | RAW={r['pred_raw']:.0f} ({r['err_raw']:.1f}) → MASKED={r['pred_masked']:.0f} ({r['err_masked']:.1f}) | Δ={imp:+.1f}")
    
    # Risultato finale
    print("\n" + "=" * 70)
    print("🏁 RISULTATO FINALE")
    print("=" * 70)
    print(f"""
   ┌────────────────────────────────────────┐
   │  MAE RAW:      {summary['mae_raw']:6.2f}                 │
   │  MAE MASKED:   {summary['mae_masked']:6.2f}  (τ = {summary['tau']})       │
   │  ─────────────────────────────────     │
   │  MIGLIORAMENTO: {summary['improvement']:+5.2f}                 │
   │  BIAS:          {summary['bias_masked']:.3f}                  │
   └────────────────────────────────────────┘
""")
    
    if summary['mae_masked'] < 65:
        print("   🎉 TARGET RAGGIUNTO! MAE < 65")
    elif summary['mae_masked'] < 68:
        print("   ✅ Ottimo risultato! MAE < 68")
    
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tau', type=float, default=0.2, help='Soglia π')
    parser.add_argument('--save', type=str, default=None, help='Salva risultati JSON')
    args = parser.parse_args()
    
    # Carica config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    data_cfg = cfg['DATA']
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    
    print("=" * 60)
    print(f"🔍 VALUTAZIONE FINALE (τ = {args.tau})")
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
        # Auto-detect
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
    
    # Valutazione
    results = evaluate_with_threshold(
        model, val_loader, device, default_down,
        tau=args.tau
    )
    
    # Stampa risultati
    print_results(results)
    
    # Salva se richiesto
    if args.save:
        # Converti numpy per JSON
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_json = convert_numpy(results)
        results_json['timestamp'] = datetime.now().isoformat()
        results_json['checkpoint'] = ckpt_path
        
        with open(args.save, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n💾 Risultati salvati in: {args.save}")


if __name__ == "__main__":
    main()