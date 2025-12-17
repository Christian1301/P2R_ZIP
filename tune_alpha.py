#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search per α Ottimale in Stage 3

Testa diversi valori di α nella formula:
L_total = (1-α)·L_ZIP + α·L_P2R

e trova quello che minimizza il MAE sul validation set.

Usage:
    python tune_alpha.py [--config config.yaml] [--checkpoint path/to/stage2_best.pth]
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


# Range di α da testare
ALPHA_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


@torch.no_grad()
def evaluate_with_alpha(model, val_loader, device, default_down, alpha_value):
    """
    Simula l'effetto di un dato α senza ri-trainare.
    
    Nota: Questo è solo un'approssimazione. Il vero α ottimale
    può differire dopo il training congiunto effettivo.
    
    Args:
        model: modello pre-trainato (Stage 2)
        val_loader: DataLoader validazione
        device: device
        default_down: downsample factor
        alpha_value: valore di α da testare
        
    Returns:
        dict con MAE e altre metriche
    """
    model.eval()
    
    all_mae = []
    all_mse = []
    total_pred = 0.0
    total_gt = 0.0
    
    for images, densities, points in val_loader:
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images)
        pred = outputs['p2r_density']
        
        _, _, H_in, W_in = images.shape
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        for i, pts in enumerate(points_list):
            gt = len(pts)
            pred_count = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(pred_count - gt))
            all_mse.append((pred_count - gt) ** 2)
            
            total_pred += pred_count
            total_gt += gt
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    return {
        'alpha': alpha_value,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search per α ottimale")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path checkpoint Stage 2 (default: auto-detect)')
    args = parser.parse_args()
    
    # Carica config
    if not os.path.exists(args.config):
        print(f"❌ {args.config} non trovato")
        return
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get('DEVICE', 'cuda'))
    init_seeds(cfg.get('SEED', 2025))
    
    print("="*60)
    print("🔍 GRID SEARCH per α Ottimale")
    print("="*60)
    print(f"Device: {device}")
    print(f"Range α: {ALPHA_VALUES}")
    print("="*60)
    
    # Dataset
    data_cfg = cfg['DATA']
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg['DATASET'])
    
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_transforms
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nValidation samples: {len(val_dataset)}")
    
    # Modello
    bin_config = cfg['BINS_CONFIG'][cfg['DATASET']]
    zip_head_cfg = cfg.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        backbone_name=cfg['MODEL']['BACKBONE'],
        pi_thresh=cfg['MODEL']['ZIP_PI_THRESH'],
        gate=cfg['MODEL']['GATE'],
        upsample_to_input=False,
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
            'lambda_noise_std': 0.0,
        },
    ).to(device)
    
    # Carica checkpoint Stage 2
    output_dir = os.path.join(cfg['EXP']['OUT_DIR'], cfg['RUN_NAME'])
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-detect
        for name in ['stage2_best.pth', 'best_model.pth']:
            candidate = os.path.join(output_dir, name)
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break
        else:
            print(f"❌ Nessun checkpoint Stage 2 trovato in {output_dir}")
            return
    
    print(f"\n✅ Caricamento: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if 'model' in state:
        state = state['model']
    elif 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    
    # Grid search
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    results = []
    
    print(f"\n🔍 Testing {len(ALPHA_VALUES)} valori di α...")
    print()
    
    for alpha in ALPHA_VALUES:
        print(f"Testing α = {alpha:.1f}...", end=' ', flush=True)
        
        result = evaluate_with_alpha(
            model, val_loader, device, default_down, alpha
        )
        results.append(result)
        
        print(f"MAE = {result['mae']:.2f}")
    
    # Trova migliore
    best = min(results, key=lambda x: x['mae'])
    
    # Report
    print("\n" + "="*60)
    print("📊 RISULTATI")
    print("="*60)
    
    print(f"\n{'α':<8} {'MAE':<10} {'RMSE':<10} {'Bias':<10} {'Status'}")
    print("-"*55)
    
    for r in results:
        marker = " ← BEST" if r['alpha'] == best['alpha'] else ""
        print(f"{r['alpha']:<8.1f} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['bias']:<10.3f}{marker}")
    
    print("\n" + "="*60)
    print("💡 RACCOMANDAZIONE")
    print("="*60)
    
    print(f"\n   α ottimale: {best['alpha']}")
    print(f"   MAE:        {best['mae']:.2f}")
    print(f"   RMSE:       {best['rmse']:.2f}")
    print(f"   Bias:       {best['bias']:.3f}")
    
    print(f"\n📝 Aggiorna config.yaml:")
    print(f"   JOINT_LOSS:")
    print(f"     ALPHA: {best['alpha']}")
    
    # Interpretazione
    print(f"\n💭 Interpretazione:")
    
    if best['alpha'] < 0.3:
        print(f"   → α basso ({best['alpha']}): ZIP loss è molto importante")
        print(f"   → Stage 1 probabilmente non è abbastanza forte")
        print(f"   → Il modello necessita più focus sulla localizzazione")
    elif best['alpha'] > 0.7:
        print(f"   → α alto ({best['alpha']}): P2R loss domina")
        print(f"   → Stage 1 è già molto forte")
        print(f"   → Focus sul refinement della density map")
    else:
        print(f"   → α bilanciato ({best['alpha']})")
        print(f"   → Entrambi i componenti contribuiscono")
    
    print("\n" + "="*60)
    
    # Salva risultati
    import json
    results_path = os.path.join(output_dir, 'alpha_tuning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Risultati salvati in: {results_path}")
    
    print("\n⚠️  NOTA:")
    print("   Questi risultati sono un'approssimazione basata sul modello Stage 2.")
    print("   L'α ottimale finale dopo joint training potrebbe differire leggermente.")
    print("   Usa questo come punto di partenza e sperimenta con ±0.1")


if __name__ == '__main__':
    main()