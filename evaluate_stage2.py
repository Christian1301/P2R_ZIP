#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Stage 2 - SENZA ri-calibrazione

⚠️ CRITICO: Usa log_scale dal checkpoint, NON ri-calibrare!
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


# Default bin definitions shared across scripts so evaluate works even if
# the config does not explicitly contain BINS_CONFIG.
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


@torch.no_grad()
def validate_stage2(model, dataloader, device, default_down):
    """Validazione IDENTICA al training."""
    model.eval()
    
    all_mae = []
    all_mse = []
    total_pred = 0.0
    total_gt = 0.0
    
    for images, gt_density, points in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        points_list = points
        
        # Forward - IDENTICO al training
        outputs = model(images)
        pred = outputs['p2r_density']
        
        _, _, H_in, W_in = images.shape
        
        # Canonicalize - IDENTICO al training
        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Calcolo MAE - IDENTICO al training
        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = (pred[i].sum() / cell_area).item()
            
            all_mae.append(abs(pred_count - gt))
            all_mse.append((pred_count - gt) ** 2)
            
            total_pred += pred_count
            total_gt += gt
    
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    bias = total_pred / total_gt if total_gt > 0 else 0
    
    return {'mae': mae, 'rmse': rmse, 'bias': bias}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Stage 2')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path al file di configurazione YAML')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    
    device = torch.device(config.get('DEVICE', 'cuda'))
    init_seeds(config.get('SEED', 42))
    
    print("="*60)
    print("🔍 EVALUATION STAGE 2 (Allineato)")
    print("="*60)
    
    # Setup modello / dataset compatto anche per vecchi config
    dataset_section = config.get('DATASET', 'shha')
    data_section = config.get('DATA')

    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = {}
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
        data_cfg.update(dataset_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}

    alias_map = {
        'shha': 'shha',
        'shanghaitecha': 'shha',
        'shanghaitechparta': 'shha',
        'shanghaitechaparta': 'shha',
        'shhb': 'shhb',
        'shanghaitechpartb': 'shhb',
        'ucf': 'ucf',
        'ucfqnrf': 'ucf',
        'nwpu': 'nwpu',
        'jhu': 'jhu',
    }
    normalized_name = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = alias_map.get(normalized_name, str(dataset_name_raw).lower())

    if not data_cfg:
        raise KeyError("Dataset configuration missing. Provide DATA or DATASET entries with ROOT information.")

    if 'ROOT' not in data_cfg:
        raise KeyError("Dataset ROOT path missing in DATA / DATASET configuration.")

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    val_split = data_cfg.get('VAL_SPLIT', data_cfg.get('SPLIT', 'val'))
    zip_block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))

    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"BINS_CONFIG missing definition for dataset '{dataset_name}' and no default is available.")
    
    model_cfg = config.get('MODEL', {})
    zip_head_cfg = config.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        use_ste_mask=model_cfg.get('USE_STE_MASK', False),
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Carica checkpoint
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), config.get('RUN_NAME', 'stage2_eval'))
    ckpt_path = os.path.join(output_dir, 'stage2_bypass_best.pth')
    
    print(f"\n✅ Caricamento: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Mostra log_scale (NON modificarlo!)
    if hasattr(model.p2r_head, 'log_scale'):
        ls = model.p2r_head.log_scale.item()
        print(f"\n📊 log_scale: {ls:.4f} (scala={np.exp(ls):.2f})")
        print(f"   ⚠️ Usando valore dal checkpoint, NESSUNA ri-calibrazione!")
    
    # Dataset
    DatasetClass = get_dataset(dataset_name)
    val_transforms = build_transforms(data_cfg, is_train=False)
    
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=val_split,
        block_size=zip_block_size,
        transforms=val_transforms,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Valida
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    results = validate_stage2(model, val_loader, device, default_down)
    
    # Report
    print("\n" + "="*60)
    print("📊 RISULTATI")
    print("="*60)
    print(f"   MAE:  {results['mae']:.2f}")
    print(f"   RMSE: {results['rmse']:.2f}")
    print(f"   Bias: {results['bias']:.3f}")
    
    # Confronto
    if 'mae' in checkpoint:
        train_mae = checkpoint['mae']
        diff = abs(results['mae'] - train_mae)
        print(f"\n📈 Confronto con Training:")
        print(f"   Training MAE:   {train_mae:.2f}")
        print(f"   Evaluation MAE: {results['mae']:.2f}")
        print(f"   Differenza:     {diff:.2f}")
        
        if diff < 1.0:
            print(f"\n   ✅ ALLINEATO! Differenza < 1 punto")
        else:
            print(f"\n   ⚠️ Ancora discrepanza di {diff:.1f} punti")
    
    print("="*60)

if __name__ == '__main__':
    main()