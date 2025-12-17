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
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])
    
    print("="*60)
    print("🔍 EVALUATION STAGE 2 (Allineato)")
    print("="*60)
    
    # Setup modello
    data_cfg = config['DATA']
    bin_config = config['BINS_CONFIG'][config['DATASET']]
    
    zip_head_cfg = config.get('ZIP_HEAD', {})
    
    model = P2R_ZIP_Model(
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        gate=config['MODEL']['GATE'],
        upsample_to_input=False,
        use_ste_mask=True,
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)
    
    # Carica checkpoint
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    ckpt_path = os.path.join(output_dir, 'stage2_best.pth')
    
    print(f"\n✅ Caricamento: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Mostra log_scale (NON modificarlo!)
    if hasattr(model.p2r_head, 'log_scale'):
        ls = model.p2r_head.log_scale.item()
        print(f"\n📊 log_scale: {ls:.4f} (scala={np.exp(ls):.2f})")
        print(f"   ⚠️ Usando valore dal checkpoint, NESSUNA ri-calibrazione!")
    
    # Dataset
    DatasetClass = get_dataset(config['DATASET'])
    val_transforms = build_transforms(data_cfg, is_train=False)
    
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
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