#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 evaluation - P2R density head metrics.

Evaluates the P2R head using MAE, RMSE, MSE, NAE, and bias. Includes
per-density-band MAE analysis, pi coverage statistics, and log_scale
reporting from the checkpoint. Supports val/test split selection and
automatic checkpoint discovery.
"""

import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import argparse
from datetime import datetime

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, canonicalize_p2r_grid


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

DATASET_ALIASES = {
    'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
    'shhb': 'shhb', 'shanghaitechpartb': 'shhb',
    'ucf': 'ucf', 'ucfqnrf': 'ucf',
    'nwpu': 'nwpu', 'jhu': 'jhu',
}


def resize_if_needed(images, max_size=2048):
    _, _, H, W = images.shape
    max_dim = max(H, W)

    if max_dim > max_size:
        scale = max_size / max_dim
        new_H = int(H * scale)
        new_W = int(W * scale)
        new_H = (new_H // 32) * 32
        new_W = (new_W // 32) * 32
        images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return images, scale
    return images, 1.0


@torch.no_grad()
def validate_stage2(model, dataloader, device, default_down, pi_threshold=0.5, max_size=2048):
    model.eval()

    all_mae = []
    all_mse = []
    all_errors = []
    total_pred = 0.0
    total_gt = 0.0

    all_gt_counts = []
    all_pred_counts = []

    density_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    density_bin_errors = {i: [] for i in range(len(density_bins) - 1)}

    pi_coverages = []
    pi_means = []

    for images, gt_density, points in tqdm(dataloader, desc="Evaluating Stage 2"):
        images = images.to(device)
        points_list = points

        images, scale = resize_if_needed(images, max_size)

        outputs = model(images)
        pred = outputs['p2r_density']
        pi_probs = outputs.get('pi_probs')

        _, _, H_in, W_in = images.shape

        pred, down_tuple, _ = canonicalize_p2r_grid(pred, (H_in, W_in), default_down)

        down_h, down_w = down_tuple
        cell_area = down_h * down_w

        if pi_probs is not None:
            coverage = (pi_probs > pi_threshold).float().mean().item() * 100
            pi_coverages.append(coverage)
            pi_means.append(pi_probs.mean().item())

        for i, pts in enumerate(points_list):
            gt = len(pts) if pts is not None else 0
            pred_count = (pred[i].sum() / cell_area).item()
            error = abs(pred_count - gt)

            all_mae.append(error)
            all_mse.append((pred_count - gt) ** 2)
            all_errors.append(error)

            total_pred += pred_count
            total_gt += gt

            all_gt_counts.append(gt)
            all_pred_counts.append(pred_count)

            for bin_idx in range(len(density_bins) - 1):
                if density_bins[bin_idx] <= gt < density_bins[bin_idx + 1]:
                    density_bin_errors[bin_idx].append(error)
                    break

    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_mse))
    bias = total_pred / total_gt if total_gt > 0 else 0

    all_gt_counts = np.array(all_gt_counts)
    all_pred_counts = np.array(all_pred_counts)

    nae = np.sum(all_errors) / np.sum(all_gt_counts) if np.sum(all_gt_counts) > 0 else 0

    density_mae = {}
    density_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    for bin_idx, label in enumerate(density_labels):
        errors = density_bin_errors[bin_idx]
        if len(errors) > 0:
            density_mae[label] = {
                'mae': float(np.mean(errors)),
                'count': len(errors),
            }

    return {
        'mae': mae,
        'rmse': rmse,
        'mse': np.mean(all_mse),
        'bias': bias,
        'nae': nae,
        'gt_total': int(total_gt),
        'pred_total': float(total_pred),
        'gt_mean': float(np.mean(all_gt_counts)),
        'pred_mean': float(np.mean(all_pred_counts)),
        'gt_std': float(np.std(all_gt_counts)),
        'pred_std': float(np.std(all_pred_counts)),
        'gt_min': int(np.min(all_gt_counts)),
        'gt_max': int(np.max(all_gt_counts)),
        'pred_min': float(np.min(all_pred_counts)),
        'pred_max': float(np.max(all_pred_counts)),
        'pi_coverage': float(np.mean(pi_coverages)) if pi_coverages else 0,
        'pi_mean': float(np.mean(pi_means)) if pi_means else 0,
        'density_mae': density_mae,
        'num_images': len(all_gt_counts),
        'log_scale': model.p2r_head.get_log_scale() if hasattr(model.p2r_head, 'get_log_scale') else 0,
        'scale': model.p2r_head.get_scale() if hasattr(model.p2r_head, 'get_scale') else 1,
    }


def print_results(results, split):
    print("\n" + "="*70)
    print(f"STAGE 2 EVALUATION RESULTS - {split.upper()} SET")
    print("="*70)

    print(f"\nMain Metrics:")
    print(f"   MAE:  {results['mae']:.2f}")
    print(f"   RMSE: {results['rmse']:.2f}")
    print(f"   MSE:  {results['mse']:.2f}")
    print(f"   NAE:  {results['nae']:.4f}")
    print(f"   Bias: {results['bias']:.3f}")

    print(f"\nCount Statistics:")
    print(f"   GT Total:   {results['gt_total']:,}")
    print(f"   Pred Total: {results['pred_total']:,.0f}")
    print(f"   GT Mean:    {results['gt_mean']:.1f} +/- {results['gt_std']:.1f}")
    print(f"   Pred Mean:  {results['pred_mean']:.1f} +/- {results['pred_std']:.1f}")
    print(f"   GT Range:   [{results['gt_min']}, {results['gt_max']}]")
    print(f"   Pred Range: [{results['pred_min']:.1f}, {results['pred_max']:.1f}]")

    print(f"\nP2R Head Statistics:")
    print(f"   log_scale: {results['log_scale']:.4f}")
    print(f"   scale:     {results['scale']:.2f}")

    print(f"\nZIP Statistics:")
    print(f"   pi coverage: {results['pi_coverage']:.1f}%")
    print(f"   pi mean:     {results['pi_mean']:.3f}")

    print(f"\nMAE per Density Band:")
    for label, data in results['density_mae'].items():
        print(f"   {label:>10}: MAE={data['mae']:.2f} (n={data['count']})")

    print(f"\nImages evaluated: {results['num_images']}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage 2')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pi-threshold', type=float, default=0.5)
    parser.add_argument('--max-size', type=int, default=2048)
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    device = torch.device(config.get('DEVICE', 'cuda'))
    init_seeds(config.get('SEED', 42))

    print("="*70)
    print(f"EVALUATION STAGE 2 ({args.split.upper()} SET)")
    print("="*70)

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

    normalized_name = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = DATASET_ALIASES.get(normalized_name, str(dataset_name_raw).lower())

    if not data_cfg:
        raise KeyError("Dataset configuration missing.")

    if 'ROOT' not in data_cfg:
        raise KeyError("Dataset ROOT path missing.")

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))
    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)

    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"BINS_CONFIG missing for dataset '{dataset_name}'")

    model_cfg = config.get('MODEL', {})
    zip_head_cfg = config.get('ZIP_HEAD', {})
    p2r_loss_cfg = config.get('P2R_LOSS', {})

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
        p2r_head_kwargs={
            'fea_channel': model_cfg.get('P2R_FEA_CHANNEL', 256),
            'log_scale_init': p2r_loss_cfg.get('LOG_SCALE_INIT', 4.0),
            'log_scale_clamp': tuple(p2r_loss_cfg.get('LOG_SCALE_CLAMP', [-2.0, 10.0])),
        },
    ).to(device)

    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), config.get('RUN_NAME', 'stage2_eval'))

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        candidates = ['stage2_bypass_best.pth', 'stage2_best.pth', 'stage2_bypass_last.pth']
        ckpt_path = None
        for c in candidates:
            path = os.path.join(output_dir, c)
            if os.path.isfile(path):
                ckpt_path = path
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"No Stage 2 checkpoint found in {output_dir}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)

    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'mae' in checkpoint:
        print(f"   Checkpoint MAE: {checkpoint['mae']:.2f}")
    if 'best_mae' in checkpoint:
        print(f"   Best MAE: {checkpoint['best_mae']:.2f}")

    if hasattr(model.p2r_head, 'log_scale'):
        ls = model.p2r_head.log_scale.item()
        print(f"\nlog_scale: {ls:.4f} (scale={np.exp(ls):.2f})")
        print(f"   Using value from checkpoint, no re-calibration.")

    DatasetClass = get_dataset(dataset_name)
    val_transforms = build_transforms(data_cfg, is_train=False)

    if args.split == 'test':
        split_name = data_cfg.get('TEST_SPLIT', None)
        if split_name is None:
            for candidate in ['test', 'testing', 'eval']:
                test_path = os.path.join(data_cfg['ROOT'], candidate)
                if os.path.isdir(test_path):
                    split_name = candidate
                    break
            if split_name is None:
                split_name = 'test'
    else:
        split_name = data_cfg.get('VAL_SPLIT', 'val')

    dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=split_name,
        block_size=block_size,
        transforms=val_transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print(f"\nDataset: {dataset_name}")
    print(f"   Split: {split_name}")
    print(f"   Images: {len(dataset)}")

    results = validate_stage2(model, dataloader, device, default_down, args.pi_threshold, args.max_size)

    print_results(results, args.split)

    if 'mae' in checkpoint:
        train_mae = checkpoint['mae']
        diff = abs(results['mae'] - train_mae)
        print(f"\nComparison with Training:")
        print(f"   Training MAE:   {train_mae:.2f}")
        print(f"   Evaluation MAE: {results['mae']:.2f}")
        print(f"   Difference:     {diff:.2f}")

        if diff < 1.0:
            print(f"\n   ALIGNED! Difference < 1 point")
        else:
            print(f"\n   Discrepancy of {diff:.1f} points")

    if args.save_results or args.output:
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"stage2_eval_{args.split}_{timestamp}.json")

        results_json = {
            **results,
            'metadata': {
                'checkpoint': ckpt_path,
                'config': args.config,
                'split': args.split,
                'pi_threshold': args.pi_threshold,
                'timestamp': datetime.now().isoformat(),
            }
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved: {output_path}")

    print("="*70)


if __name__ == '__main__':
    main()
