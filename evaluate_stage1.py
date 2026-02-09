#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 evaluation - ZIP pre-training metrics.

Computes block-level classification metrics (accuracy, precision, recall, F1),
pi/lambda statistics, and indicative count MAE using pi*lambda. Supports
val/test split selection, automatic checkpoint discovery, image resizing
for OOM prevention, and optional JSON results export.
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
import argparse
from datetime import datetime

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

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
def validate_with_metrics(model, dataloader, device, block_size, pi_threshold=0.3, max_size=2048):
    model.eval()

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    pi_mean_list = []
    pi_std_list = []
    lambda_mean_list = []
    lambda_max_list = []
    coverage_list = []

    total_mae = 0.0
    total_pred_count = 0.0
    total_gt_count = 0.0
    n_samples = 0

    all_gt_counts = []
    all_pred_counts = []

    for images, gt_density, points in tqdm(dataloader, desc="Evaluating Stage 1"):
        images, gt_density = images.to(device), gt_density.to(device)

        images, scale = resize_if_needed(images, max_size)
        if scale != 1.0:
            gt_density = F.interpolate(gt_density, size=images.shape[-2:], mode='bilinear', align_corners=False)

        preds = model(images)

        pi_probs = preds.get('pi_probs')
        if pi_probs is None:
            pi_logits = preds.get("logit_pi_maps")
            if pi_logits is not None:
                pi_probs = torch.sigmoid(pi_logits[:, 1:2])
            else:
                raise KeyError("No pi_probs or logit_pi_maps in model output")

        lambda_maps = preds.get('lambda_maps')

        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=block_size,
            stride=block_size
        ) * (block_size ** 2)
        gt_occupancy = (gt_counts_per_block > 0.5).float()

        if gt_occupancy.shape[-2:] != pi_probs.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy,
                size=pi_probs.shape[-2:],
                mode='nearest'
            )

        pred_occupancy = (pi_probs > pi_threshold).float()

        tp = ((pred_occupancy == 1) & (gt_occupancy == 1)).sum().item()
        tn = ((pred_occupancy == 0) & (gt_occupancy == 0)).sum().item()
        fp = ((pred_occupancy == 1) & (gt_occupancy == 0)).sum().item()
        fn = ((pred_occupancy == 0) & (gt_occupancy == 1)).sum().item()

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        pi_mean_list.append(pi_probs.mean().item())
        pi_std_list.append(pi_probs.std().item())
        coverage_list.append((pi_probs > pi_threshold).float().mean().item() * 100)

        if lambda_maps is not None:
            lambda_mean_list.append(lambda_maps.mean().item())
            lambda_max_list.append(lambda_maps.max().item())

        if lambda_maps is not None:
            pred_count_map = pi_probs * lambda_maps

            for idx, pts in enumerate(points):
                gt = len(pts) if pts is not None else 0
                pred = pred_count_map[idx].sum().item()

                total_mae += abs(pred - gt)
                total_pred_count += pred
                total_gt_count += gt
                n_samples += 1

                all_gt_counts.append(gt)
                all_pred_counts.append(pred)

    total_samples = total_tp + total_tn + total_fp + total_fn

    accuracy = (total_tp + total_tn) / max(total_samples, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)

    avg_pi_mean = np.mean(pi_mean_list)
    avg_pi_std = np.mean(pi_std_list)
    avg_coverage = np.mean(coverage_list)

    avg_lambda_mean = np.mean(lambda_mean_list) if lambda_mean_list else 0
    avg_lambda_max = np.mean(lambda_max_list) if lambda_max_list else 0

    mae = total_mae / n_samples if n_samples > 0 else 0
    bias = total_pred_count / total_gt_count if total_gt_count > 0 else 0

    all_gt_counts = np.array(all_gt_counts)
    all_pred_counts = np.array(all_pred_counts)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'tp': total_tp,
            'tn': total_tn,
            'fp': total_fp,
            'fn': total_fn,
            'total': total_samples,
        },
        'pi_stats': {
            'mean': avg_pi_mean,
            'std': avg_pi_std,
            'coverage': avg_coverage,
        },
        'lambda_stats': {
            'mean': avg_lambda_mean,
            'max': avg_lambda_max,
        },
        'count_metrics': {
            'mae': mae,
            'bias': bias,
            'gt_total': int(total_gt_count),
            'pred_total': float(total_pred_count),
            'gt_mean': float(np.mean(all_gt_counts)) if len(all_gt_counts) > 0 else 0,
            'pred_mean': float(np.mean(all_pred_counts)) if len(all_pred_counts) > 0 else 0,
        },
        'num_images': n_samples,
    }


def print_results(results, split):
    print("\n" + "="*70)
    print(f"STAGE 1 EVALUATION RESULTS - {split.upper()} SET")
    print("="*70)

    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"   True Positives (TP):   {cm['tp']:,}")
    print(f"   True Negatives (TN):   {cm['tn']:,}")
    print(f"   False Positives (FP):  {cm['fp']:,}")
    print(f"   False Negatives (FN):  {cm['fn']:,}")
    print(f"   Total blocks:          {cm['total']:,}")

    print(f"\nClassification Metrics:")
    print(f"   Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   Precision: {results['precision']*100:.2f}%")
    print(f"   Recall:    {results['recall']*100:.2f}%")
    print(f"   F1-Score:  {results['f1_score']*100:.2f}%")

    print(f"\nPi-Head Statistics:")
    pi = results['pi_stats']
    print(f"   Pi Mean:     {pi['mean']:.3f}")
    print(f"   Pi Std:      {pi['std']:.3f}")
    print(f"   Coverage:   {pi['coverage']:.1f}%")

    print(f"\nLambda-Head Statistics:")
    lam = results['lambda_stats']
    print(f"   Lambda Mean:     {lam['mean']:.3f}")
    print(f"   Lambda Max:      {lam['max']:.3f}")

    print(f"\nCount Metrics (Indicative - pi*lambda):")
    cnt = results['count_metrics']
    print(f"   MAE:        {cnt['mae']:.2f}")
    print(f"   Bias:       {cnt['bias']:.3f}")
    print(f"   GT Total:   {cnt['gt_total']:,}")
    print(f"   Pred Total: {cnt['pred_total']:,.0f}")
    print(f"   GT Mean:    {cnt['gt_mean']:.1f}")
    print(f"   Pred Mean:  {cnt['pred_mean']:.1f}")

    print(f"\nImages evaluated: {results['num_images']}")

    print("\n" + "-"*70)
    print("Interpretation:")

    if results['recall'] < 0.85:
        print(f"   Recall low ({results['recall']*100:.1f}%)")
        print(f"       -> Model misses too many occupied blocks (high FN)")
    elif results['recall'] > 0.90:
        print(f"   Recall high ({results['recall']*100:.1f}%)")
        print(f"       -> Good coverage of occupied blocks")

    if results['precision'] < 0.70:
        print(f"   Precision low ({results['precision']*100:.1f}%)")
        print(f"       -> Too many false positives (high FP)")
    elif results['precision'] > 0.85:
        print(f"   Precision high ({results['precision']*100:.1f}%)")
        print(f"       -> Few false detections")

    if results['f1_score'] > 0.80:
        print(f"   F1-Score high ({results['f1_score']*100:.1f}%)")
    elif results['f1_score'] < 0.70:
        print(f"   F1-Score low ({results['f1_score']*100:.1f}%)")

    if pi['coverage'] < 10:
        print(f"   Coverage very low ({pi['coverage']:.1f}%)")
    elif pi['coverage'] > 40:
        print(f"   Coverage very high ({pi['coverage']:.1f}%)")
    else:
        print(f"   Coverage reasonable ({pi['coverage']:.1f}%)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage 1 ZIP')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Split to use (val or test)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: auto-discover)')
    parser.add_argument('--pi-threshold', type=float, default=0.5,
                        help='Threshold for pi (default: 0.5)')
    parser.add_argument('--max-size', type=int, default=2048,
                        help='Max image size per side to avoid OOM (default: 2048)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"{args.config} not found")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    print("="*70)
    print(f"EVALUATION STAGE 1 - ZIP Pre-training ({args.split.upper()} SET)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Pi threshold: {args.pi_threshold}")
    print("="*70)

    dataset_section = config.get('DATASET', {})
    data_section = config.get('DATA')

    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = dict(dataset_section)
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}

    normalized = ''.join(c for c in str(dataset_name_raw).lower() if c.isalnum())
    dataset_name = DATASET_ALIAS_MAP.get(normalized, dataset_name_raw.lower() if isinstance(dataset_name_raw, str) else 'shha')

    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"No BINS_CONFIG for dataset '{dataset_name}'")

    bins, bin_centers = bin_config['bins'], bin_config['bin_centers']

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    block_size = data_cfg.get('ZIP_BLOCK_SIZE', data_cfg.get('BLOCK_SIZE', 16))

    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 1.2),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
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

    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(
        exp_cfg.get('OUT_DIR', config.get('EXPERIMENT_DIR', 'exp')),
        config.get('RUN_NAME', 'stage1')
    )

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(output_dir, "best_model.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(output_dir, "last.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found in {output_dir}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = raw_state.get('model', raw_state)
    model.load_state_dict(state_dict, strict=False)

    if 'epoch' in raw_state:
        print(f"   Epoch: {raw_state['epoch']}")
    if 'best_metrics' in raw_state:
        bm = raw_state['best_metrics']
        print(f"   Best F1: {bm.get('f1', 0)*100:.1f}%, Recall: {bm.get('recall', 0)*100:.1f}%")

    DatasetClass = get_dataset(dataset_name)
    val_tf = build_transforms(data_cfg, is_train=False)

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
        transforms=val_tf,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('OPTIM_ZIP', {}).get('NUM_WORKERS', 4),
        collate_fn=collate_fn
    )

    print(f"\nDataset: {dataset_name}")
    print(f"   Split: {split_name}")
    print(f"   Images: {len(dataset)}")

    results = validate_with_metrics(
        model, dataloader, device, block_size,
        pi_threshold=args.pi_threshold,
        max_size=args.max_size
    )

    print_results(results, args.split)

    if args.save_results or args.output:
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"stage1_eval_{args.split}_{timestamp}.json")

        results_json = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'confusion_matrix': results['confusion_matrix'],
            'pi_stats': {k: float(v) for k, v in results['pi_stats'].items()},
            'lambda_stats': {k: float(v) for k, v in results['lambda_stats'].items()},
            'count_metrics': results['count_metrics'],
            'num_images': results['num_images'],
            'metadata': {
                'checkpoint': checkpoint_path,
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


if __name__ == '__main__':
    main()
