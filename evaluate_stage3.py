#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 evaluation with scale compensation, TTA, and confusion matrix.

Evaluates three density modes (raw, hard-masked, soft-weighted) with optional
test-time augmentation (flip and multi-scale). Generates a pixel-level confusion
matrix for ZIP head classification performance and per-density-band MAE analysis.
Supports automatic checkpoint and scale compensation loading.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import math
import json
from datetime import datetime

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn, canonicalize_p2r_grid


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


class ScaleCompensation(nn.Module):
    def __init__(self, init_scale: float = 1.0, learnable: bool = True):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale)),
            requires_grad=learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale.clamp(-2, 2))
        return x * scale

    def get_scale(self) -> float:
        with torch.no_grad():
            return torch.exp(self.log_scale.clamp(-2, 2)).item()


def apply_soft_weighting(density, pi_probs, alpha=0.2):
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
        )
    weights = (1 - alpha) + alpha * pi_probs
    return density * weights


def apply_hard_mask(density, pi_probs, threshold=0.5):
    if pi_probs.shape[-2:] != density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, size=density.shape[-2:], mode='bilinear', align_corners=False
        )
    mask = (pi_probs > threshold).float()
    return density * mask


class TTAWrapper:

    def __init__(
        self,
        model,
        scale_comp=None,
        use_flip: bool = True,
        scales: list = None,
        device: torch.device = None,
    ):
        self.model = model
        self.scale_comp = scale_comp
        self.use_flip = use_flip
        self.scales = scales or [1.0]
        self.device = device or torch.device('cuda')

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, default_down: int) -> dict:
        B, C, H, W = images.shape

        base_density = None
        base_pi = None

        all_counts = []
        all_pi_means = []

        for scale in self.scales:
            if scale != 1.0:
                new_H = int(H * scale)
                new_W = int(W * scale)
                new_H = (new_H // 32) * 32
                new_W = (new_W // 32) * 32
                if new_H < 128 or new_W < 128:
                    continue
                scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
            else:
                scaled_images = images

            density, pi, cell_area = self._forward_single(scaled_images, default_down)
            count = (density.sum(dim=[1, 2, 3]) / cell_area)
            all_counts.append(count)
            all_pi_means.append(pi.mean(dim=[1, 2, 3]))

            if scale == 1.0:
                base_density = density
                base_pi = pi
                base_cell_area = cell_area

            if self.use_flip:
                flipped_images = torch.flip(scaled_images, dims=[3])
                density_flip, pi_flip, cell_area_flip = self._forward_single(flipped_images, default_down)
                count_flip = (density_flip.sum(dim=[1, 2, 3]) / cell_area_flip)
                all_counts.append(count_flip)
                all_pi_means.append(pi_flip.mean(dim=[1, 2, 3]))

        avg_count = torch.stack(all_counts, dim=0).mean(dim=0)
        avg_pi_mean = torch.stack(all_pi_means, dim=0).mean(dim=0)

        if base_density is not None:
            original_count = base_density.sum(dim=[1, 2, 3]) / base_cell_area
            scale_factor = avg_count / (original_count + 1e-8)
            scaled_density = base_density * scale_factor.view(B, 1, 1, 1)
        else:
            scaled_density = base_density

        return {
            'p2r_density': scaled_density,
            'pi_probs': base_pi,
            'avg_count': avg_count,
            'avg_pi_mean': avg_pi_mean,
            'n_augments': len(all_counts),
            'cell_area': base_cell_area,
        }

    def _forward_single(self, images: torch.Tensor, default_down: int):
        outputs = self.model(images)
        density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']

        if self.scale_comp is not None:
            density = self.scale_comp(density)

        _, _, H_in, W_in = images.shape
        density, down_tuple, _ = canonicalize_p2r_grid(density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]

        return density, pi_probs, cell_area


@torch.no_grad()
def evaluate(
    model,
    scale_comp,
    dataloader,
    device,
    default_down,
    pi_threshold=0.5,
    soft_alpha=0.2,
    use_tta=False,
    tta_scales=None,
    max_size=2048,
    output_dir=None,
):
    model.eval()
    if scale_comp is not None:
        scale_comp.eval()

    if use_tta:
        tta_wrapper = TTAWrapper(
            model=model,
            scale_comp=scale_comp,
            use_flip=True,
            scales=tta_scales or [0.9, 1.0, 1.1],
            device=device,
        )
        print(f"TTA attivo: flip + scales={tta_scales or [0.9, 1.0, 1.1]}")

    results = {
        'raw': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0, 'preds': [], 'gts': []},
        'masked': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0, 'preds': [], 'gts': []},
        'soft': {'mae': [], 'mse': [], 'pred': 0, 'gt': 0, 'preds': [], 'gts': []},
    }

    density_bins = {
        'sparse': (0, 100),
        'medium': (100, 500),
        'dense': (500, float('inf')),
    }
    density_results = {
        mode: {bin_name: {'mae': [], 'count': 0} for bin_name in density_bins}
        for mode in ['raw', 'masked', 'soft']
    }

    total_cm = np.zeros((2, 2), dtype=np.int64)

    coverages = []
    pi_ratios = []
    scale_value = scale_comp.get_scale() if scale_comp else 1.0

    for images, densities, points in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        gt_density_map = densities.to(device)
        points_list = [p.to(device) for p in points]

        images, scale = resize_if_needed(images, max_size)

        if use_tta:
            outputs = tta_wrapper(images, default_down)
            raw_density = outputs['p2r_density']
            pi_probs = outputs['pi_probs']
            cell_area = outputs['cell_area']
        else:
            outputs = model(images)
            raw_density = outputs['p2r_density']
            pi_probs = outputs['pi_probs']

            _, _, H_in, W_in = images.shape
            raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
            cell_area = down_tuple[0] * down_tuple[1]

            if scale_comp is not None:
                raw_density = raw_density * scale_value

        if pi_probs.shape[-2:] != gt_density_map.shape[-2:]:
            pi_probs_resized = F.interpolate(
                pi_probs, size=gt_density_map.shape[-2:], mode='bilinear', align_corners=False
            )
        else:
            pi_probs_resized = pi_probs

        binary_gt = (gt_density_map > 1e-3).cpu().numpy().flatten()

        binary_pred = (pi_probs_resized >= 0.5).cpu().numpy().flatten()

        batch_cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
        total_cm += batch_cm

        coverage = (pi_probs_resized > pi_threshold).float().mean().item() * 100
        coverages.append(coverage)
        pi_ratios.append(pi_probs_resized.mean().item())

        masked_density = apply_hard_mask(raw_density, pi_probs, pi_threshold)
        soft_density = apply_soft_weighting(raw_density, pi_probs, soft_alpha)

        for i, pts in enumerate(points_list):
            gt = len(pts)

            pred_raw = (raw_density[i].sum() / cell_area).item()
            pred_masked = (masked_density[i].sum() / cell_area).item()
            pred_soft = (soft_density[i].sum() / cell_area).item()

            for mode, pred in [('raw', pred_raw), ('masked', pred_masked), ('soft', pred_soft)]:
                results[mode]['mae'].append(abs(pred - gt))
                results[mode]['mse'].append((pred - gt) ** 2)
                results[mode]['pred'] += pred
                results[mode]['gt'] += gt
                results[mode]['preds'].append(pred)
                results[mode]['gts'].append(gt)

                for bin_name, (lo, hi) in density_bins.items():
                    if lo <= gt < hi:
                        density_results[mode][bin_name]['mae'].append(abs(pred - gt))
                        density_results[mode][bin_name]['count'] += 1

    if output_dir:
        try:
            cm_normalized = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                        xticklabels=["Background", "Crowd"],
                        yticklabels=["Background", "Crowd"])
            plt.xlabel("Predicted Label (via ZIP)")
            plt.ylabel("True Label (via Density)")
            plt.title("ZIP Head Classification Performance (Pixel-level)")

            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nConfusion Matrix salvata in: {cm_path}")

            tn, fp, fn, tp = total_cm.ravel()
            print(f"   TN (Bkg correctly identified): {tn:,}")
            print(f"   FP (Bkg classified as Crowd):  {fp:,}")
            print(f"   FN (Crowd missed - dangerous): {fn:,}")
            print(f"   TP (Crowd correctly found):    {tp:,}")

        except Exception as e:
            print(f"Errore durante generazione plot CM: {e}")

    final_results = {}

    for mode in ['raw', 'masked', 'soft']:
        r = results[mode]
        preds = np.array(r['preds'])
        gts = np.array(r['gts'])

        final_results[mode] = {
            'mae': float(np.mean(r['mae'])),
            'rmse': float(np.sqrt(np.mean(r['mse']))),
            'mse': float(np.mean(r['mse'])),
            'bias': float(r['pred'] / r['gt']) if r['gt'] > 0 else 0,
            'nae': float(np.sum(r['mae']) / np.sum(gts)) if np.sum(gts) > 0 else 0,
            'gt_total': int(np.sum(gts)),
            'pred_total': float(np.sum(preds)),
            'gt_mean': float(np.mean(gts)),
            'pred_mean': float(np.mean(preds)),
            'gt_std': float(np.std(gts)),
            'pred_std': float(np.std(preds)),
        }

        final_results[mode]['by_density'] = {}
        for bin_name in density_bins:
            dr = density_results[mode][bin_name]
            if dr['mae']:
                final_results[mode]['by_density'][bin_name] = {
                    'mae': float(np.mean(dr['mae'])),
                    'count': dr['count'],
                }

    final_results['coverage'] = float(np.mean(coverages))
    final_results['pi_mean'] = float(np.mean(pi_ratios))
    final_results['scale'] = scale_value
    final_results['num_images'] = len(results['raw']['mae'])

    return final_results


def print_results(results, pi_threshold, soft_alpha, use_tta, split):

    print("=" * 70)
    print(f"STAGE 3 EVALUATION RESULTS - {split.upper()} SET {'(con TTA)' if use_tta else ''}")
    print("=" * 70)

    print(f"\nScale Compensation: {results['scale']:.4f}")

    print(f"\n{'Metrica':<12} {'RAW':<15} {'MASKED (τ={:.2f})':<20} {'SOFT (α={:.2f})':<15}".format(
        pi_threshold, soft_alpha))
    print("-" * 62)

    for metric in ['mae', 'rmse', 'bias', 'nae']:
        raw = results['raw'][metric]
        masked = results['masked'][metric]
        soft = results['soft'][metric]

        if metric in ['bias', 'nae']:
            print(f"{metric.upper():<12} {raw:<15.3f} {masked:<20.3f} {soft:<15.3f}")
        else:
            print(f"{metric.upper():<12} {raw:<15.2f} {masked:<20.2f} {soft:<15.2f}")

    print(f"\nStatistiche Conteggi (RAW):")
    print(f"   GT Total:   {results['raw']['gt_total']:,}")
    print(f"   Pred Total: {results['raw']['pred_total']:,.0f}")
    print(f"   GT Mean:    {results['raw']['gt_mean']:.1f} ± {results['raw']['gt_std']:.1f}")
    print(f"   Pred Mean:  {results['raw']['pred_mean']:.1f} ± {results['raw']['pred_std']:.1f}")

    print(f"\nCoverage medio (tau={pi_threshold}): {results['coverage']:.1f}%")
    print(f"   pi-head active ratio: {results['pi_mean']*100:.1f}%")

    print("\nMAE per densita:")
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

        if count > 0:
            print(f"   {density_labels[bin_name]}: RAW={raw_mae:.1f} | MASKED={masked_mae:.1f} | SOFT={soft_mae:.1f} [{count} imgs]")

    print(f"\nImages evaluated: {results['num_images']}")

    print("\n" + "=" * 70)
    print("RISULTATO FINALE")
    print("=" * 70)

    best_mode = min(['raw', 'masked', 'soft'], key=lambda m: results[m]['mae'])
    best_mae = results[best_mode]['mae']

    print(f"   MAE RAW:        {results['raw']['mae']:.2f}")
    print(f"   MAE MASKED:     {results['masked']['mae']:.2f}")
    print(f"   MAE SOFT:       {results['soft']['mae']:.2f}")
    print(f"   MIGLIORE:       {best_mode.upper()} (MAE={best_mae:.2f})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage 3')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pi-thresh', type=float, default=None)
    parser.add_argument('--soft-alpha', type=float, default=0.3)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--tta-scales', type=str, default='0.9,1.0,1.1')
    parser.add_argument('--no-scale-comp', action='store_true')
    parser.add_argument('--tta-flip-only', action='store_true')
    parser.add_argument('--max-size', type=int, default=2048)
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('DEVICE', 'cuda'))

    if args.tta_flip_only:
        tta_scales = [1.0]
    else:
        tta_scales = [float(s) for s in args.tta_scales.split(',')]

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

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', 16)

    pi_threshold = args.pi_thresh if args.pi_thresh is not None else config.get('MODEL', {}).get('ZIP_PI_THRESH', 0.5)
    soft_alpha = args.soft_alpha

    print("=" * 70)
    print(f"EVALUATE STAGE 3 ({args.split.upper()} SET)")
    print("=" * 70)
    print(f"   pi-threshold: {pi_threshold}")
    print(f"   soft-alpha: {soft_alpha}")
    print(f"   TTA: {'ON' if args.tta else 'OFF'}")
    if args.tta:
        if args.tta_flip_only:
            print(f"   TTA mode: FLIP ONLY")
        else:
            print(f"   TTA scales: {tta_scales}")
    print("=" * 70)

    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(dataset_name)

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
        transforms=val_tf
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    print(f"\nDataset: {dataset_name}")
    print(f"   Split: {split_name}")
    print(f"   Images: {len(dataset)}")

    bins_config = config.get('BINS_CONFIG', {})
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in DEFAULT_BINS_CONFIG:
        bin_config = DEFAULT_BINS_CONFIG[dataset_name]
    else:
        raise KeyError(f"BINS_CONFIG missing for dataset '{dataset_name}'")

    zip_head_cfg = config.get('ZIP_HEAD', {})
    model_cfg = config.get('MODEL', {})

    model = P2R_ZIP_Model(
        backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
        pi_thresh=pi_threshold,
        gate=model_cfg.get('GATE', 'multiply'),
        upsample_to_input=model_cfg.get('UPSAMPLE_TO_INPUT', False),
        bins=bin_config['bins'],
        bin_centers=bin_config['bin_centers'],
        zip_head_kwargs={
            'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
            'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
            'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
        },
    ).to(device)

    scale_comp = ScaleCompensation(init_scale=1.0, learnable=False).to(device)

    run_name = config.get('RUN_NAME', 'stage3')
    exp_cfg = config.get('EXP', {})
    output_dir = os.path.join(exp_cfg.get('OUT_DIR', 'exp'), run_name)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        search_order = [
            'stage3_fusion_best.pth',
            'stage3_fusion_last.pth',
            'stage3_curriculum_best.pth',
            'stage3_best.pth',
        ]
        ckpt_path = None
        for name in search_order:
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                ckpt_path = path
                break

        if ckpt_path is None:
            raise FileNotFoundError(f"Nessun checkpoint Stage 3 trovato in {output_dir}")

    print(f"\nCaricamento: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    if 'scale_comp' in state and not args.no_scale_comp:
        scale_comp.load_state_dict(state['scale_comp'])
        print(f"Scale Compensation caricato: {scale_comp.get_scale():.4f}")
    else:
        scale_comp = None
        print("Scale Compensation non trovato o disabilitato")

    if 'epoch' in state:
        print(f"   Epoch: {state['epoch']}")
    if 'best_mae' in state:
        print(f"   Best MAE (training): {state['best_mae']:.2f}")

    if args.output:
        cm_output_dir = os.path.dirname(args.output)
    else:
        cm_output_dir = output_dir
    os.makedirs(cm_output_dir, exist_ok=True)

    results = evaluate(
        model=model,
        scale_comp=scale_comp,
        dataloader=dataloader,
        device=device,
        default_down=default_down,
        pi_threshold=pi_threshold,
        soft_alpha=soft_alpha,
        use_tta=args.tta,
        tta_scales=tta_scales,
        max_size=args.max_size,
        output_dir=cm_output_dir
    )

    print_results(results, pi_threshold, soft_alpha, args.tta, args.split)

    if args.save_results or args.output:
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"stage3_eval_{args.split}_{timestamp}.json")

        results_json = {
            **results,
            'metadata': {
                'checkpoint': ckpt_path,
                'config': args.config,
                'split': args.split,
                'pi_threshold': pi_threshold,
                'soft_alpha': soft_alpha,
                'tta': args.tta,
                'tta_scales': tta_scales if args.tta else None,
                'timestamp': datetime.now().isoformat(),
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nRisultati salvati: {output_path}")


if __name__ == '__main__':
    main()
