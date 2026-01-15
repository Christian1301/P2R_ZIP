#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Stage 3 - Con Scale Compensation e TTA

Supporta:
1. Scale Compensation dal checkpoint Stage 3
2. TTA: flip orizzontale + multi-scale
3. Modalità: RAW, MASKED, SOFT

Usage:
    python evaluate_stage3_tta.py --config config.yaml --checkpoint stage3_fusion_best.pth
    python evaluate_stage3_tta.py --config config.yaml --tta --tta-scales 0.9,1.0,1.1
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

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn, canonicalize_p2r_grid


# =============================================================================
# SCALE COMPENSATION (deve matchare quello in train_stage3_fusion.py)
# =============================================================================

class ScaleCompensation(nn.Module):
    """Modulo che compensa la riduzione del soft weighting."""
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


# =============================================================================
# SOFT WEIGHTING FUNCTIONS
# =============================================================================

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


# =============================================================================
# TTA (Test-Time Augmentation)
# =============================================================================

class TTAWrapper:
    """
    Test-Time Augmentation per crowd counting.
    
    Strategia: calcola COUNT per ogni augmentation, poi media i count.
    Non media le density map direttamente (problematico con scale diverse).
    
    Augmentations:
    - Flip orizzontale
    - Multi-scale (opzionale, spesso non aiuta molto)
    """
    
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
        """
        Forward con TTA.
        
        Returns:
            dict con 'p2r_density', 'pi_probs' e count mediato
        """
        B, C, H, W = images.shape
        
        # Per scale=1.0, teniamo la density originale per il soft weighting
        base_density = None
        base_pi = None
        
        all_counts = []
        all_pi_means = []
        
        for scale in self.scales:
            # Resize se scale != 1.0
            if scale != 1.0:
                new_H = int(H * scale)
                new_W = int(W * scale)
                # Arrotonda a multiplo di 32
                new_H = (new_H // 32) * 32
                new_W = (new_W // 32) * 32
                if new_H < 128 or new_W < 128:
                    continue
                scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
            else:
                scaled_images = images
            
            # Forward normale
            density, pi, cell_area = self._forward_single(scaled_images, default_down)
            count = (density.sum(dim=[1, 2, 3]) / cell_area)
            all_counts.append(count)
            all_pi_means.append(pi.mean(dim=[1, 2, 3]))
            
            # Salva base per scale=1.0
            if scale == 1.0:
                base_density = density
                base_pi = pi
                base_cell_area = cell_area
            
            # Flip orizzontale
            if self.use_flip:
                flipped_images = torch.flip(scaled_images, dims=[3])
                density_flip, pi_flip, cell_area_flip = self._forward_single(flipped_images, default_down)
                count_flip = (density_flip.sum(dim=[1, 2, 3]) / cell_area_flip)
                all_counts.append(count_flip)
                all_pi_means.append(pi_flip.mean(dim=[1, 2, 3]))
        
        # Media dei count
        avg_count = torch.stack(all_counts, dim=0).mean(dim=0)  # [B]
        avg_pi_mean = torch.stack(all_pi_means, dim=0).mean(dim=0)  # [B]
        
        # Per soft weighting, usiamo la density di scale=1.0 ma la scalamo
        # per matchare il count medio
        if base_density is not None:
            original_count = base_density.sum(dim=[1, 2, 3]) / base_cell_area
            # Scala la density per ottenere il count medio
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
        """Forward singolo, ritorna density, pi, e cell_area."""
        outputs = self.model(images)
        density = outputs['p2r_density']
        pi_probs = outputs['pi_probs']
        
        # Applica scale compensation se presente
        if self.scale_comp is not None:
            density = self.scale_comp(density)
        
        # Canonicalize
        _, _, H_in, W_in = images.shape
        density, down_tuple, _ = canonicalize_p2r_grid(density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]
        
        return density, pi_probs, cell_area


# =============================================================================
# EVALUATION
# =============================================================================

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
):
    """
    Valuta con tre modalità: RAW, MASKED (hard), SOFT.
    Opzionalmente usa TTA.
    """
    model.eval()
    if scale_comp is not None:
        scale_comp.eval()
    
    # Setup TTA
    if use_tta:
        tta_wrapper = TTAWrapper(
            model=model,
            scale_comp=scale_comp,
            use_flip=True,
            scales=tta_scales or [0.9, 1.0, 1.1],
            device=device,
        )
        print(f"🔄 TTA attivo: flip + scales={tta_scales or [0.9, 1.0, 1.1]}")
    
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
    scale_value = scale_comp.get_scale() if scale_comp else 1.0
    
    for images, densities, points in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        if use_tta:
            # TTA forward
            outputs = tta_wrapper(images, default_down)
            raw_density = outputs['p2r_density']
            pi_probs = outputs['pi_probs']
            cell_area = outputs['cell_area']
        else:
            # Normal forward
            outputs = model(images)
            raw_density = outputs['p2r_density']
            pi_probs = outputs['pi_probs']
            
            _, _, H_in, W_in = images.shape
            raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
            cell_area = down_tuple[0] * down_tuple[1]
            
            # Applica scale compensation
            if scale_comp is not None:
                raw_density = raw_density * scale_value
        
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
    final_results['scale'] = scale_value
    
    return final_results


def print_results(results, pi_threshold, soft_alpha, use_tta):
    """Stampa risultati in formato tabellare."""
    
    print("=" * 70)
    print(f"📊 RISULTATI VALUTAZIONE {'(con TTA)' if use_tta else ''}")
    print("=" * 70)
    
    print(f"\n📏 Scale Compensation: {results['scale']:.4f}")
    
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
        
        if count > 0:
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage 3 con TTA')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path checkpoint (default: cerca automaticamente)')
    parser.add_argument('--pi-thresh', type=float, default=None)
    parser.add_argument('--soft-alpha', type=float, default=0.3,
                        help='Alpha per soft weighting (default: 0.3)')
    parser.add_argument('--tta', action='store_true',
                        help='Abilita Test-Time Augmentation')
    parser.add_argument('--tta-scales', type=str, default='0.9,1.0,1.1',
                        help='Scale per TTA (default: 0.9,1.0,1.1)')
    parser.add_argument('--no-scale-comp', action='store_true',
                        help='Ignora scale compensation anche se presente')
    parser.add_argument('--tta-flip-only', action='store_true',
                        help='TTA con solo flip (senza multi-scale, più stabile)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('DEVICE', 'cuda'))
    
    # Parse TTA scales
    if args.tta_flip_only:
        tta_scales = [1.0]  # Solo scale 1.0 = solo flip
    else:
        tta_scales = [float(s) for s in args.tta_scales.split(',')]
    
    # Gestione robusta config DATA/DATASET
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

    # Normalizza nome dataset
    alias_map = {
        'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
        'shhb': 'shhb', 'shanghaitechpartb': 'shhb',
        'ucf': 'ucf', 'ucfqnrf': 'ucf',
        'nwpu': 'nwpu', 'jhu': 'jhu'
    }
    normalized_name = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = alias_map.get(normalized_name, str(dataset_name_raw).lower())

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    default_down = data_cfg.get('P2R_DOWNSAMPLE', 8)
    block_size = data_cfg.get('ZIP_BLOCK_SIZE', 16)
    
    pi_threshold = args.pi_thresh if args.pi_thresh is not None else config['MODEL'].get('ZIP_PI_THRESH', 0.5)
    soft_alpha = args.soft_alpha
    
    print("=" * 70)
    print("🔬 EVALUATE STAGE 3 CON SCALE COMPENSATION + TTA")
    print("=" * 70)
    print(f"   π-threshold: {pi_threshold}")
    print(f"   soft-alpha: {soft_alpha}")
    print(f"   TTA: {'ON' if args.tta else 'OFF'}")
    if args.tta:
        if args.tta_flip_only:
            print(f"   TTA mode: FLIP ONLY (più stabile)")
        else:
            print(f"   TTA scales: {tta_scales}")
    print("=" * 70)
    
    # Dataset
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(dataset_name)
    val_ds = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('VAL_SPLIT', 'val'),
        block_size=block_size,
        transforms=val_tf
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    print(f"📊 Dataset: {dataset_name}, {len(val_ds)} immagini")
    
    # Model
    bins_config = config.get('BINS_CONFIG', {})
    default_bins = {
        'shha': {
            'bins': [[0, 0], [1, 3], [4, 6], [7, 10], [11, 15], [16, 22], [23, 32], [33, 9999]],
            'bin_centers': [0.0, 2.0, 5.0, 8.5, 13.0, 19.0, 27.5, 45.0],
        },
    }
    
    if dataset_name in bins_config:
        bin_config = bins_config[dataset_name]
    elif dataset_name in default_bins:
        bin_config = default_bins[dataset_name]
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
    
    # Scale Compensation module
    scale_comp = ScaleCompensation(init_scale=1.0, learnable=False).to(device)
    
    # Load checkpoint
    run_name = config.get('RUN_NAME', 'shha_v15')
    output_dir = os.path.join(config.get('EXP', {}).get('OUT_DIR', 'exp'), run_name)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Cerca automaticamente - priorità Stage 3 fusion
        search_order = [
            'stage3_fusion_best.pth',
            'stage3_fusion_last.pth', 
            'stage3_curriculum_best.pth',
            'stage3_best.pth',
            'stage2_bypass_best.pth',
            'stage2_best.pth',
            'best_model.pth',
        ]
        ckpt_path = None
        for name in search_order:
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                ckpt_path = path
                break
        
        if ckpt_path is None:
            raise FileNotFoundError(f"Nessun checkpoint trovato in {output_dir}")
    
    print(f"\n🔄 Caricamento: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Carica model weights
    if 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    
    # Carica scale_comp se presente
    if 'scale_comp' in state and not args.no_scale_comp:
        scale_comp.load_state_dict(state['scale_comp'])
        print(f"✅ Scale Compensation caricato: {scale_comp.get_scale():.4f}")
    else:
        scale_comp = None
        print("ℹ️ Scale Compensation non trovato o disabilitato")
    
    # Info dal checkpoint
    if 'epoch' in state:
        print(f"   Epoch: {state['epoch']}")
    if 'best_mae' in state:
        print(f"   Best MAE (training): {state['best_mae']:.2f}")
    if 'alpha' in state:
        print(f"   Alpha (training): {state['alpha']:.3f}")
    
    # Valutazione
    print(f"\n{'='*70}")
    results = evaluate(
        model=model,
        scale_comp=scale_comp,
        dataloader=val_loader,
        device=device,
        default_down=default_down,
        pi_threshold=pi_threshold,
        soft_alpha=soft_alpha,
        use_tta=args.tta,
        tta_scales=tta_scales,
    )
    
    # Stampa risultati
    print_results(results, pi_threshold, soft_alpha, args.tta)
    
    # Confronto con/senza TTA
    if args.tta:
        print("\n🔄 Confronto senza TTA per riferimento:")
        results_no_tta = evaluate(
            model=model,
            scale_comp=scale_comp,
            dataloader=val_loader,
            device=device,
            default_down=default_down,
            pi_threshold=pi_threshold,
            soft_alpha=soft_alpha,
            use_tta=False,
            tta_scales=None,
        )
        
        print(f"\n   {'Modalità':<12} {'Senza TTA':<15} {'Con TTA':<15} {'Δ MAE':<10}")
        print("   " + "-" * 52)
        for mode in ['raw', 'masked', 'soft']:
            mae_no_tta = results_no_tta[mode]['mae']
            mae_tta = results[mode]['mae']
            delta = mae_no_tta - mae_tta
            print(f"   {mode.upper():<12} {mae_no_tta:<15.2f} {mae_tta:<15.2f} {delta:+.2f}")
        
        print(f"\n   {'Modalità':<12} {'RMSE no TTA':<15} {'RMSE TTA':<15} {'Δ RMSE':<10}")
        print("   " + "-" * 52)
        for mode in ['raw', 'masked', 'soft']:
            rmse_no_tta = results_no_tta[mode]['rmse']
            rmse_tta = results[mode]['rmse']
            delta = rmse_no_tta - rmse_tta
            print(f"   {mode.upper():<12} {rmse_no_tta:<15.2f} {rmse_tta:<15.2f} {delta:+.2f}")


if __name__ == '__main__':
    main()