#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 - BIAS CORRECTION END-TO-END FINE-TUNING

PROBLEMA:
- Stage 2 P2R ha MAE ~103 con bias 1.10 (sovrastima 10%)
- Stage 3 V5 (π-only) arriva a MAE ~99 ma con alta varianza
- Target: MAE 65-70

SOLUZIONE:
1. Sbloccare P2R head con LR MOLTO basso (1e-6)
2. Learning rate separato per log_scale (parametro chiave del bias)
3. Loss ASIMMETRICA: penalizza sovrastima più di sottostima
4. Gradient clipping aggressivo per stabilità
5. π-head continua a essere trainato

STRATEGIA:
- Backbone: FROZEN
- P2R head: LR 1e-6 (micro-adjustment)
- log_scale: LR 1e-5 (target principale della correzione)
- π-head: LR 1e-4 (come V5)
- Nuova loss: BiasAwareCountLoss che penalizza ratio > 1.0
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    canonicalize_p2r_grid, collate_fn
)


# =============================================================================
# CALIBRAZIONE LOG_SCALE (da Stage 2)
# =============================================================================

@torch.no_grad()
def calibrate_density_scale(
    model,
    loader,
    device,
    default_down,
    max_batches=10,
    clamp_range=(-2.0, 10.0),
    max_adjust=3.0,
    bias_eps=0.05,
    verbose=True,
):
    """
    Calibrazione della scala basata sul bias mediano.
    
    Questa funzione è CRITICA per allineare le metriche tra stage.
    Senza calibrazione, il log_scale salvato nel checkpoint potrebbe
    non essere ottimale, causando discrepanze di MAE.
    """
    if not hasattr(model, "p2r_head") or not hasattr(model.p2r_head, "log_scale"):
        if verbose:
            print("⚠️ Calibrazione saltata: log_scale non trovato")
        return None

    model.eval()
    ratios = []

    for batch_idx, (images, _, points) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)

        outputs = model(images)
        pred_density = outputs.get("p2r_density", outputs.get("density"))
        if pred_density is None:
            continue

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w

        for i, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            if gt == 0:
                continue
            
            pred = (pred_density[i].sum() / cell_area).item()
            if gt > 0:
                ratios.append(pred / gt)

    if len(ratios) == 0:
        if verbose:
            print("⚠️ Calibrazione saltata: nessun dato valido")
        return None

    ratios_np = np.array(ratios)
    bias_median = np.median(ratios_np)
    
    if verbose:
        print(f"\n📊 Statistiche Calibrazione:")
        print(f"   Bias mediano: {bias_median:.3f}")
        print(f"   Ratio range: [{ratios_np.min():.3f}, {ratios_np.max():.3f}]")

    if abs(bias_median - 1.0) < bias_eps:
        if verbose:
            print(f"✅ Calibrazione: bias già accettabile ({bias_median:.3f})")
        return bias_median

    prev_log_scale = float(model.p2r_head.log_scale.detach().item())
    raw_adjust = float(np.log(max(bias_median, 0.01)))
    
    if max_adjust is not None:
        adjust = float(np.clip(raw_adjust, -max_adjust, max_adjust))
    else:
        adjust = raw_adjust
    
    model.p2r_head.log_scale.data -= torch.tensor(adjust, device=device)
    
    if clamp_range is not None:
        min_val, max_val = float(clamp_range[0]), float(clamp_range[1])
        model.p2r_head.log_scale.data.clamp_(min_val, max_val)
    
    new_log_scale = float(model.p2r_head.log_scale.detach().item())
    new_scale = float(torch.exp(model.p2r_head.log_scale.detach()).item())
    
    if verbose:
        print(f"🔧 Calibrazione: bias={bias_median:.3f} → "
              f"log_scale {prev_log_scale:.2f}→{new_log_scale:.2f} (scala={new_scale:.2f})")
    
    return bias_median


# =============================================================================
# BIAS-AWARE LOSS FUNCTIONS
# =============================================================================

class BiasAwareCountLoss(nn.Module):
    """
    Loss che penalizza ASIMMETRICAMENTE sovrastima vs sottostima.
    
    Il bias sistematico del 10% indica che log_scale è troppo alto.
    Questa loss penalizza di più quando pred > gt.
    
    L_total = L_mae + α * L_overestimate_penalty
    
    dove:
    - L_mae = |pred - gt|
    - L_overestimate_penalty = max(0, (pred/gt) - 1)² quando pred > gt
    """
    def __init__(
        self,
        overestimate_weight: float = 2.0,  # Quanto penalizzare sovrastima
        target_bias: float = 1.0,          # Bias target (1.0 = perfetto)
        bias_tolerance: float = 0.02,      # Tolleranza (0.98-1.02 ok)
    ):
        super().__init__()
        self.overestimate_weight = overestimate_weight
        self.target_bias = target_bias
        self.bias_tolerance = bias_tolerance
    
    def forward(self, pred_count, gt_count):
        """
        Args:
            pred_count: [B] predicted counts
            gt_count: [B] ground truth counts
        """
        # MAE base
        mae_loss = torch.abs(pred_count - gt_count).mean()
        
        # Ratio pred/gt
        ratio = pred_count / (gt_count + 1e-6)
        
        # Penalizza sovrastima (ratio > 1)
        overestimate = F.relu(ratio - (self.target_bias + self.bias_tolerance))
        overestimate_penalty = (overestimate ** 2).mean()
        
        # Penalizza sottostima (ratio < 1) - ma meno
        underestimate = F.relu((self.target_bias - self.bias_tolerance) - ratio)
        underestimate_penalty = (underestimate ** 2).mean() * 0.5  # Metà peso
        
        total_loss = mae_loss + self.overestimate_weight * (overestimate_penalty + underestimate_penalty)
        
        # Metriche
        with torch.no_grad():
            mean_ratio = ratio.mean().item()
            overest_ratio = (ratio > 1.05).float().mean().item() * 100
            underest_ratio = (ratio < 0.95).float().mean().item() * 100
        
        return total_loss, {
            "mae": mae_loss.item(),
            "bias": mean_ratio,
            "overest_pct": overest_ratio,
            "underest_pct": underest_ratio,
        }


class PiHeadLoss(nn.Module):
    """Loss per π-head (come V5)."""
    def __init__(self, pos_weight: float = 8.0, block_size: int = 16, occupancy_threshold: float = 0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), reduction='mean')
    
    def compute_gt_occupancy(self, gt_density):
        gt_counts_per_block = F.avg_pool2d(gt_density, kernel_size=self.block_size, stride=self.block_size) * (self.block_size ** 2)
        return (gt_counts_per_block > self.occupancy_threshold).float()
    
    def forward(self, logit_pi_maps, gt_density):
        logit_pieno = logit_pi_maps[:, 1:2, :, :]
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest')
        
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        
        loss = self.bce(logit_pieno, gt_occupancy)
        
        with torch.no_grad():
            pred_prob = torch.sigmoid(logit_pieno)
            pred_occupancy = (pred_prob > 0.5).float()
            coverage = pred_occupancy.mean().item() * 100
            recall = 100.0
            if gt_occupancy.sum() > 0:
                tp = (pred_occupancy * gt_occupancy).sum()
                fn = ((1 - pred_occupancy) * gt_occupancy).sum()
                recall = (tp / (tp + fn + 1e-6)).item() * 100
        
        return loss, {"coverage": coverage, "recall": recall}


# =============================================================================
# COUNT FUNCTIONS
# =============================================================================

def compute_count_raw(pred_density, down_h, down_w):
    """Conteggio RAW (senza maschera)."""
    cell_area = down_h * down_w
    return torch.sum(pred_density, dim=(1, 2, 3)) / cell_area


def compute_count_masked(pred_density, pi_probs, down_h, down_w, threshold=0.5):
    """Conteggio MASKED (con maschera π)."""
    cell_area = down_h * down_w
    if pi_probs.shape[-2:] != pred_density.shape[-2:]:
        pi_probs = F.interpolate(pi_probs, size=pred_density.shape[-2:], mode='bilinear', align_corners=False)
    mask = (pi_probs > threshold).float()
    masked_density = pred_density * mask
    return torch.sum(masked_density, dim=(1, 2, 3)) / cell_area


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_mae": best_mae,
        "best_results": best_results,
    }
    
    latest_path = os.path.join(output_dir, "stage4_latest.pth")
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(output_dir, "stage4_best.pth")
        torch.save(checkpoint, best_path)
        print(f"💾 Saved: stage4_latest.pth + stage4_best.pth (MAE={best_mae:.2f})")
    else:
        print(f"💾 Saved: stage4_latest.pth (epoch {epoch})")


def resume_checkpoint(model, optimizer, scheduler, output_dir, device):
    latest_path = os.path.join(output_dir, "stage4_latest.pth")
    if not os.path.isfile(latest_path):
        return 1, float('inf'), {}
    
    print(f"🔄 Resume da: {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("best_mae", float('inf')), checkpoint.get("best_results", {})


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, count_loss_fn, pi_loss_fn, dataloader, optimizer, device, default_down, epoch, config):
    """Training con bias correction."""
    model.train()
    
    # Backbone in eval (BatchNorm frozen)
    model.backbone.eval()
    
    total_count_loss = 0.0
    total_pi_loss = 0.0
    total_bias = 0.0
    total_coverage = 0.0
    num_batches = 0
    
    pi_weight = config.get("PI_LOSS_WEIGHT", 0.3)
    count_weight = config.get("COUNT_LOSS_WEIGHT", 1.0)
    
    progress_bar = tqdm(dataloader, desc=f"Stage4 [Ep {epoch}]")
    
    for images, gt_density, points in progress_bar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # P2R density
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(pred_density, (h_in, w_in), default_down)
        down_h, down_w = down_tuple
        
        # Count RAW (senza masking per count loss)
        pred_count = compute_count_raw(pred_density, down_h, down_w)
        
        # GT count
        gt_count = torch.tensor([len(pts) if pts is not None else 0 for pts in points], 
                                device=device, dtype=torch.float32)
        
        # Count loss con bias penalty
        count_loss, count_metrics = count_loss_fn(pred_count, gt_count)
        
        # Pi loss
        logit_pi = outputs["logit_pi_maps"]
        pi_loss, pi_metrics = pi_loss_fn(logit_pi, gt_density)
        
        # Total loss
        total_loss = count_weight * count_loss + pi_weight * pi_loss
        
        total_loss.backward()
        
        # Gradient clipping AGGRESSIVO per stabilità
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_count_loss += count_loss.item()
        total_pi_loss += pi_loss.item()
        total_bias += count_metrics["bias"]
        total_coverage += pi_metrics["coverage"]
        num_batches += 1
        
        progress_bar.set_postfix({
            "CL": f"{count_loss.item():.3f}",
            "bias": f"{count_metrics['bias']:.3f}",
            "cov": f"{pi_metrics['coverage']:.1f}%",
        })
    
    n = max(num_batches, 1)
    print(f"   Train: CountLoss={total_count_loss/n:.4f}, PiLoss={total_pi_loss/n:.4f}, "
          f"Bias={total_bias/n:.3f}, Coverage={total_coverage/n:.1f}%")
    
    return total_count_loss / n


@torch.no_grad()
def validate(model, dataloader, device, default_down, pi_threshold=0.5):
    """Validazione completa."""
    model.eval()
    
    mae_raw, mae_masked = 0.0, 0.0
    mse_raw = 0.0
    total_pred_raw, total_gt = 0.0, 0.0
    total_coverage = 0.0
    n_samples = 0
    
    errors_by_density = {"sparse": [], "medium": [], "dense": []}
    
    for images, gt_density, points in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(pred_density, (h_in, w_in), default_down)
        down_h, down_w = down_tuple
        
        pred_count_raw = compute_count_raw(pred_density, down_h, down_w)
        
        pi_probs = outputs.get("pi_probs")
        if pi_probs is None:
            logit_pi = outputs["logit_pi_maps"]
            pi_probs = logit_pi.softmax(dim=1)[:, 1:2]
        
        pred_count_masked = compute_count_masked(pred_density, pi_probs, down_h, down_w, threshold=pi_threshold)
        
        # Coverage
        pred_occupancy = (pi_probs > 0.5).float()
        coverage = pred_occupancy.mean().item() * 100
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_r = pred_count_raw[idx].item()
            pred_m = pred_count_masked[idx].item()
            
            err_r = abs(pred_r - gt)
            err_m = abs(pred_m - gt)
            
            mae_raw += err_r
            mae_masked += err_m
            mse_raw += err_r ** 2
            total_pred_raw += pred_r
            total_gt += gt
            total_coverage += coverage
            n_samples += 1
            
            if gt <= 100:
                errors_by_density["sparse"].append((err_r, err_m, pred_r, gt))
            elif gt <= 500:
                errors_by_density["medium"].append((err_r, err_m, pred_r, gt))
            else:
                errors_by_density["dense"].append((err_r, err_m, pred_r, gt))
    
    mae_raw /= n_samples
    mae_masked /= n_samples
    rmse_raw = np.sqrt(mse_raw / n_samples)
    bias_raw = total_pred_raw / total_gt if total_gt > 0 else 0
    coverage = total_coverage / n_samples
    
    # Report
    print(f"\n{'='*60}")
    print(f"📊 Stage 4 Validation Results")
    print(f"{'='*60}")
    print(f"   MAE RAW:    {mae_raw:.2f}  ← METRICA PRINCIPALE")
    print(f"   MAE MASKED: {mae_masked:.2f}")
    print(f"   RMSE:       {rmse_raw:.2f}")
    print(f"   Bias:       {bias_raw:.3f} {'✅' if 0.98 <= bias_raw <= 1.02 else '⚠️'}")
    print(f"{'─'*60}")
    print(f"   π coverage: {coverage:.1f}%")
    print(f"{'─'*60}")
    
    print(f"   Per densità:")
    for name, errors in errors_by_density.items():
        if errors:
            raw_errs = [e[0] for e in errors]
            biases = [e[2]/(e[3]+1e-6) for e in errors]
            print(f"      {name}: MAE={np.mean(raw_errs):.1f}, bias={np.mean(biases):.3f} ({len(errors)} imgs)")
    
    # Target check
    target_low, target_high = 65, 70
    if mae_raw <= target_high:
        print(f"\n   🎯 TARGET RAGGIUNTO! MAE {mae_raw:.2f} ≤ {target_high}")
    else:
        gap = mae_raw - target_high
        print(f"\n   📉 Gap da target: {gap:.1f} punti (MAE {mae_raw:.2f} vs target {target_high})")
    
    print(f"{'='*60}\n")
    
    return {
        "mae": mae_raw,
        "mae_raw": mae_raw,
        "mae_masked": mae_masked,
        "rmse": rmse_raw,
        "bias": bias_raw,
        "coverage": coverage,
    }


# =============================================================================
# PARAMETER GROUPS WITH DIFFERENT LR
# =============================================================================

def get_parameter_groups(model, config):
    """
    Crea gruppi di parametri con learning rate diversi.
    
    Strategia:
    - Backbone: FROZEN (lr=0)
    - P2R head (escluso log_scale): lr MOLTO basso (1e-6)
    - P2R log_scale: lr medio (1e-5) - TARGET PRINCIPALE
    - ZIP π-head: lr normale (1e-4)
    """
    lr_p2r = config.get("LR_P2R", 1e-6)
    lr_log_scale = config.get("LR_LOG_SCALE", 1e-5)
    lr_pi = config.get("LR_PI", 1e-4)
    
    # Congela backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Identifica parametri
    p2r_params = []
    log_scale_params = []
    pi_params = []
    
    for name, param in model.p2r_head.named_parameters():
        if "log_scale" in name:
            log_scale_params.append(param)
            param.requires_grad = True
            print(f"   🎯 log_scale: {name} (LR={lr_log_scale})")
        else:
            p2r_params.append(param)
            param.requires_grad = True
    
    for name, param in model.zip_head.named_parameters():
        if "bin_head" in name:
            param.requires_grad = False  # bin_head congelata
        else:
            pi_params.append(param)
            param.requires_grad = True
    
    param_groups = [
        {"params": p2r_params, "lr": lr_p2r, "name": "p2r_head"},
        {"params": log_scale_params, "lr": lr_log_scale, "name": "log_scale"},
        {"params": pi_params, "lr": lr_pi, "name": "pi_head"},
    ]
    
    # Report
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 Parameter Groups:")
    print(f"   P2R head:   {sum(p.numel() for p in p2r_params):,} params @ LR={lr_p2r}")
    print(f"   log_scale:  {sum(p.numel() for p in log_scale_params):,} params @ LR={lr_log_scale}")
    print(f"   π-head:     {sum(p.numel() for p in pi_params):,} params @ LR={lr_pi}")
    print(f"   Trainable:  {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.1f}%)")
    
    return param_groups


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    
    print("="*60)
    print("🚀 Stage 4 - BIAS CORRECTION END-TO-END")
    print("="*60)
    print(f"Device: {device}")
    print("Strategia: P2R unfrozen con LR basso + Bias Penalty Loss")
    print("Target: MAE 65-70, Bias ~1.0")
    print("="*60)
    
    # Config
    data_cfg = config["DATA"]
    stage4_cfg = config.get("STAGE4_LOSS", {})
    optim_cfg = config.get("OPTIM_STAGE4", {})
    
    # Override con valori ottimizzati per bias correction
    lr_p2r = float(optim_cfg.get("LR_P2R", 1e-6))
    lr_log_scale = float(optim_cfg.get("LR_LOG_SCALE", 1e-5))
    lr_pi = float(optim_cfg.get("LR_PI", 1e-4))
    overestimate_weight = float(stage4_cfg.get("OVERESTIMATE_WEIGHT", 2.0))
    
    print(f"\n⚙️ Config:")
    print(f"   LR_P2R: {lr_p2r}")
    print(f"   LR_LOG_SCALE: {lr_log_scale}")
    print(f"   LR_PI: {lr_pi}")
    print(f"   OVERESTIMATE_WEIGHT: {overestimate_weight}")
    
    # Dataset
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_transforms
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg.get("BATCH_SIZE", 4),  # Batch più piccolo per stabilità
        shuffle=True,
        num_workers=optim_cfg.get("NUM_WORKERS", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg.get("NUM_WORKERS", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": config["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": config["ZIP_HEAD"].get("USE_SOFTPLUS", True),
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,
        use_ste_mask=True,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # Output Directory
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================
    # CARICA BEST CHECKPOINT DISPONIBILE
    # =========================================
    # Priorità: stage3_v5_best > stage3_best > stage2_best
    checkpoint_priority = [
        "stage3_v5_best.pth",
        "stage3_best.pth", 
        "stage2_best.pth"
    ]
    
    loaded = False
    for ckpt_name in checkpoint_priority:
        ckpt_path = os.path.join(output_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            print(f"\n✅ Caricamento: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            if "model" in state:
                state = state["model"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            loaded = True
            break
    
    if not loaded:
        print(f"❌ Nessun checkpoint trovato in {output_dir}")
        return
    
    # Definisci default_down prima della calibrazione
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    # =========================================
    # CALIBRAZIONE LOG_SCALE (CRITICO!)
    # =========================================
    # Senza calibrazione, il log_scale del checkpoint potrebbe
    # non essere ottimale, causando discrepanze di MAE
    print("\n🔧 Calibrazione log_scale...")
    
    # Crea un loader temporaneo per calibrazione
    calib_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg.get("NUM_WORKERS", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    calibrate_density_scale(
        model, calib_loader, device, default_down,
        max_batches=10, verbose=True
    )
    
    # =========================================
    # SETUP PARAMETRI CON LR DIFFERENZIATI
    # =========================================
    print("\n🔧 Setup parametri:")
    
    param_config = {
        "LR_P2R": lr_p2r,
        "LR_LOG_SCALE": lr_log_scale,
        "LR_PI": lr_pi,
    }
    param_groups = get_parameter_groups(model, param_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(optim_cfg.get("WEIGHT_DECAY", 1e-4))
    )
    
    epochs = optim_cfg.get("EPOCHS", 150)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    
    # Loss functions
    count_loss_fn = BiasAwareCountLoss(
        overestimate_weight=overestimate_weight,
        target_bias=1.0,
        bias_tolerance=0.02,
    )
    
    pi_loss_fn = PiHeadLoss(
        pos_weight=float(config.get("JOINT_LOSS", {}).get("PI_POS_WEIGHT", 8.0)),
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        occupancy_threshold=0.5,
    ).to(device)
    
    loss_config = {
        "PI_LOSS_WEIGHT": float(stage4_cfg.get("PI_WEIGHT", 0.3)),
        "COUNT_LOSS_WEIGHT": float(stage4_cfg.get("COUNT_WEIGHT", 1.0)),
    }
    
    # Valutazione iniziale (post-calibrazione)
    print("\n📋 Valutazione iniziale:")
    init_results = validate(model, val_loader, device, default_down)
    
    best_mae = init_results["mae"]
    best_results = init_results
    initial_bias = init_results["bias"]
    
    print(f"\n🎯 Baseline:")
    print(f"   MAE iniziale: {best_mae:.2f}")
    print(f"   Bias iniziale: {initial_bias:.3f}")
    print(f"   Target: MAE 65-70, Bias ~1.0")
    
    # Training
    print(f"\n🚀 START Training Stage 4")
    print(f"   Epochs: 1 → {epochs}")
    
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 50)
    no_improve = 0
    val_interval = optim_cfg.get("VAL_INTERVAL", 3)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        # Log current LRs
        lrs = [f"{pg['name']}:{pg['lr']:.2e}" for pg in optimizer.param_groups]
        print(f"   LRs: {', '.join(lrs)}")
        
        train_loss = train_one_epoch(
            model, count_loss_fn, pi_loss_fn, train_loader, 
            optimizer, device, default_down, epoch, loss_config
        )
        
        scheduler.step()
        
        # Validazione
        if epoch % val_interval == 0:
            results = validate(model, val_loader, device, default_down)
            
            current_mae = results["mae"]
            current_bias = results["bias"]
            
            # È best se MAE migliora E bias si avvicina a 1.0
            bias_improved = abs(current_bias - 1.0) < abs(best_results.get("bias", 2.0) - 1.0)
            mae_improved = current_mae < best_mae - 0.3
            
            is_better = mae_improved or (current_mae < best_mae and bias_improved)
            
            if is_better:
                best_mae = current_mae
                best_results = results
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=True
                )
                print(f"🏆 NEW BEST: MAE={best_mae:.2f}, Bias={current_bias:.3f}")
                no_improve = 0
            else:
                no_improve += 1
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False
                )
                print(f"   No improvement ({no_improve}/{patience})")
            
            # Check target
            if current_mae <= 70 and 0.95 <= current_bias <= 1.05:
                print(f"\n🎯🎯🎯 TARGET RAGGIUNTO! MAE={current_mae:.2f}, Bias={current_bias:.3f}")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, current_mae, results, output_dir, is_best=True
                )
                break
            
            # Early stopping
            if no_improve >= patience:
                print(f"⛔ Early stopping a epoch {epoch}")
                break
    
    # Risultati finali
    print("\n" + "="*60)
    print("🏁 STAGE 4 COMPLETATO")
    print("="*60)
    print(f"   MAE iniziale:  {init_results['mae']:.2f}")
    print(f"   Bias iniziale: {initial_bias:.3f}")
    print(f"   ──────────────────────")
    print(f"   Best MAE:      {best_results.get('mae', best_mae):.2f}")
    print(f"   Best Bias:     {best_results.get('bias', 0):.3f}")
    print(f"   Best RMSE:     {best_results.get('rmse', 0):.2f}")
    
    gap = best_results.get('mae', best_mae) - 70
    if gap <= 0:
        print(f"\n   🎯 TARGET RAGGIUNTO!")
    else:
        print(f"\n   📉 Gap residuo: {gap:.1f} punti")
    
    print(f"\n💾 Best model: {os.path.join(output_dir, 'stage4_best.pth')}")


if __name__ == "__main__":
    main()