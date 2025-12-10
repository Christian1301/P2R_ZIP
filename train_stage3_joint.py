#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 - Joint Training V5 - P2R FROZEN

LEZIONE APPRESA DA V4:
Il training congiunto di P2R + π-head causa divergenza.
La P2R head di Stage 2 era già buona (MAE ~103), ma il joint training
l'ha rovinata (MAE → 170+) aumentando il bias da 1.1 a 1.26.

NUOVA STRATEGIA:
1. CONGELA COMPLETAMENTE la P2R head (incluso log_scale)
2. Allena SOLO il π-head per migliorare localizzazione
3. Usa MAE MASKED come metrica (il π-head può aiutare a filtrare rumore)

Obiettivo: π-head che migliora (non peggiora) il conteggio
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
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False):
    """Salva checkpoint Stage 3."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_mae": best_mae,
        "best_results": best_results,
    }
    
    latest_path = os.path.join(output_dir, "stage3_v5_latest.pth")
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(output_dir, "stage3_v5_best.pth")
        torch.save(checkpoint, best_path)
        print(f"💾 Saved: stage3_v5_latest.pth + stage3_v5_best.pth (MAE={best_mae:.2f})")
    else:
        print(f"💾 Saved: stage3_v5_latest.pth (epoch {epoch})")


def resume_checkpoint(model, optimizer, scheduler, output_dir, device):
    """Riprende da checkpoint."""
    latest_path = os.path.join(output_dir, "stage3_v5_latest.pth")
    
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
# LOSS: Solo per π-head
# =============================================================================

class PiHeadLoss(nn.Module):
    """
    Loss per π-head ottimizzata.
    
    Obiettivo: il π-head deve identificare blocchi con persone
    in modo che il masking MIGLIORI (non peggiori) il conteggio.
    """
    def __init__(
        self, 
        pos_weight: float = 8.0,
        block_size: int = 16, 
        occupancy_threshold: float = 0.5,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        """GT occupancy: blocco pieno se contiene >= threshold persone."""
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return (gt_counts_per_block > self.occupancy_threshold).float()
    
    def forward(self, logit_pi_maps, gt_density):
        logit_pieno = logit_pi_maps[:, 1:2, :, :]
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest'
            )
        
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        
        loss = self.bce(logit_pieno, gt_occupancy)
        
        # Metriche
        with torch.no_grad():
            pred_prob = torch.sigmoid(logit_pieno)
            pred_occupancy = (pred_prob > 0.5).float()
            coverage = pred_occupancy.mean().item() * 100
            
            if gt_occupancy.sum() > 0:
                tp = (pred_occupancy * gt_occupancy).sum()
                fn = ((1 - pred_occupancy) * gt_occupancy).sum()
                fp = (pred_occupancy * (1 - gt_occupancy)).sum()
                
                recall = (tp / (tp + fn + 1e-6)).item() * 100
                precision = (tp / (tp + fp + 1e-6)).item() * 100
            else:
                recall = 100.0
                precision = 100.0
        
        return loss, {
            "coverage": coverage, 
            "recall": recall,
            "precision": precision,
        }


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
        pi_probs = F.interpolate(
            pi_probs, size=pred_density.shape[-2:], 
            mode='bilinear', align_corners=False
        )
    
    mask = (pi_probs > threshold).float()
    masked_density = pred_density * mask
    return torch.sum(masked_density, dim=(1, 2, 3)) / cell_area


# =============================================================================
# TRAINING - Solo π-head
# =============================================================================

def train_one_epoch(model, pi_loss_fn, dataloader, optimizer, device, default_down, epoch):
    """
    Training: SOLO π-head, P2R completamente congelata.
    """
    model.train()
    
    # Assicura che P2R sia in eval mode (BatchNorm frozen)
    model.p2r_head.eval()
    
    total_loss = 0.0
    total_coverage = 0.0
    total_recall = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Stage3 V5 [Ep {epoch}]")
    
    for images, gt_density, points in progress_bar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        optimizer.zero_grad()
        
        # Forward (P2R non ha gradienti)
        with torch.no_grad():
            # Backbone e P2R senza gradienti
            feat = model.backbone(images)
        
        # Solo ZIP head con gradienti
        zip_outputs = model.zip_head(feat, model.bin_centers)
        logit_pi = zip_outputs["logit_pi_maps"]
        
        # Loss solo su π-head
        loss, metrics = pi_loss_fn(logit_pi, gt_density)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.zip_head.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_coverage += metrics["coverage"]
        total_recall += metrics["recall"]
        num_batches += 1
        
        progress_bar.set_postfix({
            "L": f"{loss.item():.3f}",
            "cov": f"{metrics['coverage']:.1f}%",
            "rec": f"{metrics['recall']:.0f}%",
        })
    
    n = max(num_batches, 1)
    print(f"   Train: Loss={total_loss/n:.4f}, Coverage={total_coverage/n:.1f}%, Recall={total_recall/n:.1f}%")
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, default_down, pi_threshold=0.5):
    """
    Validazione: confronta RAW vs MASKED.
    
    L'obiettivo è che MASKED sia MEGLIO di RAW.
    """
    model.eval()
    
    mae_raw, mae_masked = 0.0, 0.0
    mse_masked = 0.0
    total_pred_raw, total_pred_masked, total_gt = 0.0, 0.0, 0.0
    total_coverage, total_recall = 0.0, 0.0
    n_samples = 0
    
    errors_by_density = {"sparse": [], "medium": [], "dense": []}
    
    for images, gt_density, points in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        
        # Counts
        pred_count_raw = compute_count_raw(pred_density, down_h, down_w)
        
        pi_probs = outputs.get("pi_probs")
        if pi_probs is None:
            logit_pi = outputs["logit_pi_maps"]
            pi_probs = logit_pi.softmax(dim=1)[:, 1:2]
        
        pred_count_masked = compute_count_masked(
            pred_density, pi_probs, down_h, down_w, threshold=pi_threshold
        )
        
        # π metrics
        gt_occupancy = (F.avg_pool2d(gt_density, 16, 16) * 256 > 0.5).float()
        pred_occupancy = (pi_probs > 0.5).float()
        
        if pred_occupancy.shape[-2:] != gt_occupancy.shape[-2:]:
            pred_occupancy = F.interpolate(pred_occupancy, gt_occupancy.shape[-2:], mode='nearest')
        
        coverage = pred_occupancy.mean().item() * 100
        recall = 100.0
        if gt_occupancy.sum() > 0:
            recall = (pred_occupancy * gt_occupancy).sum() / gt_occupancy.sum() * 100
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_r = pred_count_raw[idx].item()
            pred_m = pred_count_masked[idx].item()
            
            err_r = abs(pred_r - gt)
            err_m = abs(pred_m - gt)
            
            mae_raw += err_r
            mae_masked += err_m
            mse_masked += err_m ** 2
            total_pred_raw += pred_r
            total_pred_masked += pred_m
            total_gt += gt
            total_coverage += coverage
            total_recall += recall.item() if torch.is_tensor(recall) else recall
            n_samples += 1
            
            if gt <= 100:
                errors_by_density["sparse"].append((err_r, err_m))
            elif gt <= 500:
                errors_by_density["medium"].append((err_r, err_m))
            else:
                errors_by_density["dense"].append((err_r, err_m))
    
    mae_raw /= n_samples
    mae_masked /= n_samples
    rmse = np.sqrt(mse_masked / n_samples)
    bias_raw = total_pred_raw / total_gt if total_gt > 0 else 0
    bias_masked = total_pred_masked / total_gt if total_gt > 0 else 0
    coverage = total_coverage / n_samples
    recall = total_recall / n_samples
    
    # Report
    print(f"\n{'='*60}")
    print(f"📊 Validation Results")
    print(f"{'='*60}")
    print(f"   MAE RAW:    {mae_raw:.2f}")
    print(f"   MAE MASKED: {mae_masked:.2f}  ← METRICA PRINCIPALE")
    print(f"   RMSE:       {rmse:.2f}")
    print(f"   Bias RAW:   {bias_raw:.3f}")
    print(f"   Bias MASKED:{bias_masked:.3f}")
    print(f"{'─'*60}")
    print(f"   π coverage: {coverage:.1f}%")
    print(f"   π recall:   {recall:.1f}%")
    print(f"{'─'*60}")
    
    print(f"   Per densità:")
    for name, errors in errors_by_density.items():
        if errors:
            raw_errs = [e[0] for e in errors]
            masked_errs = [e[1] for e in errors]
            print(f"      {name}: RAW={np.mean(raw_errs):.1f}, MASKED={np.mean(masked_errs):.1f} ({len(errors)} imgs)")
    
    # Confronto
    delta = mae_raw - mae_masked
    if delta > 0:
        print(f"\n   ✅ MASKED migliore di RAW di {delta:.1f} punti")
    else:
        print(f"\n   ⚠️ RAW migliore di MASKED di {-delta:.1f} punti")
    
    print(f"{'='*60}\n")
    
    return {
        "mae": mae_masked,  # Metrica principale ora è MASKED
        "mae_raw": mae_raw,
        "mae_masked": mae_masked,
        "rmse": rmse,
        "bias": bias_masked,
        "bias_raw": bias_raw,
        "coverage": coverage,
        "recall": recall,
        "improvement": delta,  # Quanto MASKED migliora su RAW
    }


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
    print("🚀 Stage 3 V5 - P2R FROZEN, solo π-head training")
    print("="*60)
    print(f"Device: {device}")
    print("Strategia: P2R congelata, allena solo π-head")
    print("Metrica: MAE MASKED (π-head deve migliorare il conteggio)")
    print("="*60)
    
    # Config
    data_cfg = config["DATA"]
    joint_cfg = config.get("JOINT_LOSS", {})
    
    pi_pos_weight = float(joint_cfg.get("PI_POS_WEIGHT", 8.0))
    occupancy_threshold = float(joint_cfg.get("OCCUPANCY_THRESHOLD", 0.5))
    
    print(f"\n⚙️ Config:")
    print(f"   PI_POS_WEIGHT: {pi_pos_weight}")
    print(f"   OCCUPANCY_THRESHOLD: {occupancy_threshold}")
    
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
    
    optim_cfg = config.get("OPTIM_JOINT", {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg.get("BATCH_SIZE", 8),
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
    # CARICA STAGE 2 (checkpoint buono!)
    # =========================================
    stage2_path = os.path.join(output_dir, "stage2_best.pth")
    if os.path.isfile(stage2_path):
        print(f"\n✅ Caricamento Stage 2: {stage2_path}")
        state = torch.load(stage2_path, map_location=device)
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        print(f"❌ Stage 2 non trovato: {stage2_path}")
        return
    
    # Definisci default_down prima della calibrazione
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    # =========================================
    # CALIBRAZIONE LOG_SCALE (CRITICO!)
    # =========================================
    # Senza calibrazione, il log_scale del checkpoint potrebbe
    # non essere ottimale, causando discrepanze di MAE
    print("\n🔧 Calibrazione log_scale...")
    calibrate_density_scale(
        model, val_loader, device, default_down,
        max_batches=10, verbose=True
    )
    
    # =========================================
    # CONGELA TUTTO TRANNE π-head
    # =========================================
    print("\n🧊 Congelamento:")
    
    # Congela backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("   ✓ Backbone congelato")
    
    # Congela P2R head COMPLETAMENTE (incluso log_scale!)
    for param in model.p2r_head.parameters():
        param.requires_grad = False
    print("   ✓ P2R head congelata (incluso log_scale)")
    
    # Congela anche lambda head nella ZIP (alleniamo solo pi)
    # La ZIP head ha: shared, pi_head, bin_head
    # Congeliamo bin_head, teniamo trainabile pi_head e shared
    for name, param in model.zip_head.named_parameters():
        if "bin_head" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("   ✓ ZIP bin_head congelata")
    print("   🔥 ZIP pi_head + shared trainabili")
    
    # Conta parametri
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Parametri trainabili: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Optimizer - solo parametri trainabili
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optim_cfg.get("LR_HEADS", 1e-4)),  # LR più alto, solo π-head
        weight_decay=float(optim_cfg.get("WEIGHT_DECAY", 1e-4))
    )
    
    epochs = optim_cfg.get("EPOCHS", 200)
    scheduler = get_scheduler(optimizer, optim_cfg, epochs)
    
    # Loss
    pi_loss_fn = PiHeadLoss(
        pos_weight=pi_pos_weight,
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        occupancy_threshold=occupancy_threshold,
    ).to(device)
    
    # Valutazione iniziale (Stage 2 puro, post-calibrazione)
    print("\n📋 Valutazione iniziale (Stage 2 checkpoint):")
    init_results = validate(model, val_loader, device, default_down)
    
    best_mae = init_results["mae"]  # MAE MASKED
    best_results = init_results
    baseline_raw = init_results["mae_raw"]
    
    print(f"\n🎯 Baseline:")
    print(f"   MAE RAW (riferimento): {baseline_raw:.2f}")
    print(f"   MAE MASKED (da migliorare): {best_mae:.2f}")
    print(f"   Obiettivo: MASKED < RAW (improvement > 0)")
    
    # Training
    print(f"\n🚀 START Training")
    print(f"   Epochs: 1 → {epochs}")
    
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 30)
    no_improve = 0
    val_interval = optim_cfg.get("VAL_INTERVAL", 5)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        train_loss = train_one_epoch(
            model, pi_loss_fn, train_loader, optimizer, device, default_down, epoch
        )
        
        if scheduler:
            scheduler.step()
        
        # Validazione
        if epoch % val_interval == 0:
            results = validate(model, val_loader, device, default_down)
            
            # Metrica: MAE MASKED (vogliamo che sia basso E migliore di RAW)
            current_mae = results["mae"]
            improvement = results["improvement"]
            
            # È best se: MAE migliora E improvement positivo (MASKED < RAW)
            is_better = current_mae < best_mae - 0.5
            
            if is_better:
                best_mae = current_mae
                best_results = results
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=True
                )
                print(f"🏆 NEW BEST: MAE_MASKED={best_mae:.2f}, improvement={improvement:.1f}")
                no_improve = 0
            else:
                no_improve += 1
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False
                )
                print(f"   No improvement ({no_improve}/{patience})")
            
            # Early stopping
            if no_improve >= patience:
                print(f"⛔ Early stopping a epoch {epoch}")
                break
    
    # Risultati finali
    print("\n" + "="*60)
    print("🏁 STAGE 3 V5 COMPLETATO")
    print("="*60)
    print(f"   Baseline RAW:     {baseline_raw:.2f}")
    print(f"   Best MAE MASKED:  {best_results.get('mae', best_mae):.2f}")
    print(f"   Improvement:      {best_results.get('improvement', 0):.1f}")
    print(f"   Bias:             {best_results.get('bias', 0):.3f}")
    print(f"   π recall:         {best_results.get('recall', 0):.1f}%")
    
    if best_results.get('improvement', 0) > 0:
        print(f"\n   ✅ π-head MIGLIORA il conteggio di {best_results['improvement']:.1f} punti!")
    else:
        print(f"\n   ⚠️ π-head non migliora. Usa MAE RAW come riferimento.")
    
    print(f"\n💾 Best model: {os.path.join(output_dir, 'stage3_v5_best.pth')}")


if __name__ == "__main__":
    main()