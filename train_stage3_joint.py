#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 - Joint Training CORRETTO V2

Modifiche V2:
- Aggiunto salvataggio di stage3_latest.pth per resume
- Aggiunta funzione resume_stage3() per riprendere training
- Mantiene tutte le correzioni V1 (soft mask, end-to-end gradients)

Obiettivo: Allenare π-head e P2R INSIEME in modo che:
1. Il conteggio finale usa P2R × π_mask (come da design)
2. Il count loss backpropaga attraverso la maschera
3. Il π-head impara a NON mascherare regioni con persone
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
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_stage3_checkpoint(model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False):
    """Salva checkpoint Stage 3 (latest + opzionalmente best)."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_mae": best_mae,
        "best_results": best_results,
    }
    
    # Salva SEMPRE latest
    latest_path = os.path.join(output_dir, "stage3_latest.pth")
    torch.save(checkpoint, latest_path)
    
    # Salva best se richiesto
    if is_best:
        best_path = os.path.join(output_dir, "stage3_best.pth")
        torch.save(checkpoint, best_path)
        print(f"💾 Saved: stage3_latest.pth + stage3_best.pth (MAE={best_mae:.2f})")
    else:
        print(f"💾 Saved: stage3_latest.pth (epoch {epoch})")


def resume_stage3(model, optimizer, scheduler, output_dir, device):
    """
    Riprende Stage 3 da stage3_latest.pth se esiste.
    
    Returns:
        start_epoch: epoca da cui riprendere (1 se partenza da zero)
        best_mae: miglior MAE finora (inf se partenza da zero)
        best_results: dizionario risultati migliori
    """
    latest_path = os.path.join(output_dir, "stage3_latest.pth")
    
    if not os.path.isfile(latest_path):
        return 1, float('inf'), {}
    
    print(f"🔄 Resume Stage 3 da: {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device)
    
    # Carica stato modello
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Carica stato optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Carica stato scheduler
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_mae = checkpoint.get("best_mae", float('inf'))
    best_results = checkpoint.get("best_results", {})
    
    print(f"✅ Ripreso da epoch {start_epoch - 1}, best_mae={best_mae:.2f}")
    
    return start_epoch, best_mae, best_results


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class PiHeadLoss(nn.Module):
    """
    Loss per π-head con alto pos_weight per penalizzare falsi negativi.
    Un falso negativo = mascherare una regione con persone.
    """
    def __init__(self, pos_weight: float = 15.0, block_size: int = 16, 
                 occupancy_threshold: float = 0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        """
        Calcola GT occupancy con soglia bassa per catturare anche
        regioni con poche persone.
        """
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        # Soglia bassa: anche 0.5 persone per blocco = occupato
        return (gt_counts_per_block > self.occupancy_threshold).float()
    
    def forward(self, logit_pi_maps, gt_density):
        """
        Args:
            logit_pi_maps: [B, 2, H, W] logits dal π-head
            gt_density: [B, 1, H, W] density map GT
        """
        logit_pieno = logit_pi_maps[:, 1:2, :, :]  # Canale "pieno"
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
            pred_occupancy = (torch.sigmoid(logit_pieno) > 0.5).float()
            coverage = pred_occupancy.mean().item() * 100
            
            # Recall: quante regioni occupate sono state trovate?
            if gt_occupancy.sum() > 0:
                recall = (pred_occupancy * gt_occupancy).sum() / gt_occupancy.sum()
                recall = recall.item() * 100
            else:
                recall = 100.0
        
        return loss, {"pi_coverage": coverage, "pi_recall": recall}


class P2RSpatialLoss(nn.Module):
    """
    Loss spaziale per P2R - incoraggia la density map ad avere
    valori alti dove ci sono persone.
    """
    def __init__(self, sigma: float = 8.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred_density, points, down_h, down_w):
        """
        MSE loss tra density predetta e target gaussiano.
        """
        B, _, H, W = pred_density.shape
        device = pred_density.device
        total_loss = 0.0
        
        for i in range(B):
            pts = points[i]
            if pts is None or len(pts) == 0:
                continue
            
            # Crea target gaussiano
            target = torch.zeros(H, W, device=device)
            
            for pt in pts:
                x, y = pt[0].item() / down_w, pt[1].item() / down_h
                
                # Coordinate griglia
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=device, dtype=torch.float32),
                    torch.arange(W, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                
                # Gaussiana
                gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (self.sigma/down_h)**2))
                target += gaussian
            
            # Normalizza
            if target.max() > 0:
                target = target / target.max()
            
            pred_norm = pred_density[i, 0]
            if pred_norm.max() > 0:
                pred_norm = pred_norm / pred_norm.max()
            
            total_loss += F.mse_loss(pred_norm, target)
        
        return total_loss / max(B, 1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_soft_mask(logit_pi_maps, temperature: float = 1.0):
    """
    Crea soft mask dal π-head usando sigmoid.
    Soft mask permette ai gradienti di fluire.
    
    Args:
        logit_pi_maps: [B, 2, H, W] 
        temperature: più basso = più sharp (simile a hard mask)
    
    Returns:
        soft_mask: [B, 1, H, W] valori in [0, 1]
    """
    logit_pieno = logit_pi_maps[:, 1:2, :, :]  # Canale "pieno"
    soft_mask = torch.sigmoid(logit_pieno / temperature)
    return soft_mask


def compute_masked_count(pred_density, soft_mask, down_h, down_w):
    """
    Calcola il conteggio usando la density mascherata.
    
    count = sum(density * mask) / cell_area
    
    I gradienti fluiscono sia a density che a mask.
    """
    cell_area = down_h * down_w
    
    # Upsample mask se necessario
    if soft_mask.shape[-2:] != pred_density.shape[-2:]:
        soft_mask = F.interpolate(
            soft_mask, 
            size=pred_density.shape[-2:], 
            mode='bilinear',
            align_corners=False
        )
    
    masked_density = pred_density * soft_mask
    count = torch.sum(masked_density, dim=(1, 2, 3)) / cell_area
    
    return count, masked_density


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model, pi_loss_fn, spatial_loss_fn,
    dataloader, optimizer, device, default_down, epoch, config
):
    model.train()
    
    # Pesi loss
    pi_weight = config["PI_WEIGHT"]
    count_weight = config["COUNT_WEIGHT"]
    spatial_weight = config["SPATIAL_WEIGHT"]
    scale_weight = config["SCALE_WEIGHT"]
    temperature = config["MASK_TEMPERATURE"]
    
    total_loss = 0.0
    total_mae = 0.0
    total_coverage = 0.0
    total_recall = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Joint [Ep {epoch}]")
    
    for images, gt_density, points in progress_bar:
        images = images.to(device)
        gt_density = gt_density.to(device)
        points = [p.to(device) if p is not None else None for p in points]
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # === 1. P2R Density ===
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        
        # === 2. Soft Mask dal π-head ===
        logit_pi = outputs["logit_pi_maps"]
        soft_mask = get_soft_mask(logit_pi, temperature=temperature)
        
        # === 3. Conteggio MASKED (design originale) ===
        pred_count, masked_density = compute_masked_count(
            pred_density, soft_mask, down_h, down_w
        )
        
        gt_count = torch.tensor(
            [len(p) if p is not None else 0 for p in points],
            device=device, dtype=torch.float32
        )
        
        # === LOSSES ===
        
        # L1: Count Loss (principale - guida tutto)
        loss_count = F.l1_loss(pred_count, gt_count)
        
        # L2: Scale Loss (errore relativo)
        with torch.no_grad():
            valid_mask = gt_count > 1
        if valid_mask.any():
            scale_errors = torch.abs(pred_count[valid_mask] - gt_count[valid_mask]) / gt_count[valid_mask]
            loss_scale = scale_errors.mean()
        else:
            loss_scale = torch.tensor(0.0, device=device)
        
        # L3: π-head Loss (supervisione diretta per coverage)
        loss_pi, pi_metrics = pi_loss_fn(logit_pi, gt_density)
        
        # L4: Spatial Loss (opzionale, aiuta la forma)
        if spatial_weight > 0:
            loss_spatial = spatial_loss_fn(pred_density, points, down_h, down_w)
        else:
            loss_spatial = torch.tensor(0.0, device=device)
        
        # === Loss Totale ===
        total = (
            count_weight * loss_count +
            scale_weight * loss_scale +
            pi_weight * loss_pi +
            spatial_weight * loss_spatial
        )
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # === Metriche ===
        with torch.no_grad():
            mae = torch.abs(pred_count - gt_count).mean().item()
            total_mae += mae
            total_coverage += pi_metrics["pi_coverage"]
            total_recall += pi_metrics["pi_recall"]
        
        total_loss += total.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            "L": f"{total.item():.1f}",
            "MAE": f"{mae:.1f}",
            "π_cov": f"{pi_metrics['pi_coverage']:.0f}%",
            "π_rec": f"{pi_metrics['pi_recall']:.0f}%"
        })
    
    n = max(num_batches, 1)
    print(f"   Train: MAE={total_mae/n:.2f}, π_coverage={total_coverage/n:.1f}%, π_recall={total_recall/n:.1f}%")
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, default_down, temperature=1.0):
    """Validazione con confronto raw vs masked."""
    model.eval()
    
    mae_masked, mae_raw = 0.0, 0.0
    mse_masked = 0.0
    total_pred, total_gt = 0.0, 0.0
    total_coverage, total_recall = 0.0, 0.0
    n_samples = 0
    
    # Per analisi stratificata
    errors_by_density = {"sparse": [], "medium": [], "dense": []}
    
    for images, gt_density, points in tqdm(dataloader, desc="Validate"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        outputs = model(images)
        
        # P2R
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        
        # Raw count
        pred_count_raw = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        # Masked count
        logit_pi = outputs["logit_pi_maps"]
        soft_mask = get_soft_mask(logit_pi, temperature=temperature)
        pred_count_masked, _ = compute_masked_count(pred_density, soft_mask, down_h, down_w)
        
        # Coverage & Recall
        gt_occupancy = (F.avg_pool2d(gt_density, 16, 16) * 256 > 0.5).float()
        pred_occupancy = (soft_mask > 0.5).float()
        
        if pred_occupancy.shape[-2:] != gt_occupancy.shape[-2:]:
            pred_occupancy_resized = F.interpolate(pred_occupancy, gt_occupancy.shape[-2:], mode='nearest')
        else:
            pred_occupancy_resized = pred_occupancy
        
        coverage = pred_occupancy.mean().item() * 100
        if gt_occupancy.sum() > 0:
            recall = (pred_occupancy_resized * gt_occupancy).sum() / gt_occupancy.sum() * 100
        else:
            recall = 100.0
        
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            pred_m = pred_count_masked[idx].item()
            pred_r = pred_count_raw[idx].item()
            
            err_m = abs(pred_m - gt)
            err_r = abs(pred_r - gt)
            
            mae_masked += err_m
            mae_raw += err_r
            mse_masked += err_m ** 2
            total_pred += pred_m
            total_gt += gt
            total_coverage += coverage
            total_recall += recall if isinstance(recall, float) else recall.item()
            n_samples += 1
            
            # Stratificazione
            if gt <= 100:
                errors_by_density["sparse"].append(err_m)
            elif gt <= 500:
                errors_by_density["medium"].append(err_m)
            else:
                errors_by_density["dense"].append(err_m)
    
    # Risultati
    mae_masked /= n_samples
    mae_raw /= n_samples
    rmse = np.sqrt(mse_masked / n_samples)
    bias = total_pred / total_gt if total_gt > 0 else 0
    coverage = total_coverage / n_samples
    recall = total_recall / n_samples
    
    print(f"\n📊 Validation:")
    print(f"   MAE MASKED: {mae_masked:.2f} (questo è il target)")
    print(f"   MAE RAW:    {mae_raw:.2f} (riferimento)")
    print(f"   RMSE:       {rmse:.2f}")
    print(f"   Bias:       {bias:.3f}")
    print(f"   π coverage: {coverage:.1f}%")
    print(f"   π recall:   {recall:.1f}%")
    
    # Analisi stratificata
    print(f"\n   Per densità:")
    for name, errors in errors_by_density.items():
        if errors:
            print(f"      {name}: MAE={np.mean(errors):.1f} ({len(errors)} imgs)")
    
    # Confronto
    if mae_masked < mae_raw:
        print(f"\n   ✅ π-head AIUTA! (masked < raw di {mae_raw - mae_masked:.1f})")
    else:
        print(f"\n   ⚠️ π-head peggiora di {mae_masked - mae_raw:.1f} (recall={recall:.1f}%)")
    
    return {
        "mae": mae_masked,
        "mae_raw": mae_raw,
        "rmse": rmse,
        "bias": bias,
        "coverage": coverage,
        "recall": recall,
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
    print(f"✅ Avvio Stage 3 Joint Training CORRETTO V2 su {device}")
    
    # === Joint Config ===
    joint_cfg = config.get("JOINT_LOSS", {})
    loss_config = {
        "PI_WEIGHT": float(joint_cfg.get("PI_WEIGHT", 0.5)),
        "COUNT_WEIGHT": float(joint_cfg.get("COUNT_WEIGHT", 2.0)),
        "SCALE_WEIGHT": float(joint_cfg.get("SCALE_WEIGHT", 0.3)),
        "SPATIAL_WEIGHT": float(joint_cfg.get("SPATIAL_WEIGHT", 0.1)),
        "MASK_TEMPERATURE": float(joint_cfg.get("MASK_TEMPERATURE", 1.0)),
        "PI_POS_WEIGHT": float(joint_cfg.get("PI_POS_WEIGHT", 15.0)),
        "OCCUPANCY_THRESHOLD": float(joint_cfg.get("OCCUPANCY_THRESHOLD", 0.5)),
    }
    
    print(f"\n⚙️ Loss Config:")
    for k, v in loss_config.items():
        print(f"   {k}: {v}")
    
    # === Dataset ===
    data_cfg = config["DATA"]
    optim_cfg = config.get("OPTIM_JOINT", {})
    
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
    
    # === Modello ===
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
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # === Output Directory ===
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    
    # === Freeze/Unfreeze ===
    # Congela backbone, allena entrambi gli head
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("🧊 Backbone congelato")
    print("🔥 ZIP head e P2R head trainabili")
    
    # === Optimizer ===
    trainable_params = list(model.zip_head.parameters()) + list(model.p2r_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optim_cfg.get("LR_HEADS", 5e-5)),
        weight_decay=float(optim_cfg.get("WEIGHT_DECAY", 1e-4))
    )
    
    epochs = optim_cfg.get("EPOCHS", 200)
    scheduler = get_scheduler(optimizer, optim_cfg, epochs)
    
    # === RESUME O CARICA STAGE 2 ===
    start_epoch, best_mae, best_results = resume_stage3(
        model, optimizer, scheduler, output_dir, device
    )
    
    # Se non c'è checkpoint Stage 3, carica Stage 2
    if start_epoch == 1:
        stage2_path = os.path.join(output_dir, "stage2_best.pth")
        if os.path.isfile(stage2_path):
            print(f"✅ Caricamento Stage 2: {stage2_path}")
            state = torch.load(stage2_path, map_location=device)
            if "model" in state:
                state = state["model"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
        else:
            print(f"⚠️ Stage 2 non trovato: {stage2_path}")
            return
    
    # === Loss Functions ===
    pi_loss_fn = PiHeadLoss(
        pos_weight=loss_config["PI_POS_WEIGHT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        occupancy_threshold=loss_config["OCCUPANCY_THRESHOLD"]
    ).to(device)
    
    spatial_loss_fn = P2RSpatialLoss(sigma=8.0).to(device)
    
    # === Valutazione iniziale (solo se partenza da zero) ===
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    if start_epoch == 1:
        print("\n📋 Valutazione iniziale (Stage 2):")
        init_results = validate(model, val_loader, device, default_down, loss_config["MASK_TEMPERATURE"])
        if init_results["mae"] < best_mae:
            best_mae = init_results["mae"]
            best_results = init_results
    
    # === Training ===
    print(f"\n🚀 START Joint Training")
    print(f"   Obiettivo: MAE MASKED < MAE RAW (π-head deve aiutare)")
    print(f"   Epochs: {start_epoch} → {epochs}")
    
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 30)
    no_improve = 0
    val_interval = optim_cfg.get("VAL_INTERVAL", 5)
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        train_loss = train_one_epoch(
            model, pi_loss_fn, spatial_loss_fn,
            train_loader, optimizer, device, default_down, epoch, loss_config
        )
        
        if scheduler:
            scheduler.step()
        
        # Salva latest ad ogni epoca
        save_stage3_checkpoint(
            model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=False
        )
        
        # Validazione
        if epoch % val_interval == 0:
            results = validate(model, val_loader, device, default_down, loss_config["MASK_TEMPERATURE"])
            
            if results["mae"] < best_mae:
                improvement = best_mae - results["mae"]
                best_mae = results["mae"]
                best_results = results
                save_stage3_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae, best_results, output_dir, is_best=True
                )
                print(f"🏆 NEW BEST: MAE={best_mae:.2f} (↓{improvement:.2f})")
                no_improve = 0
            else:
                no_improve += 1
                print(f"   No improvement ({no_improve}/{patience})")
            
            if patience > 0 and no_improve >= patience:
                print(f"⛔ Early stopping a epoch {epoch}")
                break
    
    # === Risultati Finali ===
    print("\n" + "="*60)
    print("🏁 STAGE 3 COMPLETATO")
    print("="*60)
    print(f"   Best MAE (masked): {best_results.get('mae', best_mae):.2f}")
    print(f"   MAE raw:           {best_results.get('mae_raw', 'N/A')}")
    print(f"   Bias:              {best_results.get('bias', 'N/A')}")
    print(f"   π coverage:        {best_results.get('coverage', 'N/A')}")
    print(f"   π recall:          {best_results.get('recall', 'N/A')}")
    
    if best_results.get('mae', float('inf')) < best_results.get('mae_raw', float('inf')):
        delta = best_results['mae_raw'] - best_results['mae']
        print(f"\n   ✅ SUCCESSO: π-head migliora il conteggio di {delta:.1f}!")
    else:
        print(f"\n   ⚠️ π-head ancora non aiuta - potrebbe servire più training")
    
    if best_mae <= 60:
        print("\n🎯 TARGET RAGGIUNTO! MAE ≤ 60")
    elif best_mae <= 70:
        print("\n✅ Buon risultato! MAE ≤ 70")
    
    print(f"\n💾 Modello: {os.path.join(output_dir, 'stage3_best.pth')}")


if __name__ == "__main__":
    main()