# train_stage1_zip_v2.py
# -*- coding: utf-8 -*-
"""
Stage 1 V2 - ZIP Pre-training con Loss IBRIDA

PROBLEMA RISOLTO:
La versione originale usava solo BCE per π-head, il che insegnava al backbone
solo a distinguere "vuoto/pieno". Le features risultanti non erano ottimali
per la regressione di densità in Stage 2.

SOLUZIONE:
Loss ibrida che combina:
1. BCE su π-head (classificazione binaria)
2. Count Loss su λ-map (regressione conteggio per blocco)

Questo fa sì che:
- Il backbone impari features utili sia per localizzazione che per magnitudine
- λ_maps abbia già una stima ragionevole prima di Stage 2
- π-head e density siano allineati sulla stessa "scala"

L_total = L_BCE(π) + α * L_count(λ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    resume_if_exists,
    save_checkpoint,
    collate_fn,
)


# ============================================================
# LOSS IBRIDA PER STAGE 1 V2
# ============================================================

class HybridZIPLoss(nn.Module):
    """
    Loss ibrida per Stage 1:
    
    L_total = L_BCE(π) + α * L_count(λ) + β * L_lambda_reg
    
    Componenti:
    1. L_BCE: Binary Cross-Entropy per π-head (vuoto/pieno)
    2. L_count: Smooth L1 loss sul conteggio per blocco (λ * π)
    3. L_lambda_reg: Regolarizzazione per evitare λ troppo alti
    
    Questo insegna al backbone a codificare BOTH localizzazione AND magnitudine.
    """
    
    def __init__(
        self,
        pos_weight_bce: float = 5.0,      # Peso per classe positiva (blocchi pieni)
        count_weight: float = 0.5,         # Peso della loss di conteggio (α)
        lambda_reg_weight: float = 0.01,   # Peso regolarizzazione λ (β)
        block_size: int = 16,
        occupancy_threshold: float = 0.5,  # Soglia per definire "blocco pieno"
        smooth_l1_beta: float = 1.0,       # Beta per Smooth L1
    ):
        super().__init__()
        self.pos_weight_bce = pos_weight_bce
        self.count_weight = count_weight
        self.lambda_reg_weight = lambda_reg_weight
        self.block_size = block_size
        self.occupancy_threshold = occupancy_threshold
        self.smooth_l1_beta = smooth_l1_beta
        
        # BCE con pos_weight per bilanciare le classi
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_bce]),
            reduction='mean'
        )
        
        # Smooth L1 per count (più robusto agli outlier di L1 puro)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=smooth_l1_beta)
    
    def compute_gt_per_block(self, gt_density):
        """
        Calcola ground truth per blocco dalla density map.
        
        Returns:
            gt_occupancy: [B, 1, Hb, Wb] - maschera binaria
            gt_counts: [B, 1, Hb, Wb] - conteggio per blocco
        """
        # Somma la densità nel blocco
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        # Occupancy binaria: blocco pieno se count > threshold
        gt_occupancy = (gt_counts_per_block > self.occupancy_threshold).float()
        
        return gt_occupancy, gt_counts_per_block
    
    def forward(self, predictions, gt_density):
        """
        Args:
            predictions: dict con 'logit_pi_maps', 'lambda_maps'
            gt_density: [B, 1, H, W] density map ground truth
            
        Returns:
            total_loss: scalar
            loss_dict: dict con componenti per logging
        """
        logit_pi_maps = predictions["logit_pi_maps"]  # [B, 2, Hb, Wb]
        lambda_maps = predictions["lambda_maps"]       # [B, 1, Hb, Wb]
        
        # Estrai logit per "pieno" (canale 1)
        logit_pieno = logit_pi_maps[:, 1:2, :, :]  # [B, 1, Hb, Wb]
        
        # Probabilità π (per il conteggio predetto)
        pi_prob = torch.sigmoid(logit_pieno)  # [B, 1, Hb, Wb]
        
        # Ground truth
        gt_occupancy, gt_counts = self.compute_gt_per_block(gt_density)
        
        # Allinea dimensioni se necessario
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest'
            )
            gt_counts = F.interpolate(
                gt_counts, size=logit_pieno.shape[-2:], mode='nearest'
            )
        
        # Sposta pos_weight sul device corretto
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
        
        # ========================
        # 1. BCE Loss su π-head
        # ========================
        loss_bce = self.bce(logit_pieno, gt_occupancy)
        
        # ========================
        # 2. Count Loss su λ (nei blocchi occupati)
        # ========================
        # Predizione conteggio: π * λ (solo dove c'è GT)
        pred_count_per_block = pi_prob * lambda_maps
        
        # Maschera per blocchi con persone (per non penalizzare blocchi vuoti)
        occupied_mask = (gt_counts > 0.5).float()
        num_occupied = occupied_mask.sum() + 1e-6
        
        # Loss solo sui blocchi occupati
        count_diff = pred_count_per_block - gt_counts
        loss_count_masked = (self.smooth_l1(
            pred_count_per_block * occupied_mask,
            gt_counts * occupied_mask
        ))
        
        # Aggiungi anche loss sui blocchi vuoti (devono predire ~0)
        empty_mask = 1.0 - occupied_mask
        loss_count_empty = (pred_count_per_block * empty_mask).mean() * 0.1
        
        loss_count = loss_count_masked + loss_count_empty
        
        # ========================
        # 3. Regolarizzazione λ
        # ========================
        # Evita che λ esploda (soft constraint)
        loss_lambda_reg = F.relu(lambda_maps - 10.0).mean()  # Penalizza λ > 10
        
        # ========================
        # Loss totale
        # ========================
        total_loss = (
            loss_bce + 
            self.count_weight * loss_count + 
            self.lambda_reg_weight * loss_lambda_reg
        )
        
        # Metriche per logging
        with torch.no_grad():
            # Accuratezza π-head
            pred_occupancy = (pi_prob > 0.5).float()
            accuracy = (pred_occupancy == gt_occupancy).float().mean().item() * 100
            
            # Coverage e Recall
            if gt_occupancy.sum() > 0:
                tp = (pred_occupancy * gt_occupancy).sum()
                fn = ((1 - pred_occupancy) * gt_occupancy).sum()
                fp = (pred_occupancy * (1 - gt_occupancy)).sum()
                recall = (tp / (tp + fn + 1e-6)).item() * 100
                precision = (tp / (tp + fp + 1e-6)).item() * 100
            else:
                recall = 100.0
                precision = 100.0
            
            coverage = pred_occupancy.mean().item() * 100
            
            # Statistiche λ
            lambda_mean = lambda_maps.mean().item()
            lambda_max = lambda_maps.max().item()
            
            # Errore conteggio medio (sui blocchi occupati)
            if num_occupied > 1:
                count_mae = (torch.abs(count_diff) * occupied_mask).sum() / num_occupied
                count_mae = count_mae.item()
            else:
                count_mae = 0.0
        
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_bce": loss_bce.item(),
            "loss_count": loss_count.item(),
            "loss_lambda_reg": loss_lambda_reg.item(),
            "accuracy": accuracy,
            "coverage": coverage,
            "recall": recall,
            "precision": precision,
            "lambda_mean": lambda_mean,
            "lambda_max": lambda_max,
            "count_mae": count_mae,
        }
        
        return total_loss, loss_dict


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch=1,
    clip_grad_norm=1.0,
):
    model.train()
    total_loss = 0.0
    metrics_accum = {}
    
    progress_bar = tqdm(dataloader, desc=f"Stage1 V2 [Ep {epoch}]")

    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        
        loss, loss_dict = criterion(predictions, gt_density)

        loss.backward()
        
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        
        # Accumula metriche
        for k, v in loss_dict.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        progress_bar.set_postfix({
            'L': f"{loss.item():.3f}",
            'bce': f"{loss_dict['loss_bce']:.3f}",
            'cnt': f"{loss_dict['loss_count']:.3f}",
            'rec': f"{loss_dict['recall']:.0f}%",
            'λ': f"{loss_dict['lambda_mean']:.2f}",
        })

    if scheduler:
        scheduler.step()
    
    # Media metriche
    n = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n
    
    print(f"   Epoch {epoch} Summary:")
    print(f"      Loss: {total_loss/n:.4f} (BCE={metrics_accum['loss_bce']:.4f}, Count={metrics_accum['loss_count']:.4f})")
    print(f"      π-head: Acc={metrics_accum['accuracy']:.1f}%, Recall={metrics_accum['recall']:.1f}%, Prec={metrics_accum['precision']:.1f}%")
    print(f"      λ-maps: mean={metrics_accum['lambda_mean']:.2f}, max={metrics_accum['lambda_max']:.2f}, CountMAE={metrics_accum['count_mae']:.2f}")
        
    return total_loss / n, metrics_accum


def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    metrics_accum = {}
    
    # Per calcolo MAE globale
    total_pred_count = 0.0
    total_gt_count = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, gt_density, points in tqdm(dataloader, desc="Validate"):
            images, gt_density = images.to(device), gt_density.to(device)

            preds = model(images)
            loss, loss_dict = criterion(preds, gt_density)
            total_loss += loss.item()
            
            # Accumula metriche
            for k, v in loss_dict.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            
            # Calcola MAE globale (come proxy per Stage 2)
            pi_prob = torch.sigmoid(preds["logit_pi_maps"][:, 1:2])
            lambda_maps = preds["lambda_maps"]
            
            # Conteggio predetto = sum(π * λ)
            pred_count_map = pi_prob * lambda_maps
            pred_count = pred_count_map.sum(dim=(1, 2, 3))
            
            for idx, pts in enumerate(points):
                gt = len(pts) if pts is not None else 0
                pred = pred_count[idx].item()
                
                total_mae += abs(pred - gt)
                total_pred_count += pred
                total_gt_count += gt
                n_samples += 1

    n = len(dataloader)
    for k in metrics_accum:
        metrics_accum[k] /= n
    
    avg_loss = total_loss / n
    mae = total_mae / n_samples if n_samples > 0 else 0
    bias = total_pred_count / total_gt_count if total_gt_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"📊 Validation Results Stage 1 V2")
    print(f"{'='*60}")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   π-head: Recall={metrics_accum['recall']:.1f}%, Precision={metrics_accum['precision']:.1f}%")
    print(f"   Coverage: {metrics_accum['coverage']:.1f}%")
    print(f"   λ-maps: mean={metrics_accum['lambda_mean']:.2f}, max={metrics_accum['lambda_max']:.2f}")
    print(f"{'─'*60}")
    print(f"   📈 Conteggio (ZIP π*λ):")
    print(f"      MAE: {mae:.2f}")
    print(f"      Bias: {bias:.3f}")
    print(f"      (Nota: questo è solo indicativo, Stage 2 userà P2R)")
    print(f"{'='*60}\n")

    return avg_loss, mae, metrics_accum


def main():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        print("❌ ERRORE: config.yaml non trovato.")
        return

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    print("="*60)
    print("🚀 Stage 1 V2 - ZIP Pre-training con Loss IBRIDA")
    print("="*60)
    print(f"Device: {device}")
    print("Strategia: BCE(π) + Count(λ) + Regularization")
    print("="*60)

    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_cfg["bins"], bin_cfg["bin_centers"]

    # Configurazione Modello
    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=config["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Congelamento P2R head (come prima)
    for p in model.p2r_head.parameters():
        p.requires_grad = False
    print("🧊 P2R head congelata")
    
    # Conta parametri trainabili
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parametri trainabili: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ========================
    # Configurazione Loss V2
    # ========================
    zip_loss_cfg = config.get("ZIP_LOSS", {})
    
    # Pesi di default ottimizzati
    pos_weight_bce = float(zip_loss_cfg.get("POS_WEIGHT_BCE", 5.0))
    count_weight = float(zip_loss_cfg.get("COUNT_WEIGHT", 0.5))
    lambda_reg_weight = float(zip_loss_cfg.get("LAMBDA_REG_WEIGHT", 0.01))
    occupancy_threshold = float(zip_loss_cfg.get("OCCUPANCY_THRESHOLD", 0.5))
    
    print(f"\n⚙️ Loss Config V2:")
    print(f"   POS_WEIGHT_BCE: {pos_weight_bce}")
    print(f"   COUNT_WEIGHT (α): {count_weight}")
    print(f"   LAMBDA_REG_WEIGHT (β): {lambda_reg_weight}")
    print(f"   OCCUPANCY_THRESHOLD: {occupancy_threshold}")
    
    criterion = HybridZIPLoss(
        pos_weight_bce=pos_weight_bce,
        count_weight=count_weight,
        lambda_reg_weight=lambda_reg_weight,
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
        occupancy_threshold=occupancy_threshold,
    ).to(device)

    # Optimizer
    optim_cfg = config["OPTIM_ZIP"]
    lr_head = optim_cfg.get("LR", optim_cfg.get("BASE_LR", 1e-4))
    lr_backbone = optim_cfg.get("LR_BACKBONE", lr_head * 0.1)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.zip_head.parameters() if p.requires_grad]

    print(f"\n⚙️ Optimizer:")
    print(f"   LR backbone: {lr_backbone}")
    print(f"   LR heads: {lr_head}")

    optimizer = get_optimizer(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        optim_cfg,
    )
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg.get("EPOCHS", 100))

    # Dataset & Dataloader
    data_cfg = config["DATA"]
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf,
    )
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_set)} samples")
    print(f"   Val: {len(val_set)} samples")

    # Output Dir
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)
    
    # Resume
    start_epoch, best_metric = resume_if_exists(model, optimizer, out_dir, device)
    
    # Reset best_metric se cambiamo logica
    if start_epoch == 1:
        best_metric = float('inf')

    epochs = optim_cfg.get("EPOCHS", 100)
    val_interval = max(1, optim_cfg.get("VAL_INTERVAL", 5))
    es_patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 500)))
    epochs_no_improve = 0
    
    print(f"\n🚀 START Training")
    print(f"   Epochs: {start_epoch} → {epochs}")
    print(f"   Val interval: {val_interval}")
    print(f"   Early stopping patience: {es_patience}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        train_loss, train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            clip_grad_norm=1.0,
        )

        if epoch % val_interval == 0 or epoch == epochs:
            val_loss, val_mae, val_metrics = validate(model, criterion, val_loader, device)
            
            # Usiamo la val_loss come metrica (combinazione di BCE + Count)
            current_metric = val_loss
            is_best = current_metric < best_metric

            if is_best:
                best_metric = current_metric
                epochs_no_improve = 0
                print(f"🏆 NEW BEST: Loss={best_metric:.4f}")
            else:
                epochs_no_improve += 1
                print(f"   No improvement ({epochs_no_improve}/{es_patience})")

            save_checkpoint(
                model, optimizer, epoch, current_metric, best_metric, out_dir, is_best=is_best
            )
            
            if es_patience > 0 and epochs_no_improve >= es_patience:
                print(f"⛔ Early stopping a epoch {epoch}")
                break

    print("\n" + "="*60)
    print("🏁 STAGE 1 V2 COMPLETATO")
    print("="*60)
    print(f"   Best Loss: {best_metric:.4f}")
    print(f"   Checkpoint: {os.path.join(out_dir, 'best_model.pth')}")
    print("\n   ➡️  Ora esegui Stage 2: python train_stage2_p2r.py")


if __name__ == "__main__":
    main()