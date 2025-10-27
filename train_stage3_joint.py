# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 3 — Joint Training (ZIP + P2R Fine-tuning)
# ============================================================

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from losses.p2r_region_loss import P2RLoss
from datasets import get_dataset
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    save_checkpoint
)
import torch.nn.functional as F

# --- utility per arrotondare a multipli di 8 ---
def _round_up_8(x: int) -> int:
    return (x + 7) // 8 * 8

def collate_joint(batch):
    """
    Collate function per dataset che restituisce dizionari.
    Ogni elemento del batch è un dict con chiavi:
        'image', 'density', 'points'
    """
    # compatibilità con dataset tuple o dict
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
        pts  = [b.get("points", None) for b in batch]
    else:
        imgs, dens, pts = zip(*[(s[0], s[1], s[2]) for s in batch])

    # misura massima nel batch, arrotondata a multiplo di 8
    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out, pts_out = [], [], []
    for im, den, p in zip(imgs, dens, pts):
        _, H, W = im.shape
        sy, sx = H_tgt / H, W_tgt / W

        # resize immagine
        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode='bilinear', align_corners=False).squeeze(0)

        # resize densità mantenendo il conteggio
        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode='bilinear', align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)

        # scala i punti
        if p is None or (hasattr(p, "numel") and p.numel() == 0):
            p_scaled = p
        else:
            p_scaled = p.clone()
            p_scaled[:, 0] *= sx
            p_scaled[:, 1] *= sy

        imgs_out.append(im_res)
        dens_out.append(den_res)
        pts_out.append(p_scaled)

    return torch.stack(imgs_out), torch.stack(dens_out), pts_out

def collate_val(batch):
    """
    Collate function per la validazione (senza punti).
    """
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
    else:
        imgs, dens = zip(*[(s[0], s[1]) for s in batch])

    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out = [], []
    for im, den in zip(imgs, dens):
        _, H, W = im.shape
        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode='bilinear', align_corners=False).squeeze(0)
        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode='bilinear', align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)
        imgs_out.append(im_res)
        dens_out.append(den_res)

    dummy_pts = [None] * len(imgs_out)
    return torch.stack(imgs_out), torch.stack(dens_out), dummy_pts


def train_one_epoch(model, criterion_zip, criterion_p2r, alpha, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 3 (Joint)")

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        outputs = model(images)

        # --- ZIP Loss ---
        loss_zip, loss_dict_zip = criterion_zip(outputs, gt_density)

        # --- P2R Loss ---
        loss_p2r = criterion_p2r(outputs["p2r_density"], points)

        # --- Combined Loss ---
        combined_loss = loss_zip + alpha * loss_p2r
        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += combined_loss.item()
        progress_bar.set_postfix({
            "total": f"{combined_loss.item():.4f}",
            "zip": f"{loss_zip.item():.4f}",
            "p2r": f"{loss_p2r.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    return total_loss / len(dataloader)


# -----------------------------------------------------------
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    mae, mse = 0.0, 0.0

    for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 3"):
        images, gt_density = images.to(device), gt_density.to(device)

        outputs = model(images)
        pred_density = outputs["p2r_density"]

        pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        gt_count = torch.sum(gt_density, dim=(1, 2, 3))

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse


# -----------------------------------------------------------
def main():
    # === CONFIG ===
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 3 (Joint ZIP + P2R) su {device}")

    # === MODEL SETUP ===
    dataset_name = config["DATASET"]
    bin_config = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_config["bins"], bin_config["bin_centers"]

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=config["MODEL"]["UPSAMPLE_TO_INPUT"]
    ).to(device)

    # === CARICA CHECKPOINT STAGE 2 ===
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    stage2_checkpoint_path = os.path.join(output_dir, "stage2_best.pth")

    try:
        state_dict = torch.load(stage2_checkpoint_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Caricati i pesi dallo Stage 2: {stage2_checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Checkpoint Stage 2 non trovato in {stage2_checkpoint_path}.")

    # === SBLOCCA TUTTI I PARAMETRI PER IL FINE-TUNING ===
    for p in model.parameters():
        p.requires_grad = True

    # === LOSS ===
    criterion_zip = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config["ZIP_LOSS"]["WEIGHT_CE"],
        zip_block_size=config["DATA"]["ZIP_BLOCK_SIZE"]
    ).to(device)

    criterion_p2r = P2RLoss().to(device)
    alpha = config["JOINT_LOSS"]["ALPHA"]

    # === OPTIMIZER E SCHEDULER ===
    # --- Costruzione gruppi di parametri per LR differenziati ---
    optim_cfg = config["OPTIM_JOINT"]

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': optim_cfg["LR_BACKBONE"]},
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
        'lr': optim_cfg["LR_HEADS"]}
    ]

    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # === DATASET ===
    DatasetClass = get_dataset(config["DATASET"])
    train_dataset = DatasetClass(
        root=config["DATA"]["ROOT"],
        split=config["DATA"]["TRAIN_SPLIT"],
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
        transforms=None
    )
    val_dataset = DatasetClass(
        root=config["DATA"]["ROOT"],
        split=config["DATA"]["VAL_SPLIT"],
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
        transforms=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_joint,   # <--- AGGIUNGI QUI
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_val,   # <--- aggiungi questo
    )

    # === TRAINING LOOP ===
    best_mae = float("inf")
    epochs_stage3 = optim_cfg["EPOCHS"]

    print(f"🚀 Inizio Stage 3 — Fine-tuning congiunto per {epochs_stage3} epoche...")

    for epoch in range(epochs_stage3):
        print(f"\n--- Epoch {epoch + 1}/{epochs_stage3} ---")
        train_loss = train_one_epoch(
            model, criterion_zip, criterion_p2r, alpha,
            train_loader, optimizer, scheduler, device
        )

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse = validate(model, val_loader, device)
            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val MAE {val_mae:.2f} | RMSE {val_rmse:.2f}")

            # === SALVATAGGIO BEST MODEL ===
            if val_mae < best_mae:
                best_mae = val_mae
                if config["EXP"]["SAVE_BEST"]:
                    best_path = os.path.join(output_dir, "stage3_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"💾 Nuovo best Stage 3 salvato ({best_path}) — MAE={best_mae:.2f}")

    print("✅ Stage 3 completato con successo!")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()