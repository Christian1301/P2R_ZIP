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
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    save_checkpoint, canonicalize_p2r_grid
)
from train_stage2_p2r import calibrate_density_scale
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


def train_one_epoch(
    model,
    criterion_zip,
    criterion_p2r,
    alpha,
    dataloader,
    optimizer,
    scheduler,
    schedule_step_mode,
    device,
    default_down,
    clamp_cfg=None,
):
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
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_train"
        )

        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)

        # --- Combined Loss ---
        combined_loss = loss_zip + alpha * loss_p2r
        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler and schedule_step_mode == "iteration":
            scheduler.step()

        if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
            if len(clamp_cfg) != 2:
                raise ValueError("LOG_SCALE_CLAMP deve avere due valori [min, max].")
            min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
            model.p2r_head.log_scale.data.clamp_(min_val, max_val)

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
def validate(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 3"):
        images, gt_density = images.to(device), gt_density.to(device)

        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_val"
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        cell_area_tensor = pred_density.new_tensor(cell_area)

        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area_tensor
        gt_count = torch.sum(gt_density, dim=(1, 2, 3))

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse, total_pred, total_gt


# -----------------------------------------------------------
def main():
    # === CONFIG ===
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 3 (Joint ZIP + P2R) su {device}")

    default_down = config["DATA"].get("P2R_DOWNSAMPLE", 8)
    loss_cfg = config.get("P2R_LOSS", {})
    clamp_cfg = loss_cfg.get("LOG_SCALE_CLAMP")
    max_adjust = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")

    # === MODEL SETUP ===
    dataset_name = config["DATASET"]
    bin_config = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_config["bins"], bin_config["bin_centers"]

    upsample_to_input = config["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    if upsample_to_input:
        print("ℹ️ Stage 3: disattivo temporaneamente UPSAMPLE_TO_INPUT per mantenere la stessa scala di Stage 2.")
        upsample_to_input = False

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
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    optim_cfg = config["OPTIM_JOINT"]

    # === CARICA CHECKPOINT ===
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    stage2_checkpoint_path = os.path.join(output_dir, "stage2_best.pth")
    stage3_checkpoint_path = os.path.join(output_dir, "stage3_best.pth")
    load_stage3_best = bool(optim_cfg.get("LOAD_STAGE3_BEST", False))
    resume_source = None

    if load_stage3_best and os.path.isfile(stage3_checkpoint_path):
        state_dict = torch.load(stage3_checkpoint_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        resume_source = stage3_checkpoint_path
        print(f"✅ Ripreso Stage 3 dal best precedente: {stage3_checkpoint_path}")
    else:
        if load_stage3_best:
            print("ℹ️ Stage3_best mancante: utilizzo comunque lo Stage 2 best.")
        try:
            state_dict = torch.load(stage2_checkpoint_path, map_location=device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            resume_source = stage2_checkpoint_path
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
        zip_block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
        count_weight=config["JOINT_LOSS"].get(
            "COUNT_L1_W",
            config["ZIP_LOSS"].get("WEIGHT_COUNT", 1.0),
        ),
    ).to(device)

    loss_kwargs = {}
    if "SCALE_WEIGHT" in loss_cfg:
        loss_kwargs["scale_weight"] = float(loss_cfg["SCALE_WEIGHT"])
    if "POS_WEIGHT" in loss_cfg:
        loss_kwargs["pos_weight"] = float(loss_cfg["POS_WEIGHT"])
    if "CHUNK_SIZE" in loss_cfg:
        loss_kwargs["chunk_size"] = int(loss_cfg["CHUNK_SIZE"])
    if "MIN_RADIUS" in loss_cfg:
        loss_kwargs["min_radius"] = float(loss_cfg["MIN_RADIUS"])
    if "MAX_RADIUS" in loss_cfg:
        loss_kwargs["max_radius"] = float(loss_cfg["MAX_RADIUS"])
    criterion_p2r = P2RLoss(**loss_kwargs).to(device)
    alpha = config["JOINT_LOSS"]["ALPHA"]

    # === OPTIMIZER E SCHEDULER ===
    # --- Costruzione gruppi di parametri per LR differenziati ---

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': optim_cfg["LR_BACKBONE"]},
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
        'lr': optim_cfg["LR_HEADS"]}
    ]

    fine_tune_factor = float(optim_cfg.get("FINE_TUNE_LR_SCALE", 1.0))
    if fine_tune_factor <= 0:
        raise ValueError("FINE_TUNE_LR_SCALE deve essere positivo.")
    if abs(fine_tune_factor - 1.0) > 1e-6:
        for group in param_groups:
            group['lr'] *= fine_tune_factor
        print(f"ℹ️ Scala i learning rate per il fine-tuning di un fattore {fine_tune_factor:.3f}.")

    schedule_step_mode = str(optim_cfg.get("SCHEDULER_STEP", "iteration")).lower()
    if schedule_step_mode not in {"iteration", "epoch"}:
        print(f"⚠️ Modalità scheduler '{schedule_step_mode}' non riconosciuta: uso 'iteration'.")
        schedule_step_mode = "iteration"

    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # === DATASET ===
    data_cfg = config["DATA"]
    # Stage 3 usa le immagini a piena risoluzione, ma serve mantenere la normalizzazione
    shared_transforms = build_transforms(data_cfg, is_train=False)

    DatasetClass = get_dataset(config["DATASET"])
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=shared_transforms,
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=shared_transforms,
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

    calibrate_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_joint,
    )

    # === TRAINING LOOP ===
    best_mae = float("inf")
    best_epoch = 0
    epochs_stage3 = optim_cfg["EPOCHS"]
    patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 0)))
    early_delta = max(0.0, float(optim_cfg.get("EARLY_STOPPING_DELTA", 0.0)))
    no_improve_rounds = 0

    print(f"🚀 Inizio Stage 3 — Fine-tuning congiunto per {epochs_stage3} epoche...")

    for epoch in range(epochs_stage3):
        print(f"\n--- Epoch {epoch + 1}/{epochs_stage3} ---")
        train_loss = train_one_epoch(
            model,
            criterion_zip,
            criterion_p2r,
            alpha,
            train_loader,
            optimizer,
            scheduler,
            schedule_step_mode,
            device,
            default_down,
            clamp_cfg=clamp_cfg,
        )

        if scheduler and schedule_step_mode == "epoch":
            scheduler.step()

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, default_down)
            bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")
            print(
                f"Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val MAE {val_mae:.2f} | "
                f"RMSE {val_rmse:.2f} | Pred/GT {bias:.3f}"
            )

            improvement_margin = best_mae - val_mae
            is_best = (improvement_margin > early_delta) or not np.isfinite(best_mae)
            best_candidate = best_mae

            if is_best:
                best_candidate = val_mae
                if config["EXP"].get("SAVE_BEST", False) and hasattr(model, "p2r_head"):
                    best_path = os.path.join(output_dir, "stage3_best.pth")
                    backup_log_scale = None
                    if hasattr(model.p2r_head, "log_scale"):
                        backup_log_scale = model.p2r_head.log_scale.detach().clone()
                        calibrate_density_scale(
                            model,
                            calibrate_loader,
                            device,
                            default_down,
                            max_batches=None,
                            clamp_range=clamp_cfg,
                            max_adjust=max_adjust,
                        )
                        cal_mae, cal_rmse, cal_tot_pred, cal_tot_gt = validate(
                            model, val_loader, device, default_down
                        )
                        cal_bias = (cal_tot_pred / cal_tot_gt) if cal_tot_gt > 0 else float("nan")
                        best_candidate = cal_mae
                        torch.save(model.state_dict(), best_path)
                        print(
                            f"💾 Nuovo best Stage 3 salvato ({best_path}) — "
                            f"MAE={cal_mae:.2f}, RMSE={cal_rmse:.2f}, Pred/GT={cal_bias:.3f}"
                        )
                        if backup_log_scale is not None:
                            model.p2r_head.log_scale.data.copy_(backup_log_scale)
                    else:
                        torch.save(model.state_dict(), best_path)
                        print(f"💾 Nuovo best Stage 3 salvato ({best_path}) — MAE={best_candidate:.2f}")

                best_mae = min(best_mae, best_candidate)
                best_epoch = epoch + 1
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if patience > 0 and no_improve_rounds >= patience:
                print(
                    f"⛔ Early stopping: nessun miglioramento MAE per {no_improve_rounds} "
                    "valutazioni consecutive."
                )
                break

    print(f"✅ Stage 3 completato con successo! Miglior MAE {best_mae:.2f} (epoch {best_epoch}).")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()