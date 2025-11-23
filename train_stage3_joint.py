# -*- coding: utf-8 -*-
import argparse
import os
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
    save_checkpoint, canonicalize_p2r_grid, load_config
)
from train_stage2_p2r import calibrate_density_scale
import torch.nn.functional as F

def _round_up_8(x: int) -> int:
    return (x + 7) // 8 * 8

def collate_joint(batch):
    """
    Collate function per dataset che restituisce dizionari.
    Ogni elemento del batch è un dict con chiavi:
        'image', 'density', 'points'
    """
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
        pts  = [b.get("points", None) for b in batch]
    else:
        imgs, dens, pts = zip(*[(s[0], s[1], s[2]) for s in batch])

    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out, pts_out = [], [], []
    for im, den, p in zip(imgs, dens, pts):
        _, H, W = im.shape
        sy, sx = H_tgt / H, W_tgt / W

        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode='bilinear', align_corners=False).squeeze(0)

        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode='bilinear', align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)

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
    zip_scale: float = 1.0,
    count_l1_weight: float = 0.0,
    density_l1_weight: float = 0.0,
):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 3 (Joint)")

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        outputs = model(images)
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_train"
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        cell_area_tensor = pred_density.new_tensor(cell_area)
        gt_density = gt_density.to(dtype=pred_density.dtype)

        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)

        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area_tensor
        gt_counts = []
        for p in points:
            if p is None:
                gt_counts.append(0.0)
            else:
                gt_counts.append(float(p.shape[0]))
        gt_count = pred_density.new_tensor(gt_counts)
        count_l1 = torch.abs(pred_count - gt_count).mean()

        density_l1 = pred_density.new_tensor(0.0)
        if density_l1_weight > 0.0:
            _, _, h_gt, w_gt = gt_density.shape
            if (h_gt, w_gt) != pred_density.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_density,
                    size=pred_density.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                scale_ratio = (h_gt * w_gt) / (pred_density.shape[-2] * pred_density.shape[-1])
                gt_resized = gt_resized * scale_ratio
            else:
                gt_resized = gt_density
            density_l1 = F.l1_loss(pred_density, gt_resized)

        combined_loss = scaled_loss_zip + alpha * loss_p2r
        if count_l1_weight > 0.0:
            combined_loss = combined_loss + count_l1_weight * count_l1
        if density_l1_weight > 0.0:
            combined_loss = combined_loss + density_l1_weight * density_l1
        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler and schedule_step_mode == "iteration":
            scheduler.step()

        if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
            if len(clamp_cfg) != 2:
                raise ValueError("LOG_SCALE_CLAMP deve avere due valori [min, max].")
            min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
            dyn_clamp = getattr(model.p2r_head, "_dynamic_clamp", None)
            if dyn_clamp is not None and len(dyn_clamp) == 2:
                min_val = min(min_val, float(dyn_clamp[0]))
                max_val = max(max_val, float(dyn_clamp[1]))
            model.p2r_head.log_scale.data.clamp_(min_val, max_val)
            model.p2r_head._dynamic_clamp = (float(min_val), float(max_val))

        total_loss += combined_loss.item()
        current_lr = max(group['lr'] for group in optimizer.param_groups)
        postfix = {
            "total": f"{combined_loss.item():.4f}",
            "zip": f"{scaled_loss_zip.item():.4f}",
            "p2r": f"{loss_p2r.item():.4f}",
            "lr": f"{current_lr:.6f}"
        }
        if count_l1_weight > 0.0:
            postfix["cnt"] = f"{count_l1.item():.2f}"
        if density_l1_weight > 0.0:
            postfix["dens"] = f"{density_l1.item():.4f}"
        progress_bar.set_postfix(postfix)

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 3"):
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
        
        gt_count_list = [len(p) for p in points if p is not None]
        if not gt_count_list:
             gt_count_list = [0] 
        gt_count = torch.tensor(gt_count_list, dtype=torch.float32, device=device)

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse, total_pred, total_gt

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 3 (Joint)")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main(config_path: str):
    config = load_config(config_path)

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 3 (Joint ZIP + P2R) su {device}")

    default_down = config["DATA"].get("P2R_DOWNSAMPLE", 8)
    loss_cfg = config.get("P2R_LOSS", {})
    clamp_cfg = loss_cfg.get("LOG_SCALE_CLAMP")
    max_adjust = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")
    log_scale_recalibrate_thr = float(loss_cfg.get("LOG_SCALE_RECALIBRATE_THR", 0.0))
    calibrate_min_samples = loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_SAMPLES")
    calibrate_min_bias = loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_BIAS")
    calibrate_max_bias = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_BIAS")
    calibrate_dynamic_floor = loss_cfg.get("LOG_SCALE_DYNAMIC_FLOOR")

    dataset_name = config["DATASET"]
    model_cfg = config["MODEL"]
    bin_config = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_config["bins"], bin_config["bin_centers"]

    upsample_to_input = model_cfg.get("UPSAMPLE_TO_INPUT", False)
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
        backbone_name=model_cfg["BACKBONE"],
        pi_thresh=model_cfg.get("ZIP_PI_THRESH"),
        gate=model_cfg.get("GATE", "multiply"),
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
        soft_pi_gate=model_cfg.get("ZIP_PI_SOFT", False),
        pi_gate_power=model_cfg.get("ZIP_PI_SOFT_POWER", 1.0),
        pi_gate_min=model_cfg.get("ZIP_PI_SOFT_MIN", 0.0),
        apply_gate_to_output=model_cfg.get("ZIP_PI_APPLY_TO_P2R", False),
    ).to(device)

    optim_cfg = config["OPTIM_JOINT"]
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

    for p in model.parameters():
        p.requires_grad = True

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
    alpha = float(config["JOINT_LOSS"]["ALPHA"])
    zip_scale = float(config["JOINT_LOSS"].get("ZIP_SCALE", 1.0))
    count_l1_weight = float(
        config["JOINT_LOSS"].get(
            "P2R_COUNT_L1_W",
            config["JOINT_LOSS"].get("COUNT_L1_W", 0.0),
        )
    )
    density_l1_weight = float(config["JOINT_LOSS"].get("DENSITY_L1_W", 0.0))

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

    data_cfg = config["DATA"]
    stage3_crop = data_cfg.get(
        "CROP_SIZE_STAGE3",
        data_cfg.get("CROP_SIZE_STAGE2", data_cfg.get("CROP_SIZE", 256)),
    )
    stage3_scale = data_cfg.get(
        "CROP_SCALE_STAGE3",
        data_cfg.get("CROP_SCALE_STAGE2", data_cfg.get("CROP_SCALE", (0.3, 1.0))),
    )
    train_transforms = build_transforms(
        data_cfg,
        is_train=True,
        override_crop_size=stage3_crop,
        override_crop_scale=stage3_scale,
    )
    val_transforms = build_transforms(data_cfg, is_train=False)

    DatasetClass = get_dataset(config["DATASET"])
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_transforms,
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_joint,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_joint,
    )

    calibrate_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_joint,
    )

    calibrate_batches_cfg = loss_cfg.get("CALIBRATE_BATCHES", 10)
    calibrate_max_batches = None
    if calibrate_batches_cfg is not None:
        try:
            calibrate_batches_val = int(calibrate_batches_cfg)
            if calibrate_batches_val > 0:
                calibrate_max_batches = calibrate_batches_val
        except (TypeError, ValueError):
            calibrate_max_batches = None

    calibrate_trim = float(loss_cfg.get("LOG_SCALE_CALIBRATION_TRIM", 0.0))
    calibrate_stat = loss_cfg.get("LOG_SCALE_CALIBRATION_STAT", "median")

    if hasattr(model, "p2r_head") and hasattr(model.p2r_head, "log_scale"):
        print("🔧 Calibrazione iniziale Stage 3...")
        calibrate_density_scale(
            model,
            calibrate_loader,
            device,
            default_down,
            max_batches=calibrate_max_batches,
            clamp_range=clamp_cfg,
            max_adjust=max_adjust,
            min_samples=calibrate_min_samples,
            min_bias=calibrate_min_bias,
            max_bias=calibrate_max_bias,
            trim_ratio=calibrate_trim,
            stat=calibrate_stat,
            dynamic_floor=calibrate_dynamic_floor,
        )

    best_mae = float("inf")
    best_epoch = 0
    epochs_stage3 = optim_cfg["EPOCHS"]
    patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 0)))
    early_delta = max(0.0, float(optim_cfg.get("EARLY_STOPPING_DELTA", 0.0)))
    no_improve_rounds = 0
    bias_exceed_streak = 0

    if count_l1_weight > 0.0:
        print(f"ℹ️ Stage 3: penalità L1 sui conteggi attiva (peso={count_l1_weight:.3f}).")
    if density_l1_weight > 0.0:
        print(f"ℹ️ Stage 3: penalità L1 sulla densità attiva (peso={density_l1_weight:.3e}).")

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
            zip_scale=zip_scale,
            count_l1_weight=count_l1_weight,
            density_l1_weight=density_l1_weight,
        )

        if scheduler and schedule_step_mode == "epoch":
            scheduler.step()

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, default_down)
            bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")

            recalibrated = False
            bias_exceed = (
                log_scale_recalibrate_thr > 0.0
                and tot_gt > 0
                and np.isfinite(bias)
                and abs(bias - 1.0) > log_scale_recalibrate_thr
            )
            bias_exceed_streak = bias_exceed_streak + 1 if bias_exceed else 0

            if (
                bias_exceed
                and bias_exceed_streak >= 2
                and hasattr(model, "p2r_head")
                and hasattr(model.p2r_head, "log_scale")
            ):
                print(
                    "🔁 Stage 3: Pred/GT {:.3f} fuori soglia ±{:.3f}. Ricalibro log_scale prima di aggiornare il best.".format(
                        bias,
                        log_scale_recalibrate_thr,
                    )
                )
                recalib_bias = calibrate_density_scale(
                    model,
                    calibrate_loader,
                    device,
                    default_down,
                    max_batches=calibrate_max_batches,
                    clamp_range=clamp_cfg,
                    max_adjust=max_adjust,
                    min_samples=calibrate_min_samples,
                    min_bias=calibrate_min_bias,
                    max_bias=calibrate_max_bias,
                    trim_ratio=calibrate_trim,
                    stat=calibrate_stat,
                    dynamic_floor=calibrate_dynamic_floor,
                )
                if recalib_bias is not None:
                    print(f"   ↳ Bias stimato dopo ricalibrazione: {recalib_bias:.3f}")
                val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, default_down)
                bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")
                recalibrated = True
                bias_exceed_streak = 0

            recalib_note = " (ricalibrato)" if recalibrated else ""
            print(
                f"Epoch {epoch + 1}: Train Loss {train_loss:.4f} | Val MAE {val_mae:.2f} | "
                f"RMSE {val_rmse:.2f} | Pred/GT {bias:.3f}{recalib_note}"
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
                            min_samples=calibrate_min_samples,
                            min_bias=calibrate_min_bias,
                            max_bias=calibrate_max_bias,
                            trim_ratio=calibrate_trim,
                            stat=calibrate_stat,
                            dynamic_floor=calibrate_dynamic_floor,
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

if __name__ == "__main__":
    args = parse_args()
    main(args.config)