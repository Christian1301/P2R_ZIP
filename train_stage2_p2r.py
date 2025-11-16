# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 2 — P2R Training
# ============================================================

import argparse
import math
import os, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from losses.p2r_region_loss import P2RLoss
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    resume_if_exists,
    save_checkpoint,
    setup_experiment,
    collate_fn,
    load_config,
)

@torch.no_grad()
def calibrate_density_scale(
    model,
    loader,
    device,
    default_down,
    max_batches=8,
    clamp_range=None,
    max_adjust=None,
    bias_eps=1e-3,
):
    """
    Stima il bias iniziale del conteggio P2R e aggiusta log_scale di conseguenza
    prima dell'addestramento vero e proprio. In questo modo il training parte già
    con conteggi vicini al ground truth anche se Stage 1 non è perfetto.
    """
    if not hasattr(model, "p2r_head") or not hasattr(model.p2r_head, "log_scale"):
        return None

    model.eval()
    total_pred, total_gt = 0.0, 0.0

    for batch_idx, (images, _, points) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = images.to(device)
        points_list = [p.to(device) for p in points]

        outputs = model(images)
        pred_density = outputs.get("p2r_density", outputs.get("density"))
        if pred_density is None:
            continue

        B, _, H_out, W_out = pred_density.shape
        _, _, H_in, W_in = images.shape

        if H_out == H_in and W_out == W_in:
            down_h = down_w = float(default_down)
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        else:
            down_h = H_in / max(H_out, 1)
            down_w = W_in / max(W_out, 1)
            cell_area = down_h * down_w
            pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area

        gt_count = torch.tensor([len(p) for p in points_list], dtype=torch.float32, device=device)

        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    if total_gt <= 0 or total_pred <= 0:
        print("ℹ️ Calibrazione densità P2R saltata: conteggi non positivi.")
        return None

    bias = total_pred / total_gt
    if not np.isfinite(bias) or bias <= 0:
        print("ℹ️ Calibrazione densità P2R saltata: bias non finito.")
        return None
    if abs(bias - 1.0) < bias_eps:
        print(f"ℹ️ Calibrazione densità P2R: bias già unitario ({bias:.3f}).")
        return bias

    prev_log_scale = float(model.p2r_head.log_scale.detach().item())
    raw_adjust = float(np.log(bias))
    adjust = raw_adjust
    clipped_adjust = False
    if max_adjust is not None:
        max_adjust_val = float(max_adjust)
        if max_adjust_val <= 0:
            raise ValueError("LOG_SCALE_CALIBRATION_MAX_DELTA deve essere positivo.")
        adjust = float(np.clip(adjust, -max_adjust_val, max_adjust_val))
        clipped_adjust = abs(adjust - raw_adjust) > 1e-6
    adjust_tensor = torch.tensor(adjust, device=device, dtype=model.p2r_head.log_scale.dtype)
    model.p2r_head.log_scale.data -= adjust_tensor
    if clamp_range is not None:
        if len(clamp_range) != 2:
            raise ValueError("LOG_SCALE_CLAMP deve avere due valori [min, max].")
        min_val, max_val = float(clamp_range[0]), float(clamp_range[1])
        model.p2r_head.log_scale.data.clamp_(min_val, max_val)
    new_log_scale = float(model.p2r_head.log_scale.detach().item())
    new_scale = torch.exp(model.p2r_head.log_scale.detach()).item()
    print(
        "🔧 Calibrazione densità P2R: bias={:.3f} → log_scale {:.4f}→{:.4f} "
        "(scala={:.4f})".format(bias, prev_log_scale, new_log_scale, new_scale)
    )
    if clipped_adjust:
        print("ℹ️ Calibrazione limitata da LOG_SCALE_CALIBRATION_MAX_DELTA: valutare un delta più alto se necessario.")
    if clamp_range is not None:
        min_val, max_val = float(clamp_range[0]), float(clamp_range[1])
        if abs(new_log_scale - min_val) < 1e-6 or abs(new_log_scale - max_val) < 1e-6:
            print("⚠️ Calibrazione limitata da LOG_SCALE_CLAMP: valuta di allargare il range se necessario.")

    return bias

# -----------------------------------------------------------
@torch.no_grad()
def evaluate_p2r(model, loader, loss_fn, device, cfg):
    """
    Valutazione P2R coerente con la P2RHead (densità ReLU/(upscale²)):
    - Applica correzione / (down^2)
    - Calcola MAE, RMSE e loss media
    - Mostra range e media delle mappe
    """
    model.eval()
    total_loss, mae_errors, mse_errors = 0.0, 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0
    num_samples = 0

    default_down = cfg["DATA"].get("P2R_DOWNSAMPLE", 8)

    for images, _, points in tqdm(loader, desc="[Validating Stage 2]"):
        images = images.to(device)
        points_list = [p.to(device) for p in points]

        # --- Forward ---
        out = model(images)
        pred_density = out.get("p2r_density", out.get("density"))
        if pred_density is None:
            raise KeyError("Output 'p2r_density' o 'density' non trovato nel modello.")

        B, _, H_out, W_out = pred_density.shape
        _, _, H_in, W_in = images.shape

        if H_out == H_in and W_out == W_in:
            down_h = down_w = float(default_down)
        else:
            down_h = H_in / max(H_out, 1)
            down_w = W_in / max(W_out, 1)

        # Avvisa solo alla prima anomalia, senza interrompere il training
        resid_h = abs(H_in - down_h * H_out)
        resid_w = abs(W_in - down_w * W_out)
        if (resid_h > 1.0 or resid_w > 1.0) and not hasattr(evaluate_p2r, "_shape_warned"):
            print(f"⚠️ Downsample non intero rilevato: input {H_in}x{W_in}, output {H_out}x{W_out}, "
                  f"down_h={down_h:.4f}, down_w={down_w:.4f}")
            evaluate_p2r._shape_warned = True

        down_tuple = (down_h, down_w)

        # --- Loss (uguale al training) ---
        loss = loss_fn(pred_density, points_list, down=down_tuple)
        total_loss += loss.item()

        if H_out == H_in and W_out == W_in:
            # Output upsamplato fino all'input → densità già normalizzata in P2RHead
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        else:
            # Output ridotto → riporta la somma alla scala dei conteggi
            cell_area = down_h * down_w
            pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        gt_count = torch.tensor([len(p) for p in points_list], dtype=torch.float32, device=device)

        # --- Debug solo la prima volta ---
        if not hasattr(evaluate_p2r, "_debug_done"):
            print("===== DEBUG STAGE 2 =====")
            print(f"Input size: {H_in}x{W_in}")
            print(f"Output size: {H_out}x{W_out}")
            print(f"Downsampling factor: ({down_h:.2f}x, {down_w:.2f}x)")
            print(f"Density map range: [{pred_density.min().item():.4f}, {pred_density.max().item():.4f}]")
            print(f"Mean density: {pred_density.mean().item():.6f}")
            print(f"[DEBUG] Pred count (scaled): {pred_count[0].item():.2f}, GT count: {gt_count[0].item():.2f}")
            print("=========================")
            evaluate_p2r._debug_done = True

        abs_diff = torch.abs(pred_count - gt_count)
        mae_errors += abs_diff.sum().item()
        mse_errors += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()
        num_samples += len(points_list)

    # --- Medie ---
    avg_loss = total_loss / max(len(loader), 1)
    mae = mae_errors / max(num_samples, 1)
    rmse = np.sqrt(mse_errors / max(num_samples, 1))

    print("\n===== RISULTATI FINALI STAGE 2 =====")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    if total_gt > 0:
        bias = total_pred / total_gt
        print(f"Pred / GT ratio: {bias:.3f} (tot_pred={total_pred:.1f}, tot_gt={total_gt:.1f})")
    print("=====================================\n")

    return avg_loss, mae, rmse, total_pred, total_gt


# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 (P2R)")
    parser.add_argument("--config", default="config.yaml", help="Percorso al file di configurazione YAML")
    return parser.parse_args()


def main(config_path: str):
    cfg = load_config(config_path)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio Stage 2 (P2R Training) su {device}")

    # --- Config ---
    optim_cfg = cfg["OPTIM_P2R"]
    data_cfg = cfg["DATA"]
    p2r_loss_cfg = cfg["P2R_LOSS"]
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)

    # --- Dataset + transforms ---
    stage2_crop = data_cfg.get("CROP_SIZE_STAGE2", data_cfg.get("CROP_SIZE", 256))
    stage2_scale = data_cfg.get("CROP_SCALE_STAGE2", data_cfg.get("CROP_SCALE", (0.3, 1.0)))
    train_transforms = build_transforms(
        data_cfg,
        is_train=True,
        override_crop_size=stage2_crop,
        override_crop_scale=stage2_scale,
    )
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    train_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_transforms
    )
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms
    )

    dl_train = DataLoader(
        train_ds, batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True, num_workers=optim_cfg["NUM_WORKERS"],
        drop_last=True, collate_fn=collate_fn, pin_memory=True
    )
    dl_val = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )

    # --- Modello ---
    dataset_name = cfg["DATASET"]
    bin_config = cfg["BINS_CONFIG"][dataset_name]
    model_cfg = cfg["MODEL"]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    pi_mode_stage2 = model_cfg.get("ZIP_PI_MODE_STAGE2", model_cfg.get("ZIP_PI_MODE", "hard"))
    pi_thresh_target = float(model_cfg.get("ZIP_PI_THRESH_STAGE2", model_cfg.get("ZIP_PI_THRESH", 0.15)))
    pi_floor_stage2 = model_cfg.get("ZIP_PI_FLOOR_STAGE2", model_cfg.get("ZIP_PI_FLOOR"))
    lambda_clamp_stage2 = model_cfg.get("ZIP_LAMBDA_CLAMP_STAGE2", model_cfg.get("ZIP_LAMBDA_CLAMP"))
    pi_thresh_start = float(model_cfg.get("ZIP_PI_THRESH_STAGE2_START", model_cfg.get("ZIP_PI_THRESH_START", pi_thresh_target)))
    pi_thresh_warm_epochs = int(model_cfg.get("ZIP_PI_THRESH_STAGE2_WARMUP_EPOCHS", model_cfg.get("ZIP_PI_THRESH_WARMUP_EPOCHS", 0)) or 0)
    pi_thresh_schedule_enabled = pi_thresh_warm_epochs > 0 and abs(pi_thresh_target - pi_thresh_start) > 1e-6
    if pi_thresh_schedule_enabled:
        print(
            f"ℹ️ Stage 2: annealing ZIP_PI_THRESH da {pi_thresh_start:.3f} a {pi_thresh_target:.3f} "
            f"in {pi_thresh_warm_epochs} epoche."
        )

    def compute_pi_thresh(epoch: int) -> float:
        if not pi_thresh_schedule_enabled:
            return float(pi_thresh_target)
        progress = (epoch - 1) / max(pi_thresh_warm_epochs - 1, 1)
        progress = max(0.0, min(1.0, progress))
        return float(pi_thresh_start + progress * (pi_thresh_target - pi_thresh_start))

    initial_pi_thresh = pi_thresh_start if pi_thresh_schedule_enabled else pi_thresh_target

    model = P2R_ZIP_Model(
        backbone_name=model_cfg["BACKBONE"],
        pi_thresh=initial_pi_thresh,
        gate=model_cfg["GATE"],
        pi_mode=pi_mode_stage2,
        pi_floor=pi_floor_stage2,
        lambda_clamp=lambda_clamp_stage2,
        upsample_to_input=False,
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # --- Carica Stage 1 ---
    stage1_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    zip_ckpt = os.path.join(stage1_dir, "best_model.pth")
    if not os.path.isfile(zip_ckpt):
        print(f"❌ Checkpoint Stage1 non trovato in {zip_ckpt}.")
        return
    state_dict = torch.load(zip_ckpt, map_location=device)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print(f"✅ Checkpoint Stage1 caricato da {zip_ckpt}")

    # --- Congela ZIPHead + backbone (default Stage 2) ---
    finetune_backbone = bool(optim_cfg.get("FINETUNE_BACKBONE", False))

    print("🧊 Congelo la ZIPHead.")
    for p in model.zip_head.parameters():
        p.requires_grad = False

    if finetune_backbone:
        print("🔓 Fine-tuning del backbone ATTIVO per Stage 2 (usa solo in casi particolari).")
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        print("🧊 Backbone congelato per tutto lo Stage 2 (comportamento consigliato).")
        for p in model.backbone.parameters():
            p.requires_grad = False

    for p in model.p2r_head.parameters():
        p.requires_grad = True  # P2RHead sempre addestrabile

    # Reinizializza il log_scale per contenere la scala iniziale se richiesto
    log_scale_init = p2r_loss_cfg.get("LOG_SCALE_INIT")
    if log_scale_init is not None and hasattr(model.p2r_head, "log_scale"):
        model.p2r_head.log_scale.data.fill_(float(log_scale_init))
        print(f"ℹ️ log_scale inizializzato a {float(log_scale_init):.3f}")

    # --- Ottimizzatore + Scheduler ---
    head_params, log_scale_params = [], []
    for name, param in model.p2r_head.named_parameters():
        if not param.requires_grad:
            continue
        if "log_scale" in name:
            log_scale_params.append(param)
        else:
            head_params.append(param)

    params_to_train = []
    backbone_group_idx = None
    if head_params:
        params_to_train.append({'params': head_params, 'lr': optim_cfg['LR']})
    if log_scale_params:
        lr_factor = float(p2r_loss_cfg.get("LOG_SCALE_LR_FACTOR", 0.1))
        params_to_train.append({'params': log_scale_params, 'lr': optim_cfg['LR'] * lr_factor})
        print(f"ℹ️ LR log_scale ridotto di un fattore {lr_factor:.2f}")

    backbone_lr = optim_cfg.get("LR_BACKBONE")
    if finetune_backbone:
        if backbone_lr is None:
            raise ValueError("FINETUNE_BACKBONE richiede di specificare LR_BACKBONE in OPTIM_P2R.")
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            backbone_group_idx = len(params_to_train)
            params_to_train.append({'params': backbone_params, 'lr': float(backbone_lr)})
            print(f"ℹ️ Backbone fine-tuning attivo con LR {float(backbone_lr):.2e}")
    else:
        if backbone_lr is not None:
            print("ℹ️ LR_BACKBONE impostato ma FINETUNE_BACKBONE=False: ignorato nello Stage 2.")

    opt = get_optimizer(params_to_train, optim_cfg)
    scheduler = get_scheduler(opt, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # --- Loss ---
    loss_kwargs = {}
    if "SCALE_WEIGHT" in p2r_loss_cfg:
        loss_kwargs["scale_weight"] = float(p2r_loss_cfg["SCALE_WEIGHT"])
    if "POS_WEIGHT" in p2r_loss_cfg:
        loss_kwargs["pos_weight"] = float(p2r_loss_cfg["POS_WEIGHT"])
    if "CHUNK_SIZE" in p2r_loss_cfg:
        loss_kwargs["chunk_size"] = int(p2r_loss_cfg["CHUNK_SIZE"])
    if "MIN_RADIUS" in p2r_loss_cfg:
        loss_kwargs["min_radius"] = float(p2r_loss_cfg["MIN_RADIUS"])
    if "MAX_RADIUS" in p2r_loss_cfg:
        loss_kwargs["max_radius"] = float(p2r_loss_cfg["MAX_RADIUS"])
    p2r_loss_fn = P2RLoss(**loss_kwargs).to(device)

    # --- Setup esperimento ---
    exp_dir = os.path.join(stage1_dir, "stage2")
    writer = setup_experiment(exp_dir)
    start_ep, best_mae = (1, float("inf"))
    if optim_cfg.get("RESUME_LAST", False):
        # Carica solo i pesi del modello, non l'optimizer
        last_ck = os.path.join(exp_dir, "last.pth")
        if os.path.isfile(last_ck):
            ckpt = torch.load(last_ck, map_location=device)
            try:
                model.load_state_dict(ckpt["model"])
            except RuntimeError as e:
                print(f"⚠️ Warning: caricamento parziale del modello — {e}")
                model.load_state_dict(ckpt["model"], strict=False)
            start_ep = ckpt.get("epoch", 0) + 1
            best_mae = ckpt.get("best_val", float("inf"))
            print(f"✅ Ripreso lo Stage 2 da {last_ck} (epoch={start_ep})")
        else:
            print("ℹ️ Nessun checkpoint precedente trovato, partenza da zero.")
            start_ep, best_mae = 1, float("inf")
    else:
        start_ep, best_mae = 1, float("inf")

    # --- Calibrazione iniziale del conteggio ---
    # Aggiorna la soglia di gating in base all'epoca di ripartenza
    current_scheduled_thresh = compute_pi_thresh(start_ep)
    model.pi_thresh = current_scheduled_thresh

    calibrate_batches_cfg = p2r_loss_cfg.get("CALIBRATE_BATCHES", 6)
    calibrate_max_batches = None
    if calibrate_batches_cfg is not None:
        try:
            calibrate_batches_val = int(calibrate_batches_cfg)
            if calibrate_batches_val > 0:
                calibrate_max_batches = calibrate_batches_val
        except (TypeError, ValueError):
            calibrate_max_batches = None
    clamp_cfg = p2r_loss_cfg.get("LOG_SCALE_CLAMP")
    max_adjust = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")
    if start_ep == 1:
        calibrate_density_scale(
            model,
            dl_val,
            device,
            default_down,
            max_batches=calibrate_max_batches,
            clamp_range=clamp_cfg,
            max_adjust=max_adjust,
        )

    # --- Training loop ---
    print(f"🚀 Inizio training Stage 2 per {optim_cfg['EPOCHS']} epoche...")
    prev_pi_thresh_value = None
    last_thresh_log = None
    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
        scheduled_thresh = compute_pi_thresh(ep)
        model.pi_thresh = scheduled_thresh
        threshold_changed = prev_pi_thresh_value is None or abs(scheduled_thresh - prev_pi_thresh_value) > 1e-6
        if pi_thresh_schedule_enabled and threshold_changed:
            log_interval = max(1, pi_thresh_warm_epochs // 5) if pi_thresh_warm_epochs > 0 else 1
            should_log = (
                last_thresh_log is None
                or ep == start_ep
                or ep >= pi_thresh_warm_epochs
                or (ep - start_ep) % log_interval == 0
            )
            if should_log:
                print(f"ℹ️ Epoch {ep}: ZIP_PI_THRESH impostato a {scheduled_thresh:.4f}")
                last_thresh_log = ep
        prev_pi_thresh_value = scheduled_thresh
        if writer:
            writer.add_scalar("p2r_head/pi_thresh", scheduled_thresh, ep)
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(dl_train, start=1), total=len(dl_train), desc=f"[P2R Train] Epoch {ep}/{optim_cfg['EPOCHS']}")

        for batch_idx, (images, _, points) in pbar:
            images = images.to(device)
            points_list = [p.to(device) for p in points]

            out = model(images)
            pred_density = out.get("p2r_density", out.get("density"))
            if pred_density is None:
                raise KeyError("Output 'p2r_density' o 'density' non trovato nel modello.")

            pi_mean_batch = float("nan")
            lambda_mean_batch = float("nan")
            logit_pi_batch = out.get("logit_pi_maps")
            if logit_pi_batch is not None:
                pi_vals = logit_pi_batch.softmax(dim=1)[:, 1:]
                if pi_vals.numel() > 0:
                    pi_mean_batch = float(pi_vals.mean().item())
            lambda_maps_batch = out.get("lambda_maps")
            if lambda_maps_batch is not None:
                if lambda_maps_batch.numel() > 0:
                    lambda_mean_batch = float(lambda_maps_batch.mean().item())

            B, _, H_out, W_out = pred_density.shape
            _, _, H_in, W_in = images.shape
            if H_out == H_in and W_out == W_in:
                down_h = down_w = float(default_down)
            else:
                down_h = H_in / max(H_out, 1)
                down_w = W_in / max(W_out, 1)
            down_tuple = (down_h, down_w)

            # ✅ ORA loss coerente con ReLU/(upscale²) — niente più logit
            loss = p2r_loss_fn(pred_density, points_list, down=down_tuple)

            opt.zero_grad()
            loss.backward()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            total_loss += loss.item()

            ds_val = None
            with torch.no_grad():
                if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
                    if len(clamp_cfg) != 2:
                        raise ValueError("LOG_SCALE_CLAMP deve essere una lista o tupla di due valori [min, max].")
                    min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
                    model.p2r_head.log_scale.data.clamp_(min_val, max_val)
                if hasattr(model.p2r_head, "log_scale"):
                    ds_val = float(torch.exp(model.p2r_head.log_scale.detach()).item())
                    if writer:
                        global_step = (ep - 1) * len(dl_train) + batch_idx
                        writer.add_scalar("p2r_head/density_scale", ds_val, global_step)
                        writer.add_scalar("p2r_head/log_scale", float(model.p2r_head.log_scale.detach().item()), global_step)

            if writer:
                global_step = (ep - 1) * len(dl_train) + batch_idx
                if not math.isnan(pi_mean_batch):
                    writer.add_scalar("zip/pi_mean", pi_mean_batch, global_step)
                if not math.isnan(lambda_mean_batch):
                    writer.add_scalar("zip/lambda_mean", lambda_mean_batch, global_step)

            postfix = {
                "loss": f"{loss.item():.4f}",
                "lr": f"{opt.param_groups[0]['lr']:.6f}",
            }
            if not math.isnan(pi_mean_batch):
                postfix["pi"] = f"{pi_mean_batch:.3f}"
            if not math.isnan(lambda_mean_batch):
                postfix["lam"] = f"{lambda_mean_batch:.3f}"
            if ds_val is not None:
                postfix["dens_scale"] = f"{ds_val:.6f}"
            pbar.set_postfix(postfix)

        # --- Scheduler step ---
        if scheduler:
            scheduler.step()

        avg_train = total_loss / len(dl_train)
        if writer:
            writer.add_scalar("train/loss_p2r", avg_train, ep)
            writer.add_scalar("lr/p2r_head", opt.param_groups[0]["lr"], ep)
        if writer and backbone_group_idx is not None:
            writer.add_scalar("lr/backbone", opt.param_groups[backbone_group_idx]["lr"], ep)

        # --- Validazione periodica ---
        if ep % optim_cfg["VAL_INTERVAL"] == 0 or ep == optim_cfg["EPOCHS"]:
            val_loss, mae, mse, tot_pred, tot_gt = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
            orig_bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")
            if writer:
                writer.add_scalar("val/loss_p2r", val_loss, ep)
                writer.add_scalar("val/MAE", mae, ep)
                writer.add_scalar("val/MSE", mse, ep)
                if tot_gt > 0:
                    writer.add_scalar("val/pred_gt_ratio", orig_bias, ep)

            # --- Best model ---
            is_best = mae < best_mae
            best_candidate = best_mae
            calibrated_metrics = None

            if is_best:
                best_candidate = mae
                if cfg["EXP"].get("SAVE_BEST", False):
                    best_path = os.path.join(stage1_dir, "stage2_best.pth")
                    saved_with_calibration = False
                    if hasattr(model.p2r_head, "log_scale"):
                        backup_log_scale = model.p2r_head.log_scale.detach().clone()
                        calibrate_density_scale(
                            model,
                            dl_val,
                            device,
                            default_down,
                            max_batches=None,
                            clamp_range=clamp_cfg,
                            max_adjust=max_adjust,
                        )
                        calibrated_metrics = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
                        cal_loss, cal_mae, cal_rmse, cal_tot_pred, cal_tot_gt = calibrated_metrics
                        best_candidate = cal_mae
                        torch.save(model.state_dict(), best_path)
                        cal_bias = (cal_tot_pred / cal_tot_gt) if cal_tot_gt > 0 else float("nan")
                        print(
                            f"💾 Nuovo best Stage 2 salvato in {best_path} "
                            f"(MAE={cal_mae:.2f}, Pred/GT={cal_bias:.3f})"
                        )
                        if writer:
                            writer.add_scalar("val/loss_p2r_calibrated", cal_loss, ep)
                            writer.add_scalar("val/MAE_calibrated", cal_mae, ep)
                            writer.add_scalar("val/MSE_calibrated", cal_rmse, ep)
                            if cal_tot_gt > 0:
                                writer.add_scalar("val/pred_gt_ratio_calibrated", cal_bias, ep)
                        model.p2r_head.log_scale.data.copy_(backup_log_scale)
                        saved_with_calibration = True
                    if not saved_with_calibration:
                        torch.save(model.state_dict(), best_path)
                        print(f"💾 Nuovo best Stage 2 salvato in {best_path} (MAE={mae:.2f})")

            # Aggiorna il tracker del best MAE con il candidato (calibrato se disponibile)
            if is_best:
                best_mae = min(best_mae, best_candidate)

            save_checkpoint(model, opt, ep, mae, best_mae, exp_dir, is_best=is_best)

            print(
                f"Epoch {ep}: Train Loss {avg_train:.4f} | Val Loss {val_loss:.4f} | MAE {mae:.2f} | "
                f"MSE {mse:.2f} | Pred/GT {orig_bias:.3f} | Best MAE {best_mae:.2f}"
            )
            if calibrated_metrics is not None:
                cal_loss, cal_mae, cal_rmse, cal_tot_pred, cal_tot_gt = calibrated_metrics
                cal_bias = (cal_tot_pred / cal_tot_gt) if cal_tot_gt > 0 else float("nan")
                print(
                    "         ↳ dopo calibrazione best: Val Loss {:.4f} | MAE {:.2f} | MSE {:.2f} | Pred/GT {:.3f}".format(
                        cal_loss, cal_mae, cal_rmse, cal_bias
                    )
                )

            # --- Debug ogni 50 epoche ---
            if ep % 50 == 0:
                with torch.no_grad():
                    sample_out = model(images[:1])
                    dens = sample_out["p2r_density"]
                    print(f"[DEBUG Epoch {ep}] Mean dens: {dens.mean().item():.4f}, Max: {dens.max().item():.4f}")

    if writer:
        writer.close()
    print("✅ Stage 2 completato.")


# -----------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args.config)
