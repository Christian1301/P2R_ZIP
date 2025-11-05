# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 2 — P2R Training
# ============================================================

import os, yaml, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from losses.p2r_region_loss import P2RLoss
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    resume_if_exists, save_checkpoint, setup_experiment, collate_fn,
    canonicalize_p2r_grid,
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

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down, warn_tag="stage2_calib"
        )
        _, _, H_eff, W_eff = pred_density.shape
        down_h, down_w = down_tuple

        if H_eff == H_in and W_eff == W_in:
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        else:
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

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down, warn_tag="stage2_eval"
        )
        down_h, down_w = down_tuple
        H_out, W_out = pred_density.shape[-2:]

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
def main():
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio Stage 2 (P2R Training) su {device}")

    # --- Config ---
    optim_cfg = cfg["OPTIM_P2R"]
    data_cfg = cfg["DATA"]
    p2r_loss_cfg = cfg["P2R_LOSS"]
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)

    # --- Dataset + transforms ---
    train_transforms = build_transforms(data_cfg, is_train=True)
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
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
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

    # --- Congela ZIPHead e sblocca l'ultimo blocco del backbone ---
    print("🧊 Congelamento ZIPHead e sblocco parziale del backbone...")
    for p in model.zip_head.parameters():
        p.requires_grad = False

    # Fissa tutto il backbone e poi riattiva l'ultimo blocco conv (layer >=34)
    for p in model.backbone.parameters():
        p.requires_grad = False
    if hasattr(model.backbone, "body"):
        for idx, module in enumerate(model.backbone.body):
            if idx >= 34:
                for param in module.parameters():
                    param.requires_grad = True

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
    if head_params:
        params_to_train.append({'params': head_params, 'lr': optim_cfg['LR']})
    if log_scale_params:
        lr_factor = float(p2r_loss_cfg.get("LOG_SCALE_LR_FACTOR", 0.1))
        params_to_train.append({'params': log_scale_params, 'lr': optim_cfg['LR'] * lr_factor})
        print(f"ℹ️ LR log_scale ridotto di un fattore {lr_factor:.2f}")

    backbone_lr = optim_cfg.get("LR_BACKBONE")
    if backbone_lr is not None:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            params_to_train.append({'params': backbone_params, 'lr': float(backbone_lr)})
            print(f"ℹ️ Backbone fine-tuning attivo con LR {float(backbone_lr):.2e}")

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
    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
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

            B, _, H_out, W_out = pred_density.shape
            _, _, H_in, W_in = images.shape
            pred_density, down_tuple, _ = canonicalize_p2r_grid(
                pred_density, (H_in, W_in), default_down, warn_tag="stage2_train"
            )

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

            if ds_val is not None:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}", dens_scale=f"{ds_val:.6f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}")

        # --- Scheduler step ---
        if scheduler:
            scheduler.step()

        avg_train = total_loss / len(dl_train)
        if writer:
            writer.add_scalar("train/loss_p2r", avg_train, ep)
            writer.add_scalar("lr/p2r_head", opt.param_groups[0]["lr"], ep)
            if len(opt.param_groups) > 2:
                writer.add_scalar("lr/backbone", opt.param_groups[-1]["lr"], ep)

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
    main()
