# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 2 — P2R Training
# ============================================================

import os, numpy as np, torch, argparse
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from losses.p2r_region_loss import P2RLoss
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    resume_if_exists, save_checkpoint, setup_experiment, collate_fn, load_config
)


def _split_loader_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Formato batch inatteso: {type(batch)}")

    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    if len(batch) == 3:
        images, densities, points = batch
        return images, densities, points, None

    raise ValueError(f"Batch collate inatteso (len={len(batch)}): {batch}")


def _collect_dense_indices(dataset, min_points, limit=None):
    """Restituisce gli indici delle immagini con almeno ``min_points`` annotazioni."""
    dense_indices = []
    total = len(dataset.image_list)
    max_items = None if limit is None else int(max(1, limit))
    for idx in range(total):
        try:
            pts = dataset.load_points(dataset.image_list[idx])
            num_pts = 0 if pts is None else int(pts.shape[0])
        except Exception as exc:
            print(f"⚠️ Impossibile calcolare il conteggio punti per index={idx}: {exc}")
            continue
        if num_pts >= min_points:
            dense_indices.append(idx)
            if max_items is not None and len(dense_indices) >= max_items:
                break
    return dense_indices

@torch.no_grad()
def calibrate_density_scale(
    model,
    loader,
    device,
    default_down,
    max_batches=8,
    clamp_range=None,
    max_adjust=None,
    min_samples=None,
    min_bias=None,
    max_bias=None,
    bias_eps=1e-3,
    trim_ratio=0.0,
    stat="median",
    dynamic_floor=None,
    adjust_damping=1.0,
):
    """
    Stima il bias iniziale del conteggio P2R e aggiusta log_scale di conseguenza
    prima dell'addestramento vero e proprio. In questo modo il training parte già
    con conteggi vicini al ground truth anche se Stage 1 non è perfetto.
    """
    if not hasattr(model, "p2r_head") or not hasattr(model.p2r_head, "log_scale"):
        return None

    model.eval()
    min_samples_int = None
    if min_samples is not None:
        try:
            min_samples_int = max(int(min_samples), 1)
        except (TypeError, ValueError):
            min_samples_int = None

    effective_clamp = None
    if clamp_range is not None:
        if len(clamp_range) != 2:
            raise ValueError("LOG_SCALE_CLAMP deve avere due valori [min, max].")
        min_val = float(clamp_range[0])
        max_val = float(clamp_range[1])
        dyn_clamp = getattr(model.p2r_head, "_dynamic_clamp", None)
        if dyn_clamp is not None and len(dyn_clamp) == 2:
            min_val = min(min_val, float(dyn_clamp[0]))
            max_val = max(max_val, float(dyn_clamp[1]))
        effective_clamp = [min_val, max_val]
    else:
        dyn_clamp = getattr(model.p2r_head, "_dynamic_clamp", None)
        if dyn_clamp is not None and len(dyn_clamp) == 2:
            effective_clamp = [float(dyn_clamp[0]), float(dyn_clamp[1])]

    total_pred, total_gt = 0.0, 0.0
    per_sample_biases = []
    per_sample_weights = []
    warned_extra_batches = False

    for batch_idx, batch in enumerate(loader, start=1):

        images, _, points, _ = _split_loader_batch(batch)

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

        with torch.no_grad():
            for sample_idx in range(len(points_list)):
                gt_val = float(gt_count[sample_idx].item())
                pred_val = float(pred_count[sample_idx].item())
                if gt_val <= 0.0 or pred_val <= 0.0:
                    continue
                per_sample_biases.append(pred_val / max(gt_val, 1e-6))
                per_sample_weights.append(gt_val)

        if max_batches is not None and batch_idx >= max_batches:
            if min_samples_int is None or len(per_sample_biases) >= min_samples_int:
                break
            if not warned_extra_batches:
                print(
                    "ℹ️ Calibrazione densità P2R: raccolgo batch aggiuntivi oltre il limite configurato per "
                    "raggiungere i campioni minimi (attesi {}, attuali {}).".format(
                        min_samples_int,
                        len(per_sample_biases),
                    )
                )
                warned_extra_batches = True

    if not per_sample_biases:
        print("ℹ️ Calibrazione densità P2R saltata: bias per-sample non disponibile.")
        return None

    trim_ratio = float(max(0.0, min(trim_ratio, 0.45)))
    num_samples = len(per_sample_biases)
    bias_array = np.array(per_sample_biases, dtype=np.float32)
    weight_array = np.array(per_sample_weights, dtype=np.float32)
    order = np.argsort(bias_array)
    sorted_biases = bias_array[order]
    sorted_weights = weight_array[order]
    trimmed_biases = sorted_biases
    trimmed_weights = sorted_weights
    fallback_trim = False
    did_trim = False
    if trim_ratio > 0.0 and sorted_biases.size >= 3:
        trim = int(sorted_biases.size * trim_ratio)
        if trim > 0 and trim * 2 < sorted_biases.size:
            low_idx = trim
            high_idx = sorted_biases.size - trim
            trimmed_biases = sorted_biases[low_idx:high_idx]
            trimmed_weights = sorted_weights[low_idx:high_idx]
            did_trim = True
    used_samples = int(trimmed_biases.size)

    if min_samples_int is not None and used_samples < max(min_samples_int, 1):
        if did_trim:
            # Riprova senza trimming per recuperare campioni utili.
            trimmed_biases = sorted_biases
            trimmed_weights = sorted_weights
            used_samples = int(trimmed_biases.size)
            fallback_trim = True
            did_trim = False

    if min_samples_int is not None and used_samples < max(min_samples_int, 1):
        print(
            "ℹ️ Calibrazione densità P2R saltata: campioni utili {} insufficienti (minimo richiesto {}).".format(
                used_samples,
                min_samples_int,
            )
        )
        return None

    if fallback_trim:
        print(
            "ℹ️ Calibrazione densità P2R: trimming disattivato per preservare {} campioni utili.".format(
                used_samples
            )
        )

    if trimmed_biases.size == 0:
        print("ℹ️ Calibrazione densità P2R saltata: dopo il trimming non restano campioni validi.")
        return None

    stat_choice = (stat or "median").lower()
    sample_stat_bias = float(np.mean(trimmed_biases)) if stat_choice == "mean" else float(np.median(trimmed_biases))

    weighted_bias = None
    if trimmed_biases.size > 0:
        weight_sum = float(trimmed_weights.sum())
        if weight_sum > 1e-6:
            weighted_bias = float(np.sum(trimmed_biases * trimmed_weights) / weight_sum)
        else:
            weighted_bias = float(np.mean(trimmed_biases))

    global_bias = None
    if total_gt > 0 and total_pred > 0:
        global_bias = float(total_pred / total_gt)
        if not np.isfinite(global_bias) or global_bias <= 0.0:
            global_bias = None

    bias = None
    bias_source = "mediana"
    if global_bias is not None:
        bias = global_bias
        bias_source = "totale"
    elif weighted_bias is not None and np.isfinite(weighted_bias) and weighted_bias > 0.0:
        bias = weighted_bias
        bias_source = "pesato"
    else:
        bias = sample_stat_bias
        bias_source = stat_choice

    if (
        max_batches is not None
        and total_gt > 0.0
        and np.isfinite(total_pred)
        and (total_pred < 0.05 * total_gt or total_pred > 5.0 * total_gt)
    ):
        print(
            "ℹ️ Calibrazione densità P2R saltata: massa predetta {:.3f} rispetto al GT {:.3f} "
            "(bias stimato {:.3f}) non è ancora affidabile.".format(
                total_pred,
                total_gt,
                bias,
            )
        )
        return None

    clip_notes = []
    if min_bias is not None:
        try:
            min_bias_val = float(min_bias)
            if bias < min_bias_val:
                bias = min_bias_val
                clip_notes.append(f"min_bias={min_bias_val:.3f}")
        except (TypeError, ValueError):
            pass
    if max_bias is not None:
        try:
            max_bias_val = float(max_bias)
            if bias > max_bias_val:
                bias = max_bias_val
                clip_notes.append(f"max_bias={max_bias_val:.3f}")
        except (TypeError, ValueError):
            pass

    if not np.isfinite(bias) or bias <= 0:
        print("ℹ️ Calibrazione densità P2R saltata: bias non finito.")
        return None
    if abs(bias - 1.0) < bias_eps:
        print(
            "ℹ️ Calibrazione densità P2R: bias già unitario ({:.3f}) [totale={}, pesato={}, {}={}].".format(
                bias,
                "n/a" if global_bias is None else f"{global_bias:.3f}",
                "n/a" if weighted_bias is None else f"{weighted_bias:.3f}",
                stat_choice,
                f"{sample_stat_bias:.3f}" if trimmed_biases.size > 0 else "n/a",
            )
        )
        return bias

    if total_gt <= 0 or total_pred <= 0:
        print("ℹ️ Calibrazione densità P2R saltata: conteggi non positivi.")
        return None

    prev_log_scale = float(model.p2r_head.log_scale.detach().item())
    raw_adjust = float(np.log(bias))
    adjust = raw_adjust
    try:
        damping = float(adjust_damping)
    except (TypeError, ValueError):
        damping = 1.0
    damping = min(max(damping, 0.05), 1.0)
    adjust *= damping
    clipped_adjust = False
    applied_max_delta = None
    if max_adjust is not None:
        applied_max_delta = float(max_adjust)
        if applied_max_delta <= 0:
            raise ValueError("LOG_SCALE_CALIBRATION_MAX_DELTA deve essere positivo.")
        pre_clip_adjust = adjust
        adjust = float(np.clip(pre_clip_adjust, -applied_max_delta, applied_max_delta))
        clipped_adjust = abs(adjust - pre_clip_adjust) > 1e-6
    target_log_scale = prev_log_scale - adjust
    model.p2r_head.log_scale.data.fill_(target_log_scale)
    if effective_clamp is not None:
        min_val, max_val = effective_clamp
        model.p2r_head.log_scale.data.clamp_(min_val, max_val)
    new_log_scale = float(model.p2r_head.log_scale.detach().item())
    new_scale = torch.exp(model.p2r_head.log_scale.detach()).item()
    global_str = "n/a" if global_bias is None else f"{global_bias:.3f}"
    weighted_str = "n/a" if weighted_bias is None else f"{weighted_bias:.3f}"
    stat_str = "n/a" if trimmed_biases.size == 0 else f"{sample_stat_bias:.3f}"
    print(
        "ℹ️ Calibrazione densità P2R: stime bias → totale={}, pesato={}, {}={}, selezionato={}.".format(
            global_str,
            weighted_str,
            stat_choice,
            stat_str,
            bias_source,
        )
    )

    clip_info = ""
    clip_parts = []
    if clip_notes:
        clip_parts.append("bias limitato da {}".format(", ".join(clip_notes)))
    if abs(new_log_scale - target_log_scale) > 1e-6:
        clip_parts.append(f"target_clipped({target_log_scale:.4f})")
    clip_parts.append(f"fonte={bias_source}")
    if damping < 0.999:
        clip_parts.append(f"damping={damping:.2f}")
    if clip_parts:
        clip_info = " [" + "; ".join(clip_parts) + "]"

    print(
        "🔧 Calibrazione densità P2R: bias={:.3f} ({}/{}, trim={:.2f}, stat={}) → log_scale {:.4f}→{:.4f} "
        "(target={:.4f}, scala={:.4f}){}".format(
            bias,
            used_samples,
            num_samples,
            0.0 if fallback_trim else trim_ratio,
            stat_choice,
            prev_log_scale,
            new_log_scale,
            target_log_scale,
            new_scale,
            clip_info,
        )
    )
    if clipped_adjust:
        delta_note = "" if applied_max_delta is None else f" (delta_max={applied_max_delta:.2f})"
        print(
            "ℹ️ Calibrazione limitata da LOG_SCALE_CALIBRATION_MAX_DELTA{}: valutare un delta più alto se necessario.".format(
                delta_note
            )
        )
    clamp_expanded = False
    try:
        clamp_floor = float(dynamic_floor) if dynamic_floor is not None else -9.0
    except (TypeError, ValueError):
        clamp_floor = -9.0
    clamp_floor = min(clamp_floor, -1e-3)
    if effective_clamp is not None:
        min_val, max_val = effective_clamp
        # Consente alla dinamica di ridurre il clamp inferiore di almeno 0.5 senza scendere oltre clamp_floor
        if clamp_floor >= min_val:
            clamp_floor = min(min_val - 0.5, clamp_floor)
        boundary_hit = abs(new_log_scale - min_val) < 1e-6 or abs(new_log_scale - max_val) < 1e-6
        if boundary_hit and abs(bias - 1.0) > bias_eps:
            expand_amount = 0.5
            if abs(new_log_scale - min_val) < 1e-6 and bias > 1.0:
                next_min = max(min_val - expand_amount, clamp_floor)
                if next_min < min_val - 1e-6:
                    min_val = next_min
                    clamp_expanded = True
            elif abs(new_log_scale - max_val) < 1e-6 and bias < 1.0:
                max_val += expand_amount
                clamp_expanded = True
            if clamp_expanded:
                desired = float(np.clip(target_log_scale, min_val, max_val))
                model.p2r_head.log_scale.data.fill_(desired)
                new_log_scale = float(model.p2r_head.log_scale.detach().item())
                new_scale = torch.exp(model.p2r_head.log_scale.detach()).item()
                effective_clamp[0], effective_clamp[1] = min_val, max_val
                direction = "inferiore" if bias > 1.0 else "superiore"
                print(
                    "ℹ️ Calibrazione: clamp {} espanso automaticamente a [{:.4f}, {:.4f}] per ridurre il bias residuo.".format(
                        direction,
                        min_val,
                        max_val,
                    )
                )
                if min_val <= clamp_floor + 1e-6 and bias > 1.0:
                    print(
                        "ℹ️ Clamp inferiore fissato a {:.2f}: ulteriori espansioni bloccate per evitare scale eccessivamente piccole.".format(
                            clamp_floor
                        )
                    )
        if abs(new_log_scale - min_val) < 1e-6 or abs(new_log_scale - max_val) < 1e-6:
            print("⚠️ Calibrazione limitata da LOG_SCALE_CLAMP: valuta di allargare il range se necessario.")
        model.p2r_head._dynamic_clamp = (float(min_val), float(max_val))

    if effective_clamp is not None and clamp_expanded:
        print(
            "ℹ️ Nuova scala densità dopo espansione clamp: log_scale={:.4f}, scala={:.4f}".format(
                new_log_scale,
                new_scale,
            )
        )

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
    diag_cfg = cfg.get("P2R_DIAGNOSTICS", {})
    diag_enabled = bool(diag_cfg.get("ENABLE", False))
    diag_thresh = float(diag_cfg.get("COUNT_ERR_THRESHOLD", 0.0))
    diag_topk = max(1, int(diag_cfg.get("TOPK", 5)))
    diag_records = []

    for batch in tqdm(loader, desc="[Validating Stage 2]"):
        images, _, points, meta = _split_loader_batch(batch)
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

        if diag_enabled and meta is not None:
            meta_batch = meta if isinstance(meta, list) else [meta]
            for sample_idx in range(len(points_list)):
                err_val = abs_diff[sample_idx].item()
                if err_val < diag_thresh:
                    continue
                meta_entry = meta_batch[sample_idx] if sample_idx < len(meta_batch) else {}
                diag_records.append({
                    "err": err_val,
                    "pred": float(pred_count[sample_idx].item()),
                    "gt": float(gt_count[sample_idx].item()),
                    "img_path": meta_entry.get("img_path") if isinstance(meta_entry, dict) else None,
                })

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

    if diag_enabled and diag_records:
        diag_records.sort(key=lambda x: x["err"], reverse=True)
        print("⚠️ Scene con errore elevato (|pred-gt| >= {:.1f}):".format(diag_thresh))
        for rec in diag_records[:diag_topk]:
            path = rec.get("img_path") or "n/a"
            print(
                "   • err={:.1f}, pred={:.1f}, gt={:.1f}, img={}".format(
                    rec["err"], rec["pred"], rec["gt"], path
                )
            )
        print("")

    return avg_loss, mae, rmse, total_pred, total_gt


# -----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 (P2R)")
    parser.add_argument("--config", default="config.yaml", help="Path al file di configurazione YAML")
    return parser.parse_args()


def main(config_path: str):
    cfg = load_config(config_path)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio Stage 2 (P2R Training) su {device}")
    print(f"ℹ️ Config in uso: {config_path}")

    # --- Config ---
    optim_cfg = cfg["OPTIM_P2R"]
    data_cfg = cfg["DATA"]
    p2r_loss_cfg = cfg["P2R_LOSS"]
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    early_stop_patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 0)))
    val_interval = max(1, int(optim_cfg.get("VAL_INTERVAL", 1)))
    extra_val_epochs = optim_cfg.get("EXTRA_VAL_EPOCHS", [])
    if not isinstance(extra_val_epochs, (list, tuple)):
        extra_val_epochs = [extra_val_epochs]
    extra_val_epochs = {
        int(ep)
        for ep in extra_val_epochs
        if isinstance(ep, (int, float)) and int(ep) > 0
    }

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

    collate_with_meta = partial(collate_fn, return_meta=True)

    hard_replay_cfg = cfg.get("P2R_HARD_REPLAY", {})
    scene_priority_cfg = cfg.get("P2R_SCENE_PRIORITIES", {}) or {}
    scene_priority_enabled = bool(scene_priority_cfg.get("ENABLE", False))
    auto_scene_boost = None
    scene_priority_rules = []
    if scene_priority_enabled:
        default_count_weight = float(scene_priority_cfg.get("DEFAULT_COUNT_WEIGHT", 0.0))
        default_count_thr = float(scene_priority_cfg.get("DEFAULT_COUNT_THRESHOLD", 0.0))
        target_list = scene_priority_cfg.get("TARGETS", []) or []
        for entry in target_list:
            match_key = str(entry.get("MATCH", "")).strip().lower()
            if not match_key:
                continue
            rule = {
                "match": match_key,
                "loss_boost": max(0.0, float(entry.get("LOSS_WEIGHT", 1.0)) - 1.0),
                "count_weight": float(entry.get("COUNT_WEIGHT", default_count_weight)),
                "count_threshold": float(entry.get("COUNT_THRESHOLD", default_count_thr)),
                "repeat_factor": max(1, int(entry.get("REPEAT_FACTOR", 1))),
            }
            scene_priority_rules.append(rule)
        if scene_priority_rules:
            print(f"ℹ️ Scene priority configurate: {len(scene_priority_rules)} target." )
        auto_scene_cfg = scene_priority_cfg.get("AUTO_COUNT_BOOST", {}) or {}
        if auto_scene_cfg.get("ENABLE", False):
            threshold = float(auto_scene_cfg.get("THRESHOLD", default_count_thr))
            count_weight = float(auto_scene_cfg.get("COUNT_WEIGHT", default_count_weight))
            margin = float(auto_scene_cfg.get("MARGIN", 0.0))
            loss_boost = max(0.0, float(auto_scene_cfg.get("LOSS_WEIGHT", 1.0)) - 1.0)
            if threshold > 0.0 and count_weight > 0.0:
                auto_scene_boost = {
                    "threshold": threshold,
                    "count_weight": count_weight,
                    "margin": margin,
                    "loss_boost": loss_boost,
                }
                boost_pct = loss_boost * 100.0
                print(
                    "ℹ️ Scene priority auto-densa: soglia={} | weight={:.2f} | margin={:.1f} | boost=+{:.1f}%".format(
                        threshold,
                        count_weight,
                        margin,
                        boost_pct,
                    )
                )

    train_dataset = train_ds
    if hard_replay_cfg.get("ENABLE", False):
        hard_names_cfg = hard_replay_cfg.get("IMAGES", []) or []
        if not isinstance(hard_names_cfg, (list, tuple)):
            hard_names_cfg = [hard_names_cfg]
        hard_name_set = {os.path.basename(str(name)).lower() for name in hard_names_cfg if str(name).strip()}
        matched_manual = set()
        if hard_name_set and hasattr(train_ds, "image_list"):
            hard_indices = []
            for idx, img_path in enumerate(train_ds.image_list):
                base_name = os.path.basename(img_path).lower()
                if base_name in hard_name_set:
                    matched_manual.add(base_name)
                    hard_indices.append(idx)
            if hard_indices:
                repeat_factor = max(1, int(hard_replay_cfg.get("REPEAT_FACTOR", 1)))
                repeated = [Subset(train_ds, hard_indices) for _ in range(repeat_factor)]
                train_dataset = ConcatDataset([train_dataset] + repeated)
                print(
                    "ℹ️ Hard replay manuale: {} immagini prioritarie replicate x{} (tot elementi {}).".format(
                        len(hard_indices), repeat_factor, len(train_dataset)
                    )
                )
            missing = sorted(hard_name_set - matched_manual)
            if missing:
                preview = ", ".join(missing[:5])
                print(
                    "⚠️ Hard replay: {} immagini non presenti nello split corrente (esempi: {}).".format(
                        len(missing), preview
                    )
                )
        elif hard_name_set:
            print("ℹ️ Hard replay configurato ma il dataset non espone 'image_list'.")

        auto_dense_cfg = hard_replay_cfg.get("AUTO_DENSE", {}) or {}
        if auto_dense_cfg.get("ENABLE", False) and hasattr(train_ds, "image_list"):
            dense_min_points = max(1, int(auto_dense_cfg.get("MIN_POINTS", 800)))
            dense_limit = auto_dense_cfg.get("MAX_SAMPLES")
            if dense_limit is not None:
                try:
                    dense_limit = max(1, int(dense_limit))
                except (TypeError, ValueError):
                    dense_limit = None
            dense_indices = _collect_dense_indices(train_ds, dense_min_points, limit=dense_limit)
            if dense_indices:
                dense_repeat = max(1, int(auto_dense_cfg.get("REPEAT_FACTOR", 1)))
                dense_sets = [Subset(train_ds, dense_indices) for _ in range(dense_repeat)]
                base_sets = list(train_dataset.datasets) if isinstance(train_dataset, ConcatDataset) else [train_dataset]
                train_dataset = ConcatDataset(base_sets + dense_sets)
                print(
                    "ℹ️ Hard replay auto-denso: {} immagini (>= {} pt) replicate x{} (tot elementi {}).".format(
                        len(dense_indices),
                        dense_min_points,
                        dense_repeat,
                        len(train_dataset),
                    )
                )
            else:
                print(
                    "ℹ️ Hard replay auto-denso: nessuna immagine supera la soglia di {} punti (limit={}).".format(
                        dense_min_points,
                        "n/d" if dense_limit is None else dense_limit,
                    )
                )

    if scene_priority_rules and hasattr(train_ds, "image_list"):
        priority_datasets = []
        cached_names = [os.path.basename(p).lower() for p in train_ds.image_list]
        for rule in scene_priority_rules:
            repeat_extra = max(0, rule["repeat_factor"] - 1)
            if repeat_extra <= 0:
                continue
            matched_idx = [idx for idx, name in enumerate(cached_names) if rule["match"] in name]
            if not matched_idx:
                continue
            priority_datasets.extend(Subset(train_ds, matched_idx) for _ in range(repeat_extra))
            print(
                "ℹ️ Scene priority replay: '{}' replicata x{} ({} elementi).".format(
                    rule["match"], repeat_extra, len(matched_idx)
                )
            )
        if priority_datasets:
            base_sets = list(train_dataset.datasets) if isinstance(train_dataset, ConcatDataset) else [train_dataset]
            train_dataset = ConcatDataset(base_sets + priority_datasets)

    dl_train = DataLoader(
        train_dataset, batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True, num_workers=optim_cfg["NUM_WORKERS"],
        drop_last=True, collate_fn=collate_with_meta, pin_memory=True
    )
    dl_val = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_with_meta, pin_memory=True
    )

    auto_recover_cfg = cfg.get("P2R_AUTO_RECOVER", {}) or {}
    auto_recover_enabled = bool(auto_recover_cfg.get("ENABLE", False))
    auto_recover_use_val_loader = bool(auto_recover_cfg.get("USE_VAL_LOADER", True))
    auto_recover_min_epoch = max(1, int(auto_recover_cfg.get("MIN_EPOCH", 50)))
    auto_recover_cooldown = max(0, int(auto_recover_cfg.get("COOLDOWN", 25)))
    auto_recover_floor_patience = max(0, int(auto_recover_cfg.get("FLOOR_PATIENCE", 48)))
    auto_recover_floor_margin = max(0.0, float(auto_recover_cfg.get("FLOOR_MARGIN", 0.08)))
    auto_recover_ratio_thr = max(0.0, float(auto_recover_cfg.get("RATIO_THR", 0.12)))
    auto_recover_ratio_patience = max(0, int(auto_recover_cfg.get("RATIO_PATIENCE", 96)))
    auto_recover_max_batches = auto_recover_cfg.get("MAX_BATCHES")
    if auto_recover_max_batches is not None:
        try:
            auto_recover_max_batches = max(1, int(auto_recover_max_batches))
        except (TypeError, ValueError):
            auto_recover_max_batches = None
    auto_recover_loader = dl_val if auto_recover_use_val_loader else dl_train

    curriculum_cfg = cfg.get("P2R_CURRICULUM", {}) or {}
    curriculum_enabled = bool(curriculum_cfg.get("ENABLE", False))
    curriculum_epochs = max(0, int(curriculum_cfg.get("EPOCHS", 0)))
    curriculum_min_points = max(1, int(curriculum_cfg.get("MIN_POINTS", 0)))
    curriculum_batch_scale = max(0.25, float(curriculum_cfg.get("BATCH_SIZE_SCALE", 1.0)))
    curriculum_limit = curriculum_cfg.get("MAX_SAMPLES")
    if curriculum_limit is not None:
        try:
            curriculum_limit = max(1, int(curriculum_limit))
        except (TypeError, ValueError):
            curriculum_limit = None
    curriculum_loader = None
    curriculum_indices = []
    if curriculum_enabled and curriculum_epochs > 0 and curriculum_min_points > 0:
        print(
            f"ℹ️ Curriculum denso attivo per le prime {curriculum_epochs} epoche"
            f" (min_points={curriculum_min_points}, batch_scale={curriculum_batch_scale:.2f})."
        )
        curriculum_indices = _collect_dense_indices(train_ds, curriculum_min_points, limit=curriculum_limit)
        if curriculum_indices:
            print(f"   ↳ Trovate {len(curriculum_indices)} immagini dense per il curriculum iniziale.")
            curriculum_subset = Subset(train_ds, curriculum_indices)
            curriculum_batch_size = max(1, int(round(optim_cfg["BATCH_SIZE"] * curriculum_batch_scale)))
            curriculum_loader = DataLoader(
                curriculum_subset,
                batch_size=curriculum_batch_size,
                shuffle=True,
                num_workers=optim_cfg["NUM_WORKERS"],
                drop_last=len(curriculum_subset) >= curriculum_batch_size,
                collate_fn=collate_with_meta,
                pin_memory=True,
            )
        else:
            print("⚠️ Curriculum richiesto ma nessuna immagine supera la soglia di punti: funzione disattivata.")
            curriculum_enabled = False
    else:
        curriculum_enabled = False

    # --- Modello ---
    dataset_name = cfg["DATASET"]
    model_cfg = cfg["MODEL"]
    bin_config = cfg["BINS_CONFIG"][dataset_name]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        backbone_name=model_cfg["BACKBONE"],
        pi_thresh=model_cfg.get("ZIP_PI_THRESH"),
        gate=model_cfg.get("GATE", "multiply"),
        upsample_to_input=False,
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs=zip_head_kwargs,
        soft_pi_gate=model_cfg.get("ZIP_PI_SOFT", False),
        pi_gate_power=model_cfg.get("ZIP_PI_SOFT_POWER", 1.0),
        pi_gate_min=model_cfg.get("ZIP_PI_SOFT_MIN", 0.0),
        apply_gate_to_output=model_cfg.get("ZIP_PI_APPLY_TO_P2R", False),
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
        print("🔓 Fine-tuning del backbone ATTIVO...")
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        print("🧊 Backbone congelato.")
        for p in model.backbone.parameters():
            p.requires_grad = False

    for p in model.p2r_head.parameters():
        p.requires_grad = True  

    log_scale_init = p2r_loss_cfg.get("LOG_SCALE_INIT")
    if log_scale_init is not None and hasattr(model.p2r_head, "log_scale"):
        model.p2r_head.log_scale.data.fill_(float(log_scale_init))
        print(f"ℹ️ log_scale inizializzato a {float(log_scale_init):.3f}")

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
    base_lr = float(optim_cfg.get('LR', 1e-4))
    if head_params:
        params_to_train.append({'params': head_params, 'lr': base_lr})
    if log_scale_params:
        lr_factor = float(p2r_loss_cfg.get("LOG_SCALE_LR_FACTOR", 0.1))
        params_to_train.append({'params': log_scale_params, 'lr': base_lr * lr_factor})
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
    if "SCALE_PENALTY_HUBER_DELTA" in p2r_loss_cfg:
        loss_kwargs["scale_huber_delta"] = float(p2r_loss_cfg["SCALE_PENALTY_HUBER_DELTA"])
    if "SCALE_PENALTY_CAP" in p2r_loss_cfg:
        loss_kwargs["scale_penalty_cap"] = float(p2r_loss_cfg["SCALE_PENALTY_CAP"])
    if "POS_WEIGHT" in p2r_loss_cfg:
        loss_kwargs["pos_weight"] = float(p2r_loss_cfg["POS_WEIGHT"])
    if "CHUNK_SIZE" in p2r_loss_cfg:
        loss_kwargs["chunk_size"] = int(p2r_loss_cfg["CHUNK_SIZE"])
    if "MIN_RADIUS" in p2r_loss_cfg:
        loss_kwargs["min_radius"] = float(p2r_loss_cfg["MIN_RADIUS"])
    if "MAX_RADIUS" in p2r_loss_cfg:
        loss_kwargs["max_radius"] = float(p2r_loss_cfg["MAX_RADIUS"])
    p2r_loss_fn = P2RLoss(**loss_kwargs).to(device)

    count_l1_weight = float(p2r_loss_cfg.get("COUNT_L1_WEIGHT", 0.0))
    density_l1_weight = float(p2r_loss_cfg.get("DENSITY_L1_WEIGHT", 0.0))

    log_scale_reg_weight = float(p2r_loss_cfg.get("LOG_SCALE_REG_WEIGHT", 0.0))
    log_scale_reg_target = float(p2r_loss_cfg.get("LOG_SCALE_REG_TARGET", 0.0))
    log_scale_recalibrate_thr = float(p2r_loss_cfg.get("LOG_SCALE_RECALIBRATE_THR", 0.0))
    count_drift_cfg = p2r_loss_cfg.get("COUNT_DRIFT_PENALTY", {}) or {}
    count_drift_weight = 0.0
    count_drift_threshold = 0.0
    if bool(count_drift_cfg.get("ENABLE", False)):
        count_drift_weight = float(count_drift_cfg.get("WEIGHT", 0.0))
        count_drift_threshold = float(count_drift_cfg.get("THRESHOLD", 0.0))

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
    calibrate_trim = float(p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_TRIM", 0.0))
    calibrate_stat = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_STAT", "median")
    calibrate_min_samples = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_SAMPLES")
    calibrate_min_bias = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_BIAS")
    calibrate_max_bias = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_BIAS")
    calibrate_dynamic_floor = p2r_loss_cfg.get("LOG_SCALE_DYNAMIC_FLOOR")
    calibrate_damping = p2r_loss_cfg.get("LOG_SCALE_CALIBRATION_DAMPING", 1.0)
    calibrate_density_scale(
        model,
        dl_val,
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
        adjust_damping=calibrate_damping,
    )

    floor_candidates = []
    clamp_min_val = None
    if clamp_cfg and len(clamp_cfg) >= 1:
        try:
            clamp_min_val = float(clamp_cfg[0])
            floor_candidates.append(clamp_min_val)
        except (TypeError, ValueError):
            clamp_min_val = None
    if calibrate_dynamic_floor is not None:
        try:
            floor_candidates.append(float(calibrate_dynamic_floor))
        except (TypeError, ValueError):
            pass
    auto_floor_scale = None
    if floor_candidates:
        floor_log = min(floor_candidates)
        auto_floor_scale = float(np.exp(floor_log))
    floor_scale_threshold = None
    if auto_floor_scale is not None:
        floor_scale_threshold = auto_floor_scale * (1.0 + auto_recover_floor_margin)
    if auto_recover_enabled and auto_recover_loader is None:
        print("⚠️ Auto recovery richiesto ma nessun loader valido disponibile: disabilitato.")
        auto_recover_enabled = False

    # --- Training loop ---
    if count_l1_weight > 0.0:
        print(f"ℹ️ Stage 2: penalità L1 sui conteggi attiva (peso={count_l1_weight:.3f}).")
    if density_l1_weight > 0.0:
        print(f"ℹ️ Stage 2: penalità L1 sulla densità attiva (peso={density_l1_weight:.3e}).")
    if count_drift_weight > 0.0:
        print(
            f"ℹ️ Stage 2: penalità dinamica sui conteggi attiva (peso={count_drift_weight:.3f}, "
            f"thr={count_drift_threshold:.1f})."
        )

    no_improve_rounds = 0
    stop_training = False
    ratio_drift_streak = 0
    floor_hit_streak = 0
    last_auto_recover_epoch = -10 ** 6
    recent_batch_ratio = 1.0
    recent_dens_scale = None

    def trigger_auto_recover(reason: str, current_epoch: int):
        nonlocal ratio_drift_streak, floor_hit_streak, last_auto_recover_epoch
        nonlocal recent_batch_ratio, recent_dens_scale
        if not auto_recover_enabled:
            return False
        if current_epoch < auto_recover_min_epoch:
            return False
        if auto_recover_cooldown > 0 and (current_epoch - last_auto_recover_epoch) < auto_recover_cooldown:
            return False
        if auto_recover_loader is None:
            return False

        print(f"⚠️ Stage 2: auto-recovery log_scale attivato ({reason}).")
        max_batches_recover = auto_recover_max_batches
        if max_batches_recover is None:
            max_batches_recover = calibrate_max_batches if calibrate_max_batches is not None else 6
        bias = calibrate_density_scale(
            model,
            auto_recover_loader,
            device,
            default_down,
            max_batches=max_batches_recover,
            clamp_range=clamp_cfg,
            max_adjust=max_adjust,
            min_samples=calibrate_min_samples,
            min_bias=calibrate_min_bias,
            max_bias=calibrate_max_bias,
            trim_ratio=calibrate_trim,
            stat=calibrate_stat,
            dynamic_floor=calibrate_dynamic_floor,
            adjust_damping=calibrate_damping,
        )
        if bias is not None:
            print(f"   ↳ Bias dopo auto-recovery: {bias:.3f}")
        last_auto_recover_epoch = current_epoch
        ratio_drift_streak = 0
        floor_hit_streak = 0
        recent_batch_ratio = 1.0
        recent_dens_scale = None
        return bias is not None

    print(f"🚀 Inizio training Stage 2 per {optim_cfg['EPOCHS']} epoche...")
    global_step_counter = 0
    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        using_curriculum = curriculum_enabled and curriculum_loader is not None and ep <= curriculum_epochs
        active_loader = curriculum_loader if using_curriculum else dl_train
        loader_label = "Curriculum" if using_curriculum else "P2R Train"
        pbar = tqdm(
            enumerate(active_loader, start=1),
            total=len(active_loader),
            desc=f"[{loader_label}] Epoch {ep}/{optim_cfg['EPOCHS']}"
        )

        for batch_idx, batch in pbar:
            images, gt_density, points, meta = _split_loader_batch(batch)
            images = images.to(device)
            gt_density = gt_density.to(device)
            points_list = [p.to(device) for p in points]
            global_step_counter += 1
            current_global_step = global_step_counter

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
            down_tuple = (down_h, down_w)

            # ✅ ORA loss coerente con ReLU/(upscale²) — niente più logit
            loss = p2r_loss_fn(pred_density, points_list, down=down_tuple)

            down_h, down_w = down_tuple
            cell_area = down_h * down_w
            cell_area_tensor = pred_density.new_tensor(cell_area)

            pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area_tensor
            gt_counts = []
            for p in points_list:
                if p is None:
                    gt_counts.append(0.0)
                else:
                    gt_counts.append(float(p.shape[0]))
            gt_count = pred_density.new_tensor(gt_counts)

            count_l1 = None
            if count_l1_weight > 0.0:
                count_l1 = torch.abs(pred_count - gt_count).mean()
                loss = loss + count_l1_weight * count_l1

            density_l1 = None
            if density_l1_weight > 0.0:
                gt_density = gt_density.to(dtype=pred_density.dtype)
                if gt_density.shape[-2:] != pred_density.shape[-2:]:
                    original_h, original_w = gt_density.shape[-2], gt_density.shape[-1]
                    gt_resized = F.interpolate(
                        gt_density,
                        size=pred_density.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    original_area = float(original_h * original_w)
                    target_area = float(pred_density.shape[-2] * pred_density.shape[-1])
                    gt_resized = gt_resized * (original_area / max(target_area, 1.0))
                else:
                    gt_resized = gt_density
                density_l1 = F.l1_loss(pred_density, gt_resized)
                loss = loss + density_l1_weight * density_l1

            reg_penalty_weighted = None
            if log_scale_reg_weight > 0.0 and hasattr(model.p2r_head, "log_scale"):
                target_val = float(log_scale_reg_target)
                log_scale_param = model.p2r_head.log_scale
                reg_penalty = F.smooth_l1_loss(
                    log_scale_param,
                    log_scale_param.new_tensor(target_val),
                    reduction="mean",
                )
                reg_penalty_weighted = log_scale_reg_weight * reg_penalty
                loss = loss + reg_penalty_weighted

            count_drift_loss = None
            if count_drift_weight > 0.0:
                drift_err = torch.abs(pred_count - gt_count)
                if count_drift_threshold > 0.0:
                    drift_err = torch.relu(drift_err - count_drift_threshold)
                count_drift_loss = drift_err.mean()
                loss = loss + count_drift_weight * count_drift_loss

            auto_scene_loss = None
            scene_priority_loss = None
            scene_matches = 0
            if scene_priority_rules and meta:
                penalties = []
                total_boost = 0.0
                meta_list = meta if isinstance(meta, list) else []
                for sample_idx, meta_entry in enumerate(meta_list):
                    img_path = str(meta_entry.get("img_path") or "")
                    img_name = os.path.basename(img_path).lower()
                    matched_rules = [rule for rule in scene_priority_rules if rule["match"] in img_name]
                    if not matched_rules:
                        continue
                    scene_matches += 1
                    for rule in matched_rules:
                        if rule["count_weight"] > 0.0:
                            err = torch.abs(pred_count[sample_idx] - gt_count[sample_idx])
                            thr = rule["count_threshold"]
                            if thr > 0.0:
                                err = torch.relu(err - thr)
                            penalties.append(rule["count_weight"] * err)
                        if rule["loss_boost"] > 0.0:
                            total_boost += rule["loss_boost"]
                if penalties:
                    scene_priority_loss = torch.stack(penalties).mean()
                    loss = loss + scene_priority_loss
                if scene_matches > 0 and total_boost > 0.0:
                    boost_factor = 1.0 + (total_boost / float(scene_matches))
                    loss = loss * boost_factor

            if auto_scene_boost is not None:
                threshold = auto_scene_boost["threshold"]
                mask = gt_count >= threshold
                if torch.count_nonzero(mask) > 0:
                    margin = auto_scene_boost.get("margin", 0.0)
                    err = torch.abs(pred_count[mask] - gt_count[mask])
                    if margin > 0.0:
                        err = torch.relu(err - margin)
                    weight = auto_scene_boost.get("count_weight", 0.0)
                    if weight > 0.0 and err.numel() > 0:
                        auto_scene_loss = err.mean() * weight
                        loss = loss + auto_scene_loss
                    boost = auto_scene_boost.get("loss_boost", 0.0)
                    if boost > 0.0:
                        loss = loss * (1.0 + boost)

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
                    dyn_clamp = getattr(model.p2r_head, "_dynamic_clamp", None)
                    if dyn_clamp is not None and len(dyn_clamp) == 2:
                        min_val = min(min_val, float(dyn_clamp[0]))
                        max_val = max(max_val, float(dyn_clamp[1]))
                    model.p2r_head.log_scale.data.clamp_(min_val, max_val)
                    model.p2r_head._dynamic_clamp = (float(min_val), float(max_val))
                global_step = None
                if hasattr(model.p2r_head, "log_scale"):
                    ds_val = float(torch.exp(model.p2r_head.log_scale.detach()).item())
                    if writer:
                        global_step = current_global_step
                        writer.add_scalar("p2r_head/density_scale", ds_val, global_step)
                        writer.add_scalar("p2r_head/log_scale", float(model.p2r_head.log_scale.detach().item()), global_step)
                        if reg_penalty_weighted is not None:
                            writer.add_scalar(
                                "loss/log_scale_reg",
                                float(reg_penalty_weighted.detach().item()),
                                global_step,
                            )
            batch_pred_sum = float(pred_count.detach().sum().item())
            batch_gt_sum = float(gt_count.detach().sum().item())
            batch_ratio = batch_pred_sum / batch_gt_sum if batch_gt_sum > 1e-6 else 0.0
            recent_batch_ratio = batch_ratio

            if writer:
                if global_step is None:
                    global_step = current_global_step
                dens_mean = float(pred_density.detach().mean().item())
                writer.add_scalar("train/batch_pred_gt_ratio", batch_ratio, global_step)
                writer.add_scalar("train/density_mean", dens_mean, global_step)
                if count_l1 is not None:
                    writer.add_scalar("loss/count_l1", float(count_l1.detach().item()), global_step)
                if density_l1 is not None:
                    writer.add_scalar("loss/density_l1", float(density_l1.detach().item()), global_step)
                if count_drift_loss is not None:
                    writer.add_scalar("loss/count_drift", float(count_drift_loss.detach().item()), global_step)
                if scene_priority_loss is not None:
                    writer.add_scalar("loss/scene_priority", float(scene_priority_loss.detach().item()), global_step)
                if auto_scene_loss is not None:
                    writer.add_scalar("loss/scene_priority_auto", float(auto_scene_loss.detach().item()), global_step)

            if auto_recover_enabled and auto_recover_ratio_thr > 0.0 and batch_gt_sum > 1e-6:
                if abs(batch_ratio - 1.0) > auto_recover_ratio_thr:
                    ratio_drift_streak += 1
                else:
                    ratio_drift_streak = max(ratio_drift_streak - 1, 0)

            if (
                auto_recover_enabled
                and auto_recover_floor_patience > 0
                and floor_scale_threshold is not None
                and ds_val is not None
            ):
                recent_dens_scale = ds_val
                if ds_val <= floor_scale_threshold:
                    floor_hit_streak += 1
                else:
                    floor_hit_streak = max(floor_hit_streak - 1, 0)

            postfix = {
                "loss": f"{loss.item():.4f}",
                "lr": f"{opt.param_groups[0]['lr']:.6f}",
            }
            if ds_val is not None:
                postfix["dens_scale"] = f"{ds_val:.6f}"
            if count_l1 is not None:
                postfix["cnt"] = f"{count_l1.item():.2f}"
            if density_l1 is not None:
                postfix["dens"] = f"{density_l1.item():.4f}"
            if count_drift_loss is not None:
                postfix["drift"] = f"{count_drift_loss.item():.2f}"
            if scene_matches:
                postfix["scene"] = str(scene_matches)
            pbar.set_postfix(postfix)

        auto_recover_triggered = False
        if auto_recover_enabled:
            recover_reason = None
            if auto_recover_floor_patience > 0 and floor_hit_streak >= auto_recover_floor_patience:
                approx_scale = recent_dens_scale if recent_dens_scale is not None else 0.0
                recover_reason = f"dens_scale al limite (~{approx_scale:.2e})"
            elif (
                auto_recover_ratio_thr > 0.0
                and auto_recover_ratio_patience > 0
                and ratio_drift_streak >= auto_recover_ratio_patience
            ):
                recover_reason = f"drift Pred/GT nel train (~{recent_batch_ratio:.3f})"

            if recover_reason:
                auto_recover_triggered = trigger_auto_recover(recover_reason, ep)

        # --- Scheduler step ---
        if scheduler:
            scheduler.step()

        avg_train = total_loss / len(active_loader)
        if writer:
            writer.add_scalar("train/loss_p2r", avg_train, ep)
            writer.add_scalar("lr/p2r_head", opt.param_groups[0]["lr"], ep)
        if writer and backbone_group_idx is not None:
            writer.add_scalar("lr/backbone", opt.param_groups[backbone_group_idx]["lr"], ep)

        # --- Validazione periodica ---
        forced_extra_val = ep in extra_val_epochs
        should_validate = forced_extra_val or (ep % val_interval == 0) or (ep == optim_cfg["EPOCHS"])

        if should_validate:
            if forced_extra_val:
                print(f"ℹ️ Validazione extra Stage 2 forzata all'epoca {ep} (EXTRA_VAL_EPOCHS).")
            val_loss, mae, mse, tot_pred, tot_gt = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
            orig_bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")

            recalibrated = False
            if (
                log_scale_recalibrate_thr > 0.0
                and tot_gt > 0
                and np.isfinite(orig_bias)
                and abs(orig_bias - 1.0) > log_scale_recalibrate_thr
                and hasattr(model.p2r_head, "log_scale")
            ):
                print(
                    "🔁 Stage 2: Pred/GT {:.3f} fuori soglia ±{:.3f}. Ricalibro log_scale prima di proseguire.".format(
                        orig_bias,
                        log_scale_recalibrate_thr,
                    )
                )
                recalib_bias = calibrate_density_scale(
                    model,
                    dl_val,
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
                    adjust_damping=calibrate_damping,
                )
                if recalib_bias is not None:
                    print(f"   ↳ Bias stimato dopo ricalibrazione: {recalib_bias:.3f}")
                val_loss, mae, mse, tot_pred, tot_gt = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
                orig_bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")
                recalibrated = True

            if writer:
                writer.add_scalar("val/loss_p2r", val_loss, ep)
                writer.add_scalar("val/MAE", mae, ep)
                writer.add_scalar("val/MSE", mse, ep)
                if tot_gt > 0:
                    writer.add_scalar("val/pred_gt_ratio", orig_bias, ep)
                writer.add_scalar("val/recalibration_performed", 1.0 if recalibrated else 0.0, ep)

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
                            min_samples=calibrate_min_samples,
                            min_bias=calibrate_min_bias,
                            max_bias=calibrate_max_bias,
                            trim_ratio=calibrate_trim,
                            stat=calibrate_stat,
                            dynamic_floor=calibrate_dynamic_floor,
                            adjust_damping=calibrate_damping,
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
                no_improve_rounds = 0
            else:
                if early_stop_patience > 0:
                    no_improve_rounds += 1
                    if no_improve_rounds >= early_stop_patience:
                        print(
                            f"⛔ Early stopping: nessun miglioramento MAE per "
                            f"{no_improve_rounds} validazioni consecutive."
                        )
                        stop_training = True

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

            if stop_training:
                break

            # --- Debug ogni 50 epoche ---
            if ep % 50 == 0:
                with torch.no_grad():
                    sample_out = model(images[:1])
                    dens = sample_out["p2r_density"]
                    print(f"[DEBUG Epoch {ep}] Mean dens: {dens.mean().item():.4f}, Max: {dens.max().item():.4f}")

    if writer:
        writer.close()
    print("✅ Stage 2 completato.")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
