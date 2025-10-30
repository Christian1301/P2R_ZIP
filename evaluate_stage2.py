# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Evaluation Stage 2 — P2R Loss + Head (ReLU/upscale²)
# ============================================================

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from losses.p2r_region_loss import P2RLoss
from train_utils import collate_fn, init_seeds


@torch.no_grad()
def evaluate_p2r(model, loader, loss_fn, device, cfg):
    """
    Valutazione Stage 2:
    - Supporta P2RHead con ReLU/(upscale²)
    - Calcola conteggi corretti anche se la densità è upsamplata all'input
    - Mostra statistiche diagnostiche dettagliate
    """
    model.eval()
    total_loss, mae_errors, mse_errors = 0.0, 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    # --- Recupera downsample del training (default 8) ---
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

        # --- Determina il fattore di scala effettivo ---
        if H_out == H_in and W_out == W_in:
            # Il modello ha upsamplato fino all'input → usa il downsampling del training
            down_h = down_w = float(default_down)
        else:
            down_h = H_in / max(H_out, 1)
            down_w = W_in / max(W_out, 1)

        if (abs(H_in - down_h * H_out) > 1.0 or abs(W_in - down_w * W_out) > 1.0) and not hasattr(evaluate_p2r, "_shape_warned"):
            print(f"⚠️ Downsample non intero rilevato: input {H_in}x{W_in}, output {H_out}x{W_out}, "
                  f"down_h={down_h:.4f}, down_w={down_w:.4f}")
            evaluate_p2r._shape_warned = True

        down_tuple = (down_h, down_w)

        # --- Calcolo loss coerente col training ---
        loss = loss_fn(pred_density, points_list, down=down_tuple)
        total_loss += loss.item()

        if H_out == H_in and W_out == W_in:
            # Output upsamplato fino all'input → densità già normalizzata in P2RHead
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        else:
            # Output ridotto → ogni cella rappresenta (down×down) pixel
            cell_area = down_h * down_w
            pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area

        gt_count = torch.tensor([len(p) for p in points_list], dtype=torch.float32, device=device)

        # --- Debug (solo primo batch) ---
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

    # --- Metriche finali ---
    n = len(loader.dataset)
    avg_loss = total_loss / len(loader)
    mae = mae_errors / n
    rmse = np.sqrt(mse_errors / n)

    print("\n===== RISULTATI FINALI STAGE 2 =====")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    if total_gt > 0:
        bias = total_pred / total_gt
        print(f"Pred / GT ratio: {bias:.3f} (tot_pred={total_pred:.1f}, tot_gt={total_gt:.1f})")
    print("=====================================\n")

    return avg_loss, mae, rmse, total_pred, total_gt


def main():
    # --- Caricamento config ---
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio valutazione Stage 2 su {device}")

    # --- Dataset ---
    DatasetClass = get_dataset(cfg["DATASET"])
    data_cfg = cfg["DATA"]
    transforms = build_transforms(data_cfg, is_train=False)
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=transforms
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["OPTIM_P2R"]["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True
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
        upsample_to_input=False,  # <<< come nel training!
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # --- Carica checkpoint Stage 2 ---
    ckpt_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    ckpt_path = os.path.join(ckpt_dir, "stage2_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("❌ Nessun checkpoint Stage 2 trovato.")
        return

    print(f"✅ Checkpoint caricato correttamente da: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    # --- Loss ---
    loss_cfg = cfg.get("P2R_LOSS", {})
    loss_kwargs = {}
    if "SCALE_WEIGHT" in loss_cfg:
        loss_kwargs["scale_weight"] = float(loss_cfg["SCALE_WEIGHT"])
    if "CHUNK_SIZE" in loss_cfg:
        loss_kwargs["chunk_size"] = int(loss_cfg["CHUNK_SIZE"])
    if "MIN_RADIUS" in loss_cfg:
        loss_kwargs["min_radius"] = float(loss_cfg["MIN_RADIUS"])
    if "MAX_RADIUS" in loss_cfg:
        loss_kwargs["max_radius"] = float(loss_cfg["MAX_RADIUS"])
    loss_fn = P2RLoss(**loss_kwargs).to(device)

    # --- Valutazione ---
    evaluate_p2r(model, val_loader, loss_fn, device, cfg)


if __name__ == "__main__":
    main()
