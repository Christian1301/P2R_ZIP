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
    resume_if_exists, save_checkpoint, setup_experiment, collate_fn
)

# -----------------------------------------------------------
@torch.no_grad()
def evaluate_p2r(model, loader, loss_fn, device, cfg):
    """
    Valutazione P2R coerente con la P2RHead (densità ReLU/(upscale²)):
    - Applica correzione * (down^2)
    - Calcola MAE, RMSE e loss media
    - Mostra range e media delle mappe
    """
    model.eval()
    total_loss, mae_errors, mse_errors = 0.0, 0.0, 0.0

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
        down = int(round(H_in / H_out))

        # --- Loss (uguale al training) ---
        loss = loss_fn(pred_density, points_list, down=down)
        total_loss += loss.item()

        if H_out == H_in and W_out == W_in:
            # Output upsamplato fino all'input → densità già normalizzata in P2RHead
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
        else:
            # Output ridotto → ogni cella rappresenta (down×down) pixel
            pred_count = torch.sum(pred_density, dim=(1, 2, 3)) * (down ** 2)
        gt_count = torch.tensor([len(p) for p in points_list], dtype=torch.float32, device=device)

        # --- Debug solo la prima volta ---
        if not hasattr(evaluate_p2r, "_debug_done"):
            print("===== DEBUG STAGE 2 =====")
            print(f"Input size: {H_in}x{W_in}")
            print(f"Output size: {H_out}x{W_out}")
            print(f"Downsampling factor: {down}x")
            print(f"Density map range: [{pred_density.min().item():.4f}, {pred_density.max().item():.4f}]")
            print(f"Mean density: {pred_density.mean().item():.6f}")
            print(f"[DEBUG] Pred count (scaled): {pred_count[0].item():.2f}, GT count: {gt_count[0].item():.2f}")
            print("=========================")
            evaluate_p2r._debug_done = True

        mae_errors += torch.abs(pred_count - gt_count).sum().item()
        mse_errors += ((pred_count - gt_count) ** 2).sum().item()

    # --- Medie ---
    n = len(loader.dataset)
    avg_loss = total_loss / len(loader)
    mae = mae_errors / n
    rmse = np.sqrt(mse_errors / n)

    print("\n===== RISULTATI FINALI STAGE 2 =====")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("=====================================\n")

    return avg_loss, mae, rmse


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
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=False,
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"]
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

    # --- Congela Backbone + ZIPHead ---
    print("🧊 Congelamento pesi di Backbone e ZIPHead...")
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.zip_head.parameters(): p.requires_grad = False
    for p in model.p2r_head.parameters(): p.requires_grad = True  # Solo P2RHead addestrabile

    # --- Ottimizzatore + Scheduler ---
    params_to_train = [{'params': model.p2r_head.parameters(), 'lr': optim_cfg['LR']}]
    opt = get_optimizer(params_to_train, optim_cfg)
    scheduler = get_scheduler(opt, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # --- Loss ---
    p2r_loss_fn = P2RLoss().to(device)

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

    # --- Training loop ---
    print(f"🚀 Inizio training Stage 2 per {optim_cfg['EPOCHS']} epoche...")
    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"[P2R Train] Epoch {ep}/{optim_cfg['EPOCHS']}")

        for images, _, points in pbar:
            images = images.to(device)
            points_list = [p.to(device) for p in points]

            out = model(images)
            pred_density = out.get("p2r_density", out.get("density"))
            if pred_density is None:
                raise KeyError("Output 'p2r_density' o 'density' non trovato nel modello.")

            B, _, H_out, W_out = pred_density.shape
            _, _, H_in, W_in = images.shape
            down = int(round(H_in / H_out))

            # ✅ ORA loss coerente con ReLU/(upscale²) — niente più logit
            loss = 2.0 * p2r_loss_fn(pred_density, points_list, down=down)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.p2r_head.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}")
            # === Monitoraggio density_scale ===
            with torch.no_grad():
                if hasattr(model.p2r_head, "density_scale"):
                    ds_val = model.p2r_head.density_scale.item()
                    # Se vuoi includerlo nella barra di progresso
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                    lr=f"{opt.param_groups[0]['lr']:.6f}",
                                    dens_scale=f"{ds_val:.6f}")
                    # E registra anche nel writer per TensorBoard, se presente
                    if writer:
                        writer.add_scalar("p2r_head/density_scale", ds_val, ep)

        # --- Scheduler step ---
        if scheduler:
            scheduler.step()

        avg_train = total_loss / len(dl_train)
        if writer:
            writer.add_scalar("train/loss_p2r", avg_train, ep)
            writer.add_scalar("lr/p2r_head", opt.param_groups[0]["lr"], ep)

        # --- Validazione periodica ---
        if ep % optim_cfg["VAL_INTERVAL"] == 0 or ep == optim_cfg["EPOCHS"]:
            val_loss, mae, mse = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
            if writer:
                writer.add_scalar("val/loss_p2r", val_loss, ep)
                writer.add_scalar("val/MAE", mae, ep)
                writer.add_scalar("val/MSE", mse, ep)

            # --- Best model ---
            is_best = mae < best_mae
            if is_best:
                best_mae = mae

            save_checkpoint(model, opt, ep, mae, best_mae, exp_dir, is_best=False)

            if is_best and cfg["EXP"]["SAVE_BEST"]:
                best_path = os.path.join(stage1_dir, "stage2_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"💾 Nuovo best Stage 2 salvato in {best_path} (MAE={mae:.2f})")

            print(f"Epoch {ep}: Train Loss {avg_train:.4f} | Val Loss {val_loss:.4f} | MAE {mae:.2f} | MSE {mse:.2f} | Best MAE {best_mae:.2f}")

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
