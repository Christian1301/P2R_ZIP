# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 3 - Joint Fine-Tuning
# Combina i checkpoint di ZIP (stage1) e P2R (stage2)
# ============================================================

import os
import yaml
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from losses.zip_nll import zip_nll
from losses.p2r_region_loss import P2RLoss
from train_utils import resume_if_exists, save_checkpoint, setup_experiment, collate_crowd


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------
# Validazione congiunta
# ------------------------------------------------------------
def evaluate_joint(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    mae_errors, mse_errors = [], []
    p2r_loss_fn = P2RLoss().to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validating]"):
            img = batch["image"].to(device)
            blocks = batch["zip_blocks"].to(device)
            points = [p.to(device) for p in batch["points"]]

            out = model(img)
            l_zip = zip_nll(out["pi"], out["lam"], blocks)
            l_p2r = p2r_loss_fn(out["density"], points, down=16)
            loss = l_zip + cfg["LOSS"]["JOINT_ALPHA"] * l_p2r
            total_loss += loss.item()

            pred_count = torch.sum(out["density"], dim=(1, 2, 3))
            gt_count = torch.tensor([len(p) for p in batch["points"]],
                                    dtype=torch.float32, device=device)
            mae_errors.extend(torch.abs(pred_count - gt_count).cpu().numpy())
            mse_errors.extend(((pred_count - gt_count) ** 2).cpu().numpy())

    model.train()
    avg_loss = total_loss / len(loader)
    mae = np.mean(mae_errors)
    rmse = np.sqrt(np.mean(mse_errors))
    return avg_loss, mae, rmse


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    cfg = yaml.safe_load(open("config.yaml"))
    set_seed(cfg["SEED"])
    device = torch.device(cfg["DEVICE"] if torch.cuda.is_available() else "cpu")

    # --- Datasets ---
    Dataset = get_dataset(cfg["DATASET"])
    train_ds = Dataset(cfg["DATA"]["ROOT"], split=cfg["DATA"]["TRAIN_SPLIT"],
                       block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"])
    val_ds = Dataset(cfg["DATA"]["ROOT"], split=cfg["DATA"]["VAL_SPLIT"],
                     block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"])

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg["OPTIM"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["OPTIM"]["NUM_WORKERS"],
        drop_last=True,
        collate_fn=lambda x: collate_crowd(x, cfg["DATA"]["ZIP_BLOCK_SIZE"])
    )

    dl_val = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: collate_crowd(x, cfg["DATA"]["ZIP_BLOCK_SIZE"])
    )

    # --- Modello ---
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"]
    ).to(device)

    # ============================================================
    # 🔹 Caricamento combinato ZIP (stage1) + P2R (stage2)
    # ============================================================
    base_name = cfg["RUN_NAME"]
    exp_dir = cfg["EXP"]["OUT_DIR"]

    ckpt_zip = os.path.join(exp_dir, f"{base_name}_zip", "best_model.pth")
    ckpt_p2r = os.path.join(exp_dir, f"{base_name}_p2r", "best_model.pth")

    if not os.path.isfile(ckpt_zip) or not os.path.isfile(ckpt_p2r):
        raise FileNotFoundError(
            f"❌ Mancano i checkpoint ZIP e/o P2R richiesti.\n"
            f"ZIP: {ckpt_zip}\nP2R: {ckpt_p2r}"
        )

    # 1️⃣ Carica backbone + ZIP Head
    zip_state = torch.load(ckpt_zip, map_location="cpu")
    model.load_state_dict(zip_state, strict=False)
    print(f"✅ Caricato backbone + ZIP Head da: {ckpt_zip}")

    # 2️⃣ Carica solo la P2R Head
    p2r_ckpt = torch.load(ckpt_p2r, map_location="cpu")

    # scegli la chiave giusta
    if "student" in p2r_ckpt:
        p2r_state = p2r_ckpt["student"]
    elif "model" in p2r_ckpt:
        p2r_state = p2r_ckpt["model"]
    else:
        raise KeyError("❌ Nessun campo 'student' o 'model' trovato nel checkpoint P2R.")

    model_state = model.state_dict()
    loaded_p2r = 0
    for name, param in p2r_state.items():
        if name in model_state and "zip_head" not in name and "backbone" not in name:
            model_state[name].copy_(param)
            loaded_p2r += 1

    print(f"✅ Caricata P2R Head da: {ckpt_p2r} ({loaded_p2r} parametri trovati)")

    # ============================================================
    # Ottimizzatore + Scheduler
    # ============================================================
    params = [
        {"params": model.backbone.parameters(), "lr": cfg["OPTIM"].get("LR_BACKBONE", cfg["OPTIM"]["LR"])},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": cfg["OPTIM"]["LR"]}
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg["OPTIM"]["WEIGHT_DECAY"])

    def warmup_lambda(epoch):
        warmup_epochs = cfg["OPTIM"].get("WARMUP_EPOCHS", 0)
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0

    warmup = LambdaLR(opt, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

    # Setup esperimento
    joint_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_joint")
    writer = setup_experiment(joint_dir)
    start_ep, best_val = (1, float("inf"))
    if cfg["OPTIM"]["RESUME_LAST"]:
        start_ep, best_val = resume_if_exists(model, opt, joint_dir, device)

    epochs = cfg["OPTIM"]["EPOCHS"]
    alpha = float(cfg["LOSS"]["JOINT_ALPHA"])
    p2r_loss_fn = P2RLoss().to(device)

    # ============================================================
    # 🔸 TRAINING LOOP
    # ============================================================
    for ep in range(start_ep, epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(dl_train, desc=f"[JOINT] Epoch {ep}/{epochs}")

        # (Facoltativo) warm-up ZIP Head: congela le prime epoche
        freeze_zip = ep <= 10
        for p in model.zip_head.parameters():
            p.requires_grad = not freeze_zip

        for batch in pbar:
            img = batch["image"].to(device)
            blocks = batch["zip_blocks"].to(device)
            points = [p.to(device) for p in batch["points"]]

            out = model(img)
            l_zip = zip_nll(out["pi"], out["lam"], blocks)
            l_p2r = p2r_loss_fn(out["density"], points, down=16)
            loss = l_zip + alpha * l_p2r

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total / len(dl_train)
        writer.add_scalar("train/loss", avg_train, ep)
        writer.add_scalar("lr/backbone", opt.param_groups[0]["lr"], ep)
        writer.add_scalar("lr/heads", opt.param_groups[1]["lr"], ep)

        # --- Validazione ---
        if ep % cfg["OPTIM"]["VAL_INTERVAL"] == 0 or ep == epochs:
            val_loss, mae, rmse = evaluate_joint(model, dl_val, device, cfg)
            writer.add_scalar("val/loss", val_loss, ep)
            writer.add_scalar("val/MAE", mae, ep)
            writer.add_scalar("val/RMSE", rmse, ep)

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            save_checkpoint(model, opt, ep, val_loss, best_val, joint_dir, is_best=is_best)

            print(f"Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | Best: {best_val:.4f}")

        # Scheduler
        if ep <= cfg["OPTIM"].get("WARMUP_EPOCHS", 0):
            warmup.step()
        else:
            cosine.step()

    writer.close()


# ------------------------------------------------------------
if __name__ == "__main__":
    main()