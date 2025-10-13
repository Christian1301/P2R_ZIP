# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 2 — P2R Training
# Addestra la testa P2R (Point-to-Region) con backbone e ZIP Head congelati
# ============================================================

import os, yaml, random, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from losses.p2r_region_loss import P2RLoss
from train_utils import resume_if_exists, save_checkpoint, setup_experiment, collate_crowd


# ------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------
# Validazione: Loss + MAE + MSE
# ------------------------------------------------------------
def evaluate_p2r(model, loader, device, cfg):
    model.eval()
    total_loss, mae_errors, mse_errors = 0.0, [], []
    p2r_loss_fn = P2RLoss().to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validating]"):
            img = batch["image"].to(device)
            points = [p.to(device) for p in batch["points"]]

            out = model(img)
            loss = p2r_loss_fn(out["density"], points, down=16)
            total_loss += loss.item()

            # Conteggi predetti e reali
            pred_count = torch.sum(out["density"], dim=(1, 2, 3))
            gt_count = torch.tensor([len(p) for p in batch["points"]],
                                    dtype=torch.float32, device=device)
            abs_err = torch.abs(pred_count - gt_count)
            sq_err = (pred_count - gt_count) ** 2
            mae_errors.extend(abs_err.cpu().numpy())
            mse_errors.extend(sq_err.cpu().numpy())

    model.train()
    avg_loss = total_loss / len(loader)
    mae = np.mean(mae_errors)
    mse = np.mean(mse_errors)
    return avg_loss, mae, mse


# ------------------------------------------------------------
def main():
    cfg = yaml.safe_load(open("config.yaml"))
    set_seed(cfg["SEED"])
    device = torch.device(cfg["DEVICE"] if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
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

    # --- Carica ZIP pre-addestrato (Stage 1) ---
    zip_ckpt = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_zip", "best_model.pth")
    assert os.path.isfile(zip_ckpt), f"❌ Checkpoint ZIP non trovato: {zip_ckpt}"
    model.load_state_dict(torch.load(zip_ckpt, map_location="cpu"), strict=False)
    print(f"✅ Caricato checkpoint ZIP da: {zip_ckpt}")

    # Congela backbone + ZIP head (solo la P2R head è allenata)
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.zip_head.parameters(): p.requires_grad = False

    # --- Ottimizzatore ---
    params = [
        {"params": model.backbone.parameters(),
         "lr": cfg["OPTIM"].get("LR_BACKBONE", cfg["OPTIM"]["LR"])},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": cfg["OPTIM"]["LR"]}
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg["OPTIM"]["WEIGHT_DECAY"])

    # --- Scheduler ---
    def warmup_lambda(epoch):
        w = cfg["OPTIM"].get("WARMUP_EPOCHS", 0)
        return (epoch + 1) / w if epoch < w else 1.0

    warmup = LambdaLR(opt, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

    # --- Setup esperimento ---
    exp_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"] + "_p2r")
    writer = setup_experiment(exp_dir)
    start_ep, best_val = (1, float("inf"))
    if cfg["OPTIM"]["RESUME_LAST"]:
        start_ep, best_val = resume_if_exists(model, opt, exp_dir, device)

    # Loss function istanziata una sola volta
    p2r_loss_fn = P2RLoss().to(device)

    # ------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------
    epochs = cfg["OPTIM"]["EPOCHS"]
    for ep in range(start_ep, epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(dl_train, desc=f"[P2R] Epoch {ep}/{epochs}")

        for batch in pbar:
            img = batch["image"].to(device)
            points = [p.to(device) for p in batch["points"]]
            out = model(img)
            loss = p2r_loss_fn(out["density"], points, down=16)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total / len(dl_train)
        writer.add_scalar("train/loss", avg_train, ep)
        writer.add_scalar("lr/backbone", opt.param_groups[0]["lr"], ep)
        writer.add_scalar("lr/p2r_head", opt.param_groups[1]["lr"], ep)

        # --- Validazione ---
        if ep % cfg["OPTIM"]["VAL_INTERVAL"] == 0 or ep == epochs:
            val_loss, mae, mse = evaluate_p2r(model, dl_val, device, cfg)
            writer.add_scalar("val/loss", val_loss, ep)
            writer.add_scalar("val/MAE", mae, ep)
            writer.add_scalar("val/MSE", mse, ep)

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            save_checkpoint(model, opt, ep, val_loss, best_val, exp_dir, is_best=is_best)
            if is_best:
                print(f"💾 Checkpoint salvato: {os.path.join(exp_dir, 'best_model.pth')}")
            print(f"Val Loss {val_loss:.4f} | MAE {mae:.2f} | MSE {mse:.2f} | Best {best_val:.4f}")

        # Scheduler step
        if ep <= cfg["OPTIM"].get("WARMUP_EPOCHS", 0):
            warmup.step()
        else:
            cosine.step()

    writer.close()


# ------------------------------------------------------------
if __name__ == "__main__":
    main()