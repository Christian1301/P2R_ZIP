# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 2 — P2R Training (config separato)
# ============================================================

import os, yaml, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from losses.p2r_region_loss import P2RLoss
from train_utils import (
    init_seeds, get_optimizer, get_scheduler, 
    resume_if_exists, save_checkpoint, setup_experiment, collate_fn
)


# ------------------------------------------------------------
@torch.no_grad()
def evaluate_p2r(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    down = cfg["DATA"].get("ZIP_BLOCK_SIZE", 16)

    for images, _, points in tqdm(loader, desc="[Validating Stage 2]"):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        out = model(images)

        loss = loss_fn(out["density"], points_list, down=down)
        total_loss += loss.item()

        pred_count = torch.sum(out["density"], dim=(1, 2, 3))
        gt_count = torch.tensor([len(p) for p in points_list], dtype=torch.float32, device=device)

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()

    n = len(loader.dataset)
    return total_loss / len(loader), mae / n, (mse / n) ** 0.5


# ------------------------------------------------------------
def main():
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio Stage 2 (P2R Training) su {device}")

    optim_cfg = cfg["OPTIM_P2R"]

    # Dataset
    DatasetClass = get_dataset(cfg["DATASET"])
    train_ds = DatasetClass(root=cfg["DATA"]["ROOT"], split=cfg["DATA"]["TRAIN_SPLIT"])
    val_ds = DatasetClass(root=cfg["DATA"]["ROOT"], split=cfg["DATA"]["VAL_SPLIT"])

    dl_train = DataLoader(train_ds, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True, num_workers=optim_cfg["NUM_WORKERS"], drop_last=True, collate_fn=collate_fn)
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=optim_cfg["NUM_WORKERS"], collate_fn=collate_fn)

    # Modello
    dataset_name = cfg["DATASET"]
    bin_config = cfg["BINS_CONFIG"][dataset_name]
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_bins=bin_config["bin_centers"]
    ).to(device)

    # Carica Stage1
    stage1_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    zip_ckpt = os.path.join(stage1_dir, "best_model.pth")
    if not os.path.isfile(zip_ckpt):
        print(f"❌ Checkpoint Stage1 non trovato in {zip_ckpt}")
        return
    model.load_state_dict(torch.load(zip_ckpt, map_location=device), strict=False)
    print(f"✅ Checkpoint Stage1 caricato da {zip_ckpt}")

    # Freeze ZIP
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.zip_head.parameters(): p.requires_grad = False
    for p in model.p2r_head.parameters(): p.requires_grad = True

    # Optim e scheduler
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    opt = get_optimizer(params_to_train, {"OPTIM": optim_cfg})
    scheduler = get_scheduler(opt, {"OPTIM": optim_cfg}, max_epochs=optim_cfg["EPOCHS"])

    # Loss
    p2r_loss_fn = P2RLoss(
        max_radius=cfg["P2R_LOSS"]["MU"],
        cost_point=cfg["P2R_LOSS"]["TAU"]
    ).to(device)

    # Setup esperimento
    exp_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"], "stage2")
    os.makedirs(exp_dir, exist_ok=True)
    writer = setup_experiment(exp_dir)

    start_ep, best_mae = resume_if_exists(model, opt, exp_dir, device)

    # Training
    print(f"🚀 Inizio training Stage2 per {optim_cfg['EPOCHS']} epoche...")
    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"[P2R Train] Epoch {ep}/{optim_cfg['EPOCHS']}")

        for images, _, points in pbar:
            images = images.to(device)
            points_list = [p.to(device) for p in points]

            out = model(images)
            loss = p2r_loss_fn(out["density"], points_list, down=cfg["DATA"]["ZIP_BLOCK_SIZE"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.p2r_head.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}")

        if scheduler:
            scheduler.step()

        avg_train = total_loss / len(dl_train)
        if writer: writer.add_scalar("train/loss", avg_train, ep)

        if ep % optim_cfg["VAL_INTERVAL"] == 0 or ep == optim_cfg["EPOCHS"]:
            val_loss, mae, rmse = evaluate_p2r(model, dl_val, p2r_loss_fn, device, cfg)
            print(f"Epoch {ep}: Train {avg_train:.4f} | Val {val_loss:.4f} | MAE {mae:.2f} | RMSE {rmse:.2f}")

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_path = os.path.join(stage1_dir, "stage2_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"💾 Nuovo best Stage2 salvato in {best_path} (MAE={mae:.2f})")

    if writer: writer.close()
    print("✅ Stage 2 completato.")


if __name__ == "__main__":
    main()