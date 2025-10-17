# -*- coding: utf-8 -*-
# ============================================================
# P2R-ZIP: Stage 1 — ZIP Pre-training (config ottimizzato)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml, os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
from train_utils import init_seeds, get_optimizer, get_scheduler, resume_if_exists, save_checkpoint, collate_fn


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 1 (ZIP)")

    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        optimizer.zero_grad()

        predictions = model(images)
        loss, loss_dict = criterion(predictions, gt_density)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}",
            'count': f"{loss_dict['zip_count_loss']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    if scheduler:
        scheduler.step()

    return total_loss / len(dataloader)


# ------------------------------------------------------------
# VALIDATE
# ------------------------------------------------------------
@torch.no_grad()
def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    block_size = criterion.zip_block_size

    for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 1"):
        images, gt_density = images.to(device), gt_density.to(device)
        predictions = model(images)
        loss, _ = criterion(predictions, gt_density)
        total_loss += loss.item()

        # Conteggio atteso corretto: Σ(1−π)*λ
        pred_count = torch.sum((1.0 - predictions["logit_pi_maps"].softmax(dim=1)[:, 0:1]) * predictions["lambda_maps"], dim=(1, 2, 3))
        gt_counts_per_block = F.avg_pool2d(gt_density, kernel_size=block_size) * (block_size ** 2)
        gt_count = torch.sum(gt_counts_per_block, dim=(1, 2, 3))

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_mse = (mse / len(dataloader.dataset)) ** 0.5
    return avg_loss, avg_mae, avg_mse


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])

    optim_cfg = cfg["OPTIM_ZIP"]
    dataset_name = cfg["DATASET"]
    bin_config = cfg["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_config["bins"], bin_config["bin_centers"]

    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_bins=bin_centers
    ).to(device)

    # Congela testa P2R
    for param in model.p2r_head.parameters():
        param.requires_grad = False

    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=cfg["ZIP_LOSS"]["WEIGHT_CE"],
        zip_block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"]
    ).to(device)

    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), {"OPTIM": optim_cfg})
    scheduler = get_scheduler(optimizer, {"OPTIM": optim_cfg}, max_epochs=optim_cfg["EPOCHS"])

    DatasetClass = get_dataset(cfg["DATASET"])
    train_dataset = DatasetClass(root=cfg["DATA"]["ROOT"], split=cfg["DATA"]["TRAIN_SPLIT"])
    val_dataset = DatasetClass(root=cfg["DATA"]["ROOT"], split=cfg["DATA"]["VAL_SPLIT"])

    dl_train = DataLoader(train_dataset, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True, num_workers=optim_cfg["NUM_WORKERS"], collate_fn=collate_fn)
    dl_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=optim_cfg["NUM_WORKERS"], collate_fn=collate_fn)

    exp_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    os.makedirs(exp_dir, exist_ok=True)

    start_ep, best_val = resume_if_exists(model, optimizer, exp_dir, device)

    for ep in range(start_ep, optim_cfg["EPOCHS"] + 1):
        print(f"--- Epoch {ep}/{optim_cfg['EPOCHS']} ---")
        train_loss = train_one_epoch(model, criterion, dl_train, optimizer, scheduler, device)

        if ep % optim_cfg["VAL_INTERVAL"] == 0 or ep == optim_cfg["EPOCHS"]:
            val_loss, mae, rmse = validate(model, criterion, dl_val, device)
            print(f"Epoch {ep}: Train {train_loss:.4f} | Val {val_loss:.4f} | MAE {mae:.2f} | RMSE {rmse:.2f}")

            is_best = mae < best_val
            if is_best:
                best_val = mae

            if cfg["EXP"]["SAVE_BEST"]:
                save_checkpoint(model, optimizer, ep, mae, best_val, exp_dir, is_best=is_best)
                if is_best:
                    print(f"✅ Nuovo best model salvato (MAE={best_val:.2f})")

if __name__ == "__main__":
    main()
