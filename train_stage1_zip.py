# P2R_ZIP/train_stage1_zip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, get_optimizer, get_scheduler, resume_if_exists, save_checkpoint, collate_fn


# ============================================================
# 🔧 Funzione di regolarizzazione extra per ZIP Head
# Penalizza pi saturato (vicino a 0 o 1) e lambda troppo costante
# ============================================================
def zip_regularization(preds, lam_weight=1e-3, pi_weight=1e-3):
    logit_pi = preds["logit_pi_maps"]
    lam = preds["lambda_maps"]

    pi_soft = logit_pi.softmax(dim=1)[:, 1:]  # prob blocco occupato
    pi_reg = ((pi_soft - 0.5) ** 2).mean()  # penalizza saturazione (vicino a 0 o 1)
    lam_reg = lam.std()  # incoraggia variazione spaziale

    return pi_weight * pi_reg - lam_weight * lam_reg  # lam_reg con segno - per favorire varianza


# ============================================================
# 🔹 Training loop con diagnostica
# ============================================================
def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 1 (ZIP)")

    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, gt_density)

        # Regolarizzazione extra
        reg_loss = zip_regularization(predictions)
        total = loss + reg_loss

        total.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += total.item()

        progress_bar.set_postfix({
            'loss': f"{total.item():.4f}",
            'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}",
            'count': f"{loss_dict['zip_count_loss']:.4f}",
            'lr_head': f"{optimizer.param_groups[-1]['lr']:.6f}"
        })

    if scheduler:
        scheduler.step()
    return total_loss / len(dataloader)


# ============================================================
# 🔹 Validazione
# ============================================================
def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    block_size = criterion.zip_block_size

    with torch.no_grad():
        for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 1"):
            images, gt_density = images.to(device), gt_density.to(device)

            preds = model(images)
            loss, _ = criterion(preds, gt_density)
            total_loss += loss.item()

            pi_maps = preds["logit_pi_maps"].softmax(dim=1)
            pi_zero = pi_maps[:, 0:1]
            lam = preds["lambda_maps"]

            pred_count = torch.sum((1 - pi_zero) * lam, dim=(1, 2, 3))
            gt_count = torch.sum(
                F.avg_pool2d(gt_density, kernel_size=block_size) * (block_size ** 2),
                dim=(1, 2, 3)
            )

            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_rmse = (mse / len(dataloader.dataset)) ** 0.5
    return avg_loss, avg_mae, avg_rmse


# ============================================================
# 🔹 Main training loop
# ============================================================
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_cfg["bins"], bin_cfg["bin_centers"]

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=config["MODEL"]["UPSAMPLE_TO_INPUT"],
    ).to(device)

    # Congela la P2R Head
    for p in model.p2r_head.parameters():
        p.requires_grad = False

    # Per Stage 1: rinforza il CE loss
    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config["ZIP_LOSS"]["WEIGHT_CE"] * 2.0,  # 🔥 CE più pesante
        zip_block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
    ).to(device)

    optim_cfg = config["OPTIM_ZIP"]
    # Compatibilità con diversi nomi di chiave nel config
    lr_head = optim_cfg.get("LR", optim_cfg.get("BASE_LR", 5e-5))
    lr_backbone = optim_cfg.get("LR_BACKBONE", optim_cfg.get("BACKBONE_LR", lr_head * 0.5))

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.zip_head.parameters() if p.requires_grad]

    optimizer = get_optimizer(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        optim_cfg,
    )

    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg.get("EPOCHS", 1300))

    # Datasets e DataLoader
    data_cfg = config["DATA"]
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])

    train_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf,
    )
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)
    start_epoch, best_mae = resume_if_exists(model, optimizer, out_dir, device)

    print(f"🚀 Inizio addestramento Stage 1 per {optim_cfg['EPOCHS']} epoche...")

    for epoch in range(start_epoch, optim_cfg["EPOCHS"] + 1):
        print(f"\n--- Epoch {epoch}/{optim_cfg['EPOCHS']} ---")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device)

        if epoch % optim_cfg["VAL_INTERVAL"] == 0 or epoch == optim_cfg["EPOCHS"]:
            val_loss, val_mae, val_rmse = validate(model, criterion, val_loader, device)
            print(f"Val → Loss {val_loss:.4f}, MAE {val_mae:.2f}, RMSE {val_rmse:.2f}")

            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae

            save_checkpoint(model, optimizer, epoch, val_mae, best_mae, out_dir, is_best=is_best)
            if is_best:
                print(f"✅ Saved new best model (MAE={best_mae:.2f})")

    print("✅ Addestramento Stage 1 completato.")


if __name__ == "__main__":
    main()