# ============================================
# P2R_ZIP/train_stage1_zip.py (VERSIONE DEFINITIVA — FIXED)
# ============================================

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    resume_if_exists,
    save_checkpoint,
    collate_fn,
    load_config,
)


# -----------------------------------------------------------
# Utility robusta per convertire qualsiasi LR in float
# -----------------------------------------------------------
def parse_lr(x, default=1e-4):
    """
    Converte x in float in modo sicuro.
    Accetta: float, int, '1e-4', "0.0001", ecc.
    """
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float(eval(str(x)))


# -----------------------------------------------------------
# ZIP regularization
# -----------------------------------------------------------
def zip_regularization(preds, gt_count: torch.Tensor, reg_cfg: dict):

    pi_target = float(reg_cfg.get("PI_TARGET", 0.28))
    pi_weight = float(reg_cfg.get("PI_WEIGHT", 0.0))
    lambda_target = float(reg_cfg.get("LAMBDA_TARGET", 1.0))
    lambda_weight = float(reg_cfg.get("LAMBDA_WEIGHT", 0.0))
    count_weight = float(reg_cfg.get("COUNT_WEIGHT", 0.0))

    pi_maps = preds["logit_pi_maps"].softmax(dim=1)
    pi_zero = pi_maps[:, 0:1]
    pi_occ = 1.0 - pi_zero

    lam = preds["lambda_maps"]
    reg_loss = lam.new_tensor(0.0)

    # 1) Sparsità π
    pi_mean = pi_occ.mean()
    if pi_weight > 0:
        pi_excess = F.relu(pi_mean - pi_target)
        loss_pi = (pi_excess ** 2) * pi_weight
        reg_loss += loss_pi
    else:
        loss_pi = lam.new_tensor(0.0)

    # 2) Reg λ
    lambda_mean = lam.mean()
    if lambda_weight > 0:
        loss_lambda = (lambda_mean - lambda_target) ** 2 * lambda_weight
        reg_loss += loss_lambda
    else:
        loss_lambda = lam.new_tensor(0.0)

    # 3) Count L1 ZIP
    zip_count_pred = (pi_occ * lam).sum(dim=(1, 2, 3))
    if gt_count.ndim > 1:
        gt_count = gt_count.view(-1)

    gt_count = gt_count.to(zip_count_pred.device)
    if count_weight > 0:
        loss_count = F.smooth_l1_loss(zip_count_pred, gt_count) * count_weight
        reg_loss += loss_count
    else:
        loss_count = lam.new_tensor(0.0)

    return reg_loss, {
        "pi_mean": float(pi_mean),
        "lambda_mean": float(lambda_mean),
        "loss_pi": float(loss_pi),
        "loss_lambda": float(loss_lambda),
        "loss_count": float(loss_count),
        "loss_reg": float(reg_loss),
    }


# -----------------------------------------------------------
# Count sampler
# -----------------------------------------------------------
def build_count_sampler(dataset, cfg):
    if not cfg or not cfg.get("ENABLE", False):
        return None
    if not hasattr(dataset, "image_list"):
        return None

    counts = []
    for p in dataset.image_list:
        try:
            counts.append(len(dataset.load_points(p)))
        except:
            counts.append(0)

    c = np.array(counts, np.float32)
    w = np.log1p(c + cfg.get("LOG_OFFSET", 1.0))
    w = np.power(w, cfg.get("POWER", 1.0))
    w = np.clip(w, 1e-6, None)

    return WeightedRandomSampler(
        torch.tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True
    )


# -----------------------------------------------------------
# Train epoch
# -----------------------------------------------------------
def train_one_epoch(model, criterion, loader, optimizer, scheduler, device, reg_cfg, clip_grad):

    model.train()
    total_loss = 0.0

    for images, gt_density, _ in tqdm(loader, desc="Train Stage1 (ZIP)"):

        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()

        preds = model(images)
        base_loss, loss_dict = criterion(preds, gt_density)

        gt_count = gt_density.sum(dim=(1, 2, 3))
        reg_loss, _ = zip_regularization(preds, gt_count, reg_cfg)

        loss = base_loss + reg_loss
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        total_loss += loss.item()

    if scheduler:
        scheduler.step()

    return total_loss / len(loader)


# -----------------------------------------------------------
# Validate
# -----------------------------------------------------------
def validate(model, criterion, loader, device):

    model.eval()
    total_loss = 0
    mae = 0
    mse = 0

    with torch.no_grad():
        for images, gt_density, _ in loader:

            images, gt_density = images.to(device), gt_density.to(device)

            preds = model(images)
            loss, _ = criterion(preds, gt_density)
            total_loss += loss.item()

            pi = preds["logit_pi_maps"].softmax(dim=1)
            pi_occ = 1.0 - pi[:, 0:1]
            lam = preds["lambda_maps"]

            pred_count = (pi_occ * lam).sum(dim=(1, 2, 3))
            gt_count = gt_density.sum(dim=(1, 2, 3))

            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += torch.pow(pred_count - gt_count, 2).sum().item()

    n = len(loader.dataset)
    return (
        total_loss / len(loader),
        mae / n,
        (mse / n) ** 0.5
    )


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(config_path):

    config = load_config(config_path)
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    dataset_name = config["DATASET"]
    data_cfg = config["DATA"]
    model_cfg = config["MODEL"]

    bins = config["BINS_CONFIG"][dataset_name]["bins"]
    centers = config["BINS_CONFIG"][dataset_name]["bin_centers"]

    # ZIP HEAD
    zh = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": parse_lr(zh.get("LAMBDA_SCALE", 0.5)),
        "lambda_max": parse_lr(zh.get("LAMBDA_MAX", 8.0)),
        "use_softplus": bool(zh.get("USE_SOFTPLUS", True)),
        "lambda_noise_std": parse_lr(zh.get("LAMBDA_NOISE_STD", 0.0)),
    }

    # MODEL
    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=centers,
        backbone_name=model_cfg["BACKBONE"],
        pi_thresh=model_cfg.get("ZIP_PI_THRESH"),
        gate=model_cfg.get("GATE", "multiply"),
        upsample_to_input=model_cfg.get("UPSAMPLE_TO_INPUT", True),
        zip_head_kwargs=zip_head_kwargs,
        soft_pi_gate=model_cfg.get("ZIP_PI_SOFT", False),
        pi_gate_power=model_cfg.get("ZIP_PI_SOFT_POWER", 1.0),
        pi_gate_min=model_cfg.get("ZIP_PI_SOFT_MIN", 0.0),
        apply_gate_to_output=model_cfg.get("ZIP_PI_APPLY_TO_P2R", False),
    ).to(device)

    # Freeze P2R head
    for p in model.p2r_head.parameters():
        p.requires_grad = False

    # Loss ZIP
    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config["ZIP_LOSS"]["WEIGHT_CE"],
        weight_nll=config["ZIP_LOSS"]["WEIGHT_NLL"],
        zip_block_size=data_cfg["ZIP_BLOCK_SIZE"],
        count_weight=config["ZIP_LOSS"]["WEIGHT_COUNT"],
    ).to(device)

    # OPTIMIZER
    optim_cfg = config["OPTIM_ZIP"]

    lr_head = parse_lr(optim_cfg.get("LR", optim_cfg.get("BASE_LR", 5e-5)))
    lr_backbone = parse_lr(optim_cfg.get("LR_BACKBONE", lr_head * 0.5))
    clip_grad = parse_lr(optim_cfg.get("CLIP_GRAD_NORM", 1.0))

    optimizer = get_optimizer(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.zip_head.parameters(), "lr": lr_head},
        ],
        optim_cfg,
    )
    scheduler = get_scheduler(optimizer, optim_cfg)

    # DATA
    train_tf = build_transforms(
        data_cfg,
        is_train=True,
        override_crop_size=data_cfg.get("CROP_SIZE_STAGE1"),
        override_crop_scale=data_cfg.get("CROP_SCALE_STAGE1"),
    )
    val_tf = build_transforms(data_cfg, is_train=False)

    Dataset = get_dataset(dataset_name)

    train_set = Dataset(data_cfg["ROOT"], data_cfg["TRAIN_SPLIT"],
                        data_cfg["ZIP_BLOCK_SIZE"], train_tf)
    val_set = Dataset(data_cfg["ROOT"], data_cfg["VAL_SPLIT"],
                      data_cfg["ZIP_BLOCK_SIZE"], val_tf)

    sampler = build_count_sampler(train_set, data_cfg.get("COUNT_SAMPLER"))

    train_loader = DataLoader(
        train_set,
        batch_size=optim_cfg["BATCH_SIZE"],
        sampler=sampler if sampler else None,
        shuffle=sampler is None,
        num_workers=optim_cfg["NUM_WORKERS"],
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=optim_cfg["NUM_WORKERS"]
    )

    # Resume
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)
    resume = optim_cfg.get("RESUME_LAST", True)
    start_epoch, best_mae = resume_if_exists(model, optimizer, out_dir, device) if resume else (1, float("inf"))

    max_epochs = int(optim_cfg["EPOCHS"])
    val_interval = int(optim_cfg["VAL_INTERVAL"])
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", None)
    no_improve = 0

    print(f"🚀 Stage 1 Training START (max {max_epochs} epochs)")

    for epoch in range(start_epoch, max_epochs + 1):

        print(f"\n----- EPOCH {epoch}/{max_epochs} -----")

        train_loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scheduler, device,
            config.get("ZIP_REG", {}), clip_grad
        )

        if epoch % val_interval == 0:
            val_loss, val_mae, val_rmse = validate(model, criterion, val_loader, device)
            print(f"VAL → loss={val_loss:.4f}, MAE={val_mae:.2f}, RMSE={val_rmse:.2f}")

            is_best = val_mae < best_mae
            save_checkpoint(model, optimizer, epoch, val_mae, best_mae, out_dir, is_best=is_best)

            if is_best:
                best_mae = val_mae
                no_improve = 0
                print(f"🔥 NEW BEST — MAE {best_mae:.2f}")
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    print("⛔ EARLY STOPPING")
                    break

    print("✅ Stage 1 COMPLETED")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config.yaml")
    return a.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
