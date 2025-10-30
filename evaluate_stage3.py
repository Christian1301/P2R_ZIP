# -*- coding: utf-8 -*-
"""
Valutazione Stage 3 (joint ZIP + P2R).
Replica la validazione usata durante il training con diagnostica opzionale
su pi/lambda della ZIP head e metriche MAE/RMSE sulla densità P2R.
"""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds


def _round_up_8(x: int) -> int:
    return (x + 7) // 8 * 8


def collate_val(batch):
    """Collate identico a quello usato in Stage 3 (senza punti)."""

    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
    else:
        imgs, dens = zip(*[(s[0], s[1]) for s in batch])

    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out = [], []
    for im, den in zip(imgs, dens):
        _, H, W = im.shape
        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode="bilinear", align_corners=False).squeeze(0)
        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode="bilinear", align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)
        imgs_out.append(im_res)
        dens_out.append(den_res)

    dummy_pts = [None] * len(imgs_out)
    return torch.stack(imgs_out), torch.stack(dens_out), dummy_pts


@torch.no_grad()
def evaluate_joint(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    pi_means, lambda_means = [], []

    for idx, (images, gt_density, _) in enumerate(tqdm(dataloader, desc="[Validate Stage 3]")):
        images, gt_density = images.to(device), gt_density.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_out, w_out = pred_density.shape
        _, _, h_in, w_in = images.shape

        if h_out == 0 or w_out == 0:
            raise ValueError("Output density map has zero spatial dimension.")

        down_h = h_in / max(h_out, 1)
        down_w = w_in / max(w_out, 1)
        if abs(h_in - down_h * h_out) > 1.0 or abs(w_in - down_w * w_out) > 1.0:
            down_h = down_w = float(default_down)

        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        gt_count = torch.sum(gt_density, dim=(1, 2, 3))

        abs_diff = torch.abs(pred_count - gt_count)
        mae += abs_diff.sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()

        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

        if idx == 0:
            print("===== DEBUG STAGE 3 =====")
            print(f"Input size: {h_in}x{w_in}")
            print(f"Output size: {h_out}x{w_out}")
            print(f"Downsampling factor: ({down_h:.2f}x, {down_w:.2f}x)")
            print(f"Density map range: [{pred_density.min().item():.4f}, {pred_density.max().item():.4f}]")
            print(f"Mean density: {pred_density.mean().item():.6f}")
            print(f"[DEBUG] Pred count (scaled): {pred_count[0].item():.2f}, GT count: {gt_count[0].item():.2f}")
            print("=========================")

        # --- Diagnostica opzionale sulla ZIP head ---
        if "logit_pi_maps" in outputs and "lambda_maps" in outputs:
            pi_logits = outputs["logit_pi_maps"]
            pi_prob = torch.softmax(pi_logits, dim=1)[:, 1:]
            lambda_map = outputs["lambda_maps"]
            pi_means.append(pi_prob.mean().item())
            lambda_means.append(lambda_map.mean().item())

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)

    print("\n===== RISULTATI FINALI STAGE 3 =====")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    if total_gt > 0:
        bias = total_pred / total_gt
        print(f"Pred / GT ratio: {bias:.3f} (tot_pred={total_pred:.1f}, tot_gt={total_gt:.1f})")
    if pi_means:
        print(f"Avg π  : {np.mean(pi_means):.3f}")
    if lambda_means:
        print(f"Avg λ  : {np.mean(lambda_means):.3f}")
    print("=====================================\n")


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio valutazione Stage 3 su {device}")

    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]

    upsample_to_input = cfg["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    if upsample_to_input:
        print("ℹ️ Stage 3 eval: forzo UPSAMPLE_TO_INPUT=False per coerenza col training.")
        upsample_to_input = False

    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    ckpt_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    ckpt_path = os.path.join(ckpt_dir, "stage3_best.pth")
    if not os.path.exists(ckpt_path):
        alt = os.path.join(ckpt_dir, "stage3", "last.pth")
        if os.path.exists(alt):
            ckpt_path = alt
    if not os.path.exists(ckpt_path):
        print("❌ Nessun checkpoint Stage 3 trovato.")
        return

    print(f"✅ Caricamento checkpoint da {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"ℹ️ Parametri mancanti ignorati: {missing}")
    if unexpected:
        print(f"ℹ️ Parametri extra ignorati: {unexpected}")

    data_cfg = cfg["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)

    DatasetClass = get_dataset(cfg["DATASET"])
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["OPTIM_JOINT"]["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_val,
    )

    default_down = cfg["DATA"].get("P2R_DOWNSAMPLE", 8)
    evaluate_joint(model, val_loader, device, default_down)


if __name__ == "__main__":
    main()
