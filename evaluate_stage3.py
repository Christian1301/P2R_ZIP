# -*- coding: utf-8 -*-
"""
Valutazione Stage 3 (joint ZIP + P2R).
Replica la validazione usata durante il training con diagnostica opzionale
su pi/lambda della ZIP head e metriche MAE/RMSE sulla densità P2R.
"""

import argparse
import os
from typing import Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, load_config
from losses.composite_loss import ZIPCompositeLoss
from losses.p2r_region_loss import P2RLoss

def _round_up_8(x: int) -> int:
    return (x + 7) // 8 * 8

def collate_val(batch):
    """Collate identico a quello usato in Stage 3 (MA CON I PUNTI)."""
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
        pts = [b.get("points", torch.zeros((0,2))) for b in batch]
    else:
        imgs, dens, pts = zip(*[(s[0], s[1], s[2]) for s in batch])

    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out, pts_out = [], [], []
    for im, den, p in zip(imgs, dens, pts): 
        _, H, W = im.shape
        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode="bilinear", align_corners=False).squeeze(0)
        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode="bilinear", align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)
        imgs_out.append(im_res)
        dens_out.append(den_res)
        pts_out.append(p if p is not None else torch.zeros((0,2)))

    return torch.stack(imgs_out), torch.stack(dens_out), pts_out

@torch.no_grad()
def evaluate_joint(
    model,
    dataloader,
    device,
    default_down,
    report_loss: bool = False,
    loss_modules: Optional[Tuple[ZIPCompositeLoss, P2RLoss, float, float]] = None,
):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0
    pi_means, lambda_means = [], []
    zip_loss_accum = 0.0
    p2r_loss_accum = 0.0
    combined_loss_accum = 0.0

    for idx, (images, gt_density, points) in enumerate(tqdm(dataloader, desc="[Validate Stage 3]")):
        images, gt_density = images.to(device), gt_density.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_eval"
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        cell_area_tensor = pred_density.new_tensor(cell_area)
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area_tensor

        gt_count_list = []
        points_for_loss = []
        for p in points:
            if p is None:
                gt_count_list.append(0)
                points_for_loss.append(torch.zeros((0, 2), device=device))
            else:
                gt_count_list.append(int(p.shape[0]))
                points_for_loss.append(p.to(device))
        if not gt_count_list:
            gt_count_list = [0]
            points_for_loss = [torch.zeros((0, 2), device=device)]
        gt_count = torch.tensor(gt_count_list, dtype=torch.float32, device=device)

        abs_diff = torch.abs(pred_count - gt_count)
        mae += abs_diff.sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()

        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

        if report_loss and loss_modules is not None:
            criterion_zip, criterion_p2r, alpha, zip_scale = loss_modules
            loss_zip, _ = criterion_zip(outputs, gt_density)
            scaled_zip = loss_zip * zip_scale
            loss_p2r = criterion_p2r(pred_density, points_for_loss, down=down_tuple)
            combined = scaled_zip + alpha * loss_p2r
            zip_loss_accum += scaled_zip.item()
            p2r_loss_accum += loss_p2r.item()
            combined_loss_accum += combined.item()

        logit_pi = outputs.get("logit_pi_maps")
        if logit_pi is not None:
            pi_soft = logit_pi.softmax(dim=1)[:, 1:]
            pi_means.append(pi_soft.mean().item())
        lambda_maps = outputs.get("lambda_maps")
        if lambda_maps is not None:
            lambda_means.append(lambda_maps.mean().item())

        if idx == 0:
            print("===== DEBUG STAGE 3 =====")
            print(f"Input size: {h_in}x{w_in}")
            h_out, w_out = pred_density.shape[-2], pred_density.shape[-1]
            print(f"Output size: {h_out}x{w_out}")
            print(f"Downsampling factor: ({down_h:.2f}x, {down_w:.2f}x)")
            print(f"Density map range: [{pred_density.min().item():.4f}, {pred_density.max().item():.4f}]")
            print(f"Mean density: {pred_density.mean().item():.6f}")
            print(f"[DEBUG] Pred count (scaled): {pred_count[0].item():.2f}, GT count: {gt_count[0].item():.2f}")
            print("=========================")

    num_samples = max(1, len(dataloader.dataset))
    mae /= num_samples
    rmse = np.sqrt(mse / num_samples)

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
    if report_loss and loss_modules is not None:
        num_batches = max(1, len(dataloader))
        print(
            "Loss media/batch — totale: {:.4f}, zip: {:.4f}, p2r: {:.4f}".format(
                combined_loss_accum / num_batches,
                zip_loss_accum / num_batches,
                p2r_loss_accum / num_batches,
            )
        )
    print("=====================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 (Joint)")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for evaluation.")
    parser.add_argument(
        "--report-loss",
        action="store_true",
        help="Se impostato, calcola anche le loss ZIP/P2R e la loro combinazione media",
    )
    return parser.parse_args()


def main(config_path: str, checkpoint_override: Optional[str] = None, report_loss: bool = False):
    cfg = load_config(config_path)

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
        pi_mode=cfg["MODEL"].get("ZIP_PI_MODE", "hard"),
        pi_soft_gamma=cfg["MODEL"].get("ZIP_SOFT_GAMMA", 1.0),
        detach_pi_mask=cfg["MODEL"].get("ZIP_PI_DETACH", False),
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    ckpt_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    if checkpoint_override:
        ckpt_path = checkpoint_override
    else:
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
    loss_modules = None
    if report_loss:
        joint_cfg = cfg.get("JOINT_LOSS", {})
        zip_loss = ZIPCompositeLoss(
            bins=bin_cfg["bins"],
            weight_ce=cfg["ZIP_LOSS"]["WEIGHT_CE"],
            zip_block_size=cfg["DATA"]["ZIP_BLOCK_SIZE"],
            count_weight=joint_cfg.get(
                "COUNT_L1_W",
                cfg["ZIP_LOSS"].get("WEIGHT_COUNT", 1.0),
            ),
        ).to(device)
        loss_kwargs = {}
        loss_cfg = cfg.get("P2R_LOSS", {})
        if "SCALE_WEIGHT" in loss_cfg:
            loss_kwargs["scale_weight"] = float(loss_cfg["SCALE_WEIGHT"])
        if "POS_WEIGHT" in loss_cfg:
            loss_kwargs["pos_weight"] = float(loss_cfg["POS_WEIGHT"])
        if "CHUNK_SIZE" in loss_cfg:
            loss_kwargs["chunk_size"] = int(loss_cfg["CHUNK_SIZE"])
        if "MIN_RADIUS" in loss_cfg:
            loss_kwargs["min_radius"] = float(loss_cfg["MIN_RADIUS"])
        if "MAX_RADIUS" in loss_cfg:
            loss_kwargs["max_radius"] = float(loss_cfg["MAX_RADIUS"])
        p2r_loss = P2RLoss(**loss_kwargs).to(device)
        alpha = float(joint_cfg.get("ALPHA", 1.0))
        zip_scale = float(joint_cfg.get("ZIP_SCALE", 1.0))
        loss_modules = (zip_loss, p2r_loss, alpha, zip_scale)

    evaluate_joint(
        model,
        val_loader,
        device,
        default_down,
        report_loss=report_loss,
        loss_modules=loss_modules,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.checkpoint, report_loss=args.report_loss)
