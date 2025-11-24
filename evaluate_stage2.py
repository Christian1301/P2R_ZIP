# -*- coding: utf-8 -*-
import argparse
import os
from typing import Optional
from functools import partial
import torch
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from losses.p2r_region_loss import P2RLoss
from train_utils import collate_fn, init_seeds, load_config
from train_stage2_p2r import calibrate_density_scale, evaluate_p2r


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 (P2R)")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to evaluate.")
    return parser.parse_args()


def main(config_path: str, checkpoint_override: Optional[str] = None):
    cfg = load_config(config_path)
    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio valutazione Stage 2 su {device}")

    DatasetClass = get_dataset(cfg["DATASET"])
    data_cfg = cfg["DATA"]
    transforms = build_transforms(data_cfg, is_train=False)
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=transforms
    )
    collate_with_meta = partial(collate_fn, return_meta=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["OPTIM_P2R"]["NUM_WORKERS"],
        collate_fn=collate_with_meta,
        pin_memory=True
    )
    dataset_name = cfg["DATASET"]
    model_cfg = cfg["MODEL"]
    bin_config = cfg["BINS_CONFIG"][dataset_name]
    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        backbone_name=model_cfg["BACKBONE"],
        pi_thresh=model_cfg.get("ZIP_PI_THRESH"),
        gate=model_cfg.get("GATE", "multiply"),
        upsample_to_input=False,  
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        zip_head_kwargs=zip_head_kwargs,
        soft_pi_gate=model_cfg.get("ZIP_PI_SOFT", False),
        pi_gate_power=model_cfg.get("ZIP_PI_SOFT_POWER", 1.0),
        pi_gate_min=model_cfg.get("ZIP_PI_SOFT_MIN", 0.0),
        apply_gate_to_output=model_cfg.get("ZIP_PI_APPLY_TO_P2R", False),
    ).to(device)
    ckpt_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    if checkpoint_override:
        ckpt_path = checkpoint_override
    else:
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
    loss_cfg = cfg.get("P2R_LOSS", {})
    loss_kwargs = {}
    if "SCALE_WEIGHT" in loss_cfg:
        loss_kwargs["scale_weight"] = float(loss_cfg["SCALE_WEIGHT"])
    if "SCALE_PENALTY_HUBER_DELTA" in loss_cfg:
        loss_kwargs["scale_huber_delta"] = float(loss_cfg["SCALE_PENALTY_HUBER_DELTA"])
    if "SCALE_PENALTY_CAP" in loss_cfg:
        loss_kwargs["scale_penalty_cap"] = float(loss_cfg["SCALE_PENALTY_CAP"])
    if "POS_WEIGHT" in loss_cfg:
        loss_kwargs["pos_weight"] = float(loss_cfg["POS_WEIGHT"])
    if "CHUNK_SIZE" in loss_cfg:
        loss_kwargs["chunk_size"] = int(loss_cfg["CHUNK_SIZE"])
    if "MIN_RADIUS" in loss_cfg:
        loss_kwargs["min_radius"] = float(loss_cfg["MIN_RADIUS"])
    if "MAX_RADIUS" in loss_cfg:
        loss_kwargs["max_radius"] = float(loss_cfg["MAX_RADIUS"])
    loss_fn = P2RLoss(**loss_kwargs).to(device)
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    clamp_cfg = loss_cfg.get("LOG_SCALE_CLAMP")
    max_adjust = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")
    calibrate_trim = float(loss_cfg.get("LOG_SCALE_CALIBRATION_TRIM", 0.0))
    calibrate_stat = loss_cfg.get("LOG_SCALE_CALIBRATION_STAT", "median")
    calibrate_min_samples = loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_SAMPLES")
    calibrate_min_bias = loss_cfg.get("LOG_SCALE_CALIBRATION_MIN_BIAS")
    calibrate_max_bias = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_BIAS")
    calibrate_dynamic_floor = loss_cfg.get("LOG_SCALE_DYNAMIC_FLOOR")
    calibrate_damping = loss_cfg.get("LOG_SCALE_CALIBRATION_DAMPING", 1.0)
    calibrate_density_scale(
        model,
        val_loader,
        device,
        default_down,
        max_batches=None,
        clamp_range=clamp_cfg,
        max_adjust=max_adjust,
        min_samples=calibrate_min_samples,
        min_bias=calibrate_min_bias,
        max_bias=calibrate_max_bias,
        trim_ratio=calibrate_trim,
        stat=calibrate_stat,
        dynamic_floor=calibrate_dynamic_floor,
        adjust_damping=calibrate_damping,
    )
    evaluate_p2r(model, val_loader, loss_fn, device, cfg)

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.checkpoint)
