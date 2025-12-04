import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn.functional as F
import warnings
import yaml


# -----------------------------------------------------------
# YAML Loader
# -----------------------------------------------------------
def load_config(config_path: str):
    """Load a YAML configuration file from disk."""
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


# -----------------------------------------------------------
# Seed initialization
# -----------------------------------------------------------
def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------
# Utility: robust float conversion
# -----------------------------------------------------------
def to_float(value, default=None):
    """
    Converte in float valori YAML interpretati come stringhe (es. '1e-4').
    """
    if value is None:
        return float(default)

    if isinstance(value, (float, int)):
        return float(value)

    try:
        return float(value)
    except Exception:
        # fallback: eval per '1e-4'
        try:
            return float(eval(str(value)))
        except Exception:
            raise ValueError(f"Impossibile convertire '{value}' in float.")


# -----------------------------------------------------------
# OPTIMIZER
# -----------------------------------------------------------
def get_optimizer(param_groups, optim_config):
    """
    Costruisce l'optimizer garantendo che LR e WD siano float.
    """

    if not isinstance(param_groups, (list, tuple)):
        param_groups = [{"params": param_groups}]

    # Recupera LR in ordine di priorità
    lr = optim_config.get("LR")
    if lr is None:
        lr = optim_config.get("BASE_LR")
    if lr is None:
        lr = optim_config.get("BACKBONE_LR")
    if lr is None:
        lr = 5e-5  # default

    lr = to_float(lr)
    wd = to_float(optim_config.get("WEIGHT_DECAY", 1e-4), default=1e-4)

    optim_type = optim_config.get("TYPE", "adamw").lower()

    print(f"⚙️  Creazione optimizer: {optim_type.upper()} (lr={lr}, weight_decay={wd})")

    if optim_type == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif optim_type == "adam":
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{optim_type}' non supportato.")


# -----------------------------------------------------------
# SCHEDULER
# -----------------------------------------------------------
def get_scheduler(optimizer, optim_config, max_epochs=None):
    sched_type = optim_config.get("SCHEDULER", "cosine").lower()

    warmup_epochs = int(optim_config.get("WARMUP_EPOCHS", 0))
    min_factor = to_float(optim_config.get("LR_MIN_FACTOR", 0.0), default=0.0)
    min_factor = max(0.0, min(1.0, min_factor))

    if max_epochs is None:
        max_epochs = int(optim_config.get("EPOCHS", 100))

    print(f"📉 Scheduler: {sched_type.upper()} (max_epochs={max_epochs})")

    # --- Multistep ---
    if sched_type == "multistep":
        milestones = optim_config.get("SCHEDULER_STEPS", [])
        gamma = to_float(optim_config.get("SCHEDULER_GAMMA", 0.1), default=0.1)

        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # --- Cosine scheduler ---
    elif sched_type == "cosine":

        def cosine_lambda(epoch):
            progress = float(epoch) / float(max_epochs)
            progress = max(0.0, min(progress, 1.0))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        if warmup_epochs > 0:

            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    warmup_ratio = float(epoch + 1) / float(warmup_epochs)
                    return min_factor + (1.0 - min_factor) * warmup_ratio

                # Cosine dopo warmup
                progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
                progress = max(0.0, min(progress, 1.0))
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            return LambdaLR(optimizer, lr_lambda=warmup_lambda)

        return LambdaLR(optimizer, lr_lambda=cosine_lambda)

    print("⚠️ Nessuno scheduler selezionato.")
    return None


# -----------------------------------------------------------
# Collate function
# -----------------------------------------------------------
def collate_fn(batch, return_meta=False):
    if not batch:
        raise ValueError("collate_fn: batch vuoto")

    item = batch[0]
    meta_list = [] if return_meta else None

    # Caso standard (image + density + points)
    if "density" in item:
        max_h = max(b["image"].shape[1] for b in batch)
        max_w = max(b["image"].shape[2] for b in batch)

        imgs, dens, pts_list = [], [], []

        for b in batch:
            img, den, pts = b["image"], b["density"], b["points"]
            pad = (0, max_w - img.shape[2], 0, max_h - img.shape[1])
            imgs.append(F.pad(img, pad))
            dens.append(F.pad(den, pad))
            pts_list.append(pts)

            if return_meta:
                meta_list.append({
                    "img_path": b.get("img_path"),
                    "orig_shape": tuple(img.shape[-2:]),
                    "points_count": int(pts.shape[0]) if isinstance(pts, torch.Tensor) else 0,
                })

        out = (torch.stack(imgs), torch.stack(dens), pts_list)

    # ZIP blocks (compatibilità vecchia)
    elif "zip_blocks" in item:
        warnings.warn("collate_fn: rilevato formato 'zip_blocks' (deprecated).")
        imgs = torch.stack([b["image"] for b in batch])
        blocks = torch.stack([b["zip_blocks"] for b in batch])
        pts = [b["points"] for b in batch]
        out = (imgs, blocks, pts)

    else:
        raise TypeError(f"collate_fn: formato batch non riconosciuto: {item.keys()}")

    if return_meta:
        return (*out, meta_list)
    return out


# -----------------------------------------------------------
# P2R Grid canonicalization
# -----------------------------------------------------------
def canonicalize_p2r_grid(pred_density, input_hw, default_down, warn_tag=None, warn_tol=0.15):
    if pred_density.ndim != 4:
        raise ValueError("pred_density deve essere [B,1,H,W]")

    h_in, w_in = input_hw
    h_out, w_out = pred_density.shape[-2:]

    if h_out == h_in and w_out == w_in:
        return pred_density, (1.0, 1.0), False

    down_h = float(h_in) / float(h_out)
    down_w = float(w_in) / float(w_out)

    return pred_density, (down_h, down_w), False


# -----------------------------------------------------------
# Experiment setup + checkpoints
# -----------------------------------------------------------
def setup_experiment(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"🧾 Experiment directory: {exp_dir}")
    return SummaryWriter(log_dir=log_dir)


def resume_if_exists(model, optimizer, exp_dir, device):
    ck = os.path.join(exp_dir, "last.pth")
    if os.path.isfile(ck):
        data = torch.load(ck, map_location=device)
        model.load_state_dict(data["model"], strict=False)
        optimizer.load_state_dict(data["opt"])
        print(f"🔄 Resume da epoch {data.get('epoch', 0)}")
        return data.get("epoch", 0) + 1, data.get("best_val", float("inf"))
    print("ℹ️ Nessun checkpoint, partenza da zero.")
    return 1, float("inf")


def save_checkpoint(model, optimizer, epoch, val_metric, best_metric, exp_dir, is_best=False):
    ck = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "best_val": best_metric,
    }

    torch.save(ck, os.path.join(exp_dir, "last.pth"))

    if is_best:
        torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    print(f"💾 Checkpoint salvato — epoch={epoch}, best={best_metric:.2f} ({'BEST' if is_best else 'LAST'})")
