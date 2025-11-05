import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR
import torch.nn.functional as F
import warnings


def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(param_groups, optim_config):
    """
    Costruisce l'optimizer leggendo in modo compatibile i parametri di LR
    (supporta 'LR', 'BASE_LR', e 'BACKBONE_LR').
    """
    if not isinstance(param_groups, (list, tuple)):
        param_groups = [{'params': param_groups}]

    # ✅ Leggi LR in modo flessibile
    lr = (
        optim_config.get("LR")
        or optim_config.get("BASE_LR")
        or optim_config.get("BACKBONE_LR")
        or 5e-5
    )
    wd = optim_config.get("WEIGHT_DECAY", 1e-4)
    optimizer_type = optim_config.get("TYPE", "adamw").lower()

    print(f"⚙️  Creazione optimizer: {optimizer_type.upper()} (lr={lr}, weight_decay={wd})")

    if optimizer_type == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif optimizer_type == "adam":
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' non supportato.")

def get_scheduler(optimizer, optim_config, max_epochs):
    scheduler_type = optim_config.get('SCHEDULER', 'cosine').lower()
    warmup_epochs = optim_config.get('WARMUP_EPOCHS', 0)

    print(f"📉 Scheduler attivo: {scheduler_type.upper()} (max_epochs={max_epochs})")

    if scheduler_type == 'multistep':
        if warmup_epochs > 0:
            print("⚠️ Warmup non tipico con MultiStepLR.")
        milestones = optim_config.get('SCHEDULER_STEPS', [])
        gamma = optim_config.get('SCHEDULER_GAMMA', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'cosine':
        if warmup_epochs > 0:
            def warmup_cosine_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(max(1, warmup_epochs))
                else:
                    progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            return LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        else:
            return CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)

    return None


def collate_fn(batch):
    item = batch[0]

    if 'density' in item and isinstance(item['points'], torch.Tensor):
        max_h = max(b['image'].shape[1] for b in batch)
        max_w = max(b['image'].shape[2] for b in batch)

        padded_images, padded_densities, points_list = [], [], []

        for b in batch:
            img, den, pts = b['image'], b['density'], b['points']
            pad = (0, max_w - img.shape[2], 0, max_h - img.shape[1])
            padded_images.append(F.pad(img, pad))
            padded_densities.append(F.pad(den, pad))
            points_list.append(pts)

        return torch.stack(padded_images, 0), torch.stack(padded_densities, 0), points_list

    elif 'zip_blocks' in item and isinstance(item['points'], torch.Tensor):
        warnings.warn("collate_fn: rilevato formato 'zip_blocks' (solo per retrocompatibilità).")
        imgs = torch.stack([b["image"] for b in batch], 0)
        blocks = torch.stack([b["zip_blocks"] for b in batch], 0)
        points = [b["points"] for b in batch]
        return imgs, blocks, points

    raise TypeError(f"Formato batch non riconosciuto in collate_fn: {item.keys()}")


def canonicalize_p2r_grid(pred_density, input_hw, default_down, warn_tag=None):
    """Forza fattori di downsampling interi e ritaglia le mappe P2R se necessario."""
    if not isinstance(input_hw, (tuple, list)) or len(input_hw) != 2:
        raise ValueError(f"input_hw deve essere una coppia (H_in, W_in), trovato {input_hw}")

    if not hasattr(canonicalize_p2r_grid, "_warned_tags"):
        canonicalize_p2r_grid._warned_tags = set()

    h_in, w_in = int(input_hw[0]), int(input_hw[1])
    if h_in <= 0 or w_in <= 0:
        raise ValueError(f"Dimensioni input non valide: H_in={h_in}, W_in={w_in}")

    _, _, h_out, w_out = pred_density.shape

    # Caso 1: mappa già upsamplata all'input (es: Stage 1) → mantieni scala configurata
    if h_out == h_in and w_out == w_in:
        down_val = float(default_down)
        return pred_density, (down_val, down_val), False

    down_h = h_in / max(h_out, 1)
    down_w = w_in / max(w_out, 1)
    down_h_int = max(1, int(round(down_h)))
    down_w_int = max(1, int(round(down_w)))

    mismatch_h = abs(h_in - down_h_int * h_out)
    mismatch_w = abs(w_in - down_w_int * w_out)

    if warn_tag and (mismatch_h > 1.0 or mismatch_w > 1.0):
        key = (warn_tag, down_h_int, down_w_int)
        if key not in canonicalize_p2r_grid._warned_tags:
            print(
                "⚠️ P2R downsampling non allineato '{}': H_in={}, W_in={}, H_out={}, W_out={}, down≈({:.3f},{:.3f}) → ({},{})".format(
                    warn_tag, h_in, w_in, h_out, w_out, down_h, down_w, down_h_int, down_w_int
                )
            )
            canonicalize_p2r_grid._warned_tags.add(key)

    target_h = max(1, int(round(h_in / down_h_int)))
    target_w = max(1, int(round(w_in / down_w_int)))

    trim_h = min(h_out, target_h)
    trim_w = min(w_out, target_w)
    trimmed = (trim_h != h_out) or (trim_w != w_out)
    if trimmed:
        pred_density = pred_density[..., :trim_h, :trim_w]

    return pred_density, (float(down_h_int), float(down_w_int)), trimmed


def setup_experiment(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"🧾 Directory esperimento: {exp_dir}")
    return SummaryWriter(log_dir=log_dir)


def resume_if_exists(model, optimizer, exp_dir, device):
    last_ck = os.path.join(exp_dir, "last.pth")
    if os.path.isfile(last_ck):
        ckpt = torch.load(last_ck, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"✅ Ripreso l'addestramento da {last_ck} (epoch={start_epoch})")
        return start_epoch, best_val
    print("ℹ️ Nessun checkpoint precedente trovato, partenza da zero.")
    return 1, float("inf")


def save_checkpoint(model, optimizer, epoch, val_metric, best_metric, exp_dir, is_best=False):
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "best_val": best_metric,
    }
    torch.save(ckpt, os.path.join(exp_dir, "last.pth"))
    if is_best:
        torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
    print(f"💾 Checkpoint salvato: epoch={epoch}, best_val={best_metric:.2f} ({'best' if is_best else 'last'})")
