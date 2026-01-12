# train_utils.py
"""
Train utilities con correzioni per padding consistente e gestione Tuple/Dict.

Modifiche:
1. collate_fn supporta Tuple (img, den, pts) dal nuovo BaseDataset
2. Padding a multipli di 16 (per VGG backbone)
3. Calibrazione per-batch robusta
"""

import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
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
    """Costruisce l'optimizer."""
    if not isinstance(param_groups, (list, tuple)):
        param_groups = [{'params': param_groups}]

    lr = (
        optim_config.get("LR")
        or optim_config.get("BASE_LR")
        or optim_config.get("BACKBONE_LR")
        or 5e-5
    )
    wd = optim_config.get("WEIGHT_DECAY", 1e-4)
    optimizer_type = optim_config.get("TYPE", "adamw").lower()

    print(f"⚙️  Optimizer: {optimizer_type.upper()} (lr={lr}, wd={wd})")

    if optimizer_type == "adamw":
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif optimizer_type == "adam":
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' non supportato.")


def get_scheduler(optimizer, optim_config, max_epochs):
    scheduler_type = optim_config.get('SCHEDULER', 'cosine').lower()
    warmup_epochs = optim_config.get('WARMUP_EPOCHS', 0)
    min_factor = float(optim_config.get('LR_MIN_FACTOR', 0.0))
    min_factor = max(0.0, min(1.0, min_factor))

    print(f"📉 Scheduler: {scheduler_type.upper()} (max_epochs={max_epochs})")

    if scheduler_type == 'multistep':
        milestones = optim_config.get('SCHEDULER_STEPS', [])
        gamma = optim_config.get('SCHEDULER_GAMMA', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'cosine':
        def warmup_cosine_lambda(current_epoch):
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                warmup_ratio = float(current_epoch + 1) / float(max(1, warmup_epochs))
                return min_factor + (1.0 - min_factor) * warmup_ratio
            
            effective_epoch = current_epoch - warmup_epochs
            effective_max = max(1, max_epochs - warmup_epochs)
            progress = float(effective_epoch + 1) / float(effective_max)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    return None


def _round_to_multiple(x: int, multiple: int = 16) -> int:
    """Arrotonda x al multiplo superiore più vicino."""
    return ((x + multiple - 1) // multiple) * multiple


def collate_fn(batch):
    """
    Collate function ROBUSTA per Tuple e Dict.
    Gestisce il padding a multipli di 16 per VGG.
    """
    if len(batch) == 0:
        raise ValueError("Batch vuoto in collate_fn")
        
    item = batch[0]

    # === CASO 1: TUPLA (Nuovo BaseDataset) ===
    # Formato atteso: (image, density, points)
    if isinstance(item, tuple):
        # Calcola dimensioni massime per il padding
        # batch[i][0] è l'immagine
        max_h = max(b[0].shape[1] for b in batch)
        max_w = max(b[0].shape[2] for b in batch)
        
        # Arrotonda a 16 per VGG (evita errori di dimensione dopo pooling)
        max_h = _round_to_multiple(max_h, 16)
        max_w = _round_to_multiple(max_w, 16)

        padded_images = []
        padded_densities = []
        points_list = []

        for b in batch:
            img, den, pts = b
            h, w = img.shape[1], img.shape[2]
            
            # Padding (left, right, top, bottom)
            pad_w = max_w - w
            pad_h = max_h - h
            pad = (0, pad_w, 0, pad_h)
            
            # Pad immagine
            padded_images.append(F.pad(img, pad, mode='constant', value=0))
            
            # Pad density (se esiste, altrimenti placeholder)
            if den is not None and den.numel() > 0:
                padded_densities.append(F.pad(den, pad, mode='constant', value=0))
            else:
                # Se density è mancante o vuota, crea zeri
                padded_densities.append(torch.zeros((1, max_h, max_w), device=img.device))
                
            points_list.append(pts)

        return torch.stack(padded_images, 0), torch.stack(padded_densities, 0), points_list

    # === CASO 2: DIZIONARIO (Legacy / Altri Dataset) ===
    elif isinstance(item, dict):
        if 'density' in item:
            # Trova dimensioni massime
            max_h = max(b['image'].shape[1] for b in batch)
            max_w = max(b['image'].shape[2] for b in batch)
            
            max_h = _round_to_multiple(max_h, 16)
            max_w = _round_to_multiple(max_w, 16)

            padded_images, padded_densities, points_list = [], [], []

            for b in batch:
                img, den, pts = b['image'], b['density'], b['points']
                h, w = img.shape[1], img.shape[2]
                
                pad_w = max_w - w
                pad_h = max_h - h
                pad = (0, pad_w, 0, pad_h)
                
                padded_images.append(F.pad(img, pad, value=0))
                padded_densities.append(F.pad(den, pad, value=0))
                points_list.append(pts)

            return torch.stack(padded_images, 0), torch.stack(padded_densities, 0), points_list

    # Caso non gestito
    raise TypeError(f"Formato batch non riconosciuto. Atteso Tuple o Dict, ricevuto: {type(item)}")


def canonicalize_p2r_grid(pred_density, input_hw, default_down, warn_tag=None, warn_tol=0.15):
    """Calcola i fattori di downsampling effettivi."""
    if not hasattr(canonicalize_p2r_grid, "_warned_tags"):
        canonicalize_p2r_grid._warned_tags = set()

    h_in, w_in = int(input_hw[0]), int(input_hw[1])
    _, _, h_out, w_out = pred_density.shape

    if h_out == h_in and w_out == w_in:
        return pred_density, (1.0, 1.0), False

    down_h = float(h_in) / float(h_out)
    down_w = float(w_in) / float(w_out)

    if isinstance(default_down, (tuple, list)) and len(default_down) == 2:
        ref_down_h, ref_down_w = float(default_down[0]), float(default_down[1])
    else:
        ref_val = float(default_down) if default_down else 1.0
        ref_down_h = ref_down_w = ref_val

    if warn_tag:
        mismatch_h = abs(down_h - ref_down_h) / max(ref_down_h, 1e-6)
        mismatch_w = abs(down_w - ref_down_w) / max(ref_down_w, 1e-6)
        if (mismatch_h > warn_tol or mismatch_w > warn_tol) and (warn_tag not in canonicalize_p2r_grid._warned_tags):
            print(f"⚠️ Downsampling atipico '{warn_tag}': "
                  f"H_in={h_in}, W_in={w_in}, H_out={h_out}, W_out={w_out}, "
                  f"down=({down_h:.3f},{down_w:.3f}) vs ref=({ref_down_h:.3f},{ref_down_w:.3f})")
            canonicalize_p2r_grid._warned_tags.add(warn_tag)

    return pred_density, (down_h, down_w), False


@torch.no_grad()
def calibrate_density_scale_v2(
    model,
    loader,
    device,
    default_down,
    max_batches=None,
    clamp_range=None,
    max_adjust=2.0,
    bias_eps=0.05,
    verbose=True,
):
    """
    Calibrazione scala per P2R.
    """
    if not hasattr(model, "p2r_head") or not hasattr(model.p2r_head, "log_scale"):
        return None

    model.eval()
    pred_counts = []
    gt_counts = []
    ratios = []

    for batch_idx, batch_data in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        
        # Unpack flessibile (gestisce sia (img, den, pts) che (img, pts) se den manca)
        if len(batch_data) == 3:
            images, _, points = batch_data
        else:
            # Fallback se collate cambia
            images = batch_data[0]
            points = batch_data[-1]

        images = images.to(device)
        points_list = [p.to(device) for p in points]

        outputs = model(images)
        pred_density = outputs.get("p2r_density", outputs.get("density"))
        if pred_density is None:
            continue

        _, _, H_in, W_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (H_in, W_in), default_down
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w

        # Per-image analysis
        for i, pts in enumerate(points_list):
            gt = len(pts)
            if gt == 0:
                continue
            
            pred = (pred_density[i].sum() / cell_area).item()
            pred_counts.append(pred)
            gt_counts.append(gt)
            ratios.append(pred / gt)

    if len(gt_counts) == 0:
        print("ℹ️ Calibrazione saltata: nessun dato valido")
        return None

    # Statistiche dettagliate
    ratios_np = np.array(ratios)
    
    bias_median = np.median(ratios_np)
    
    if verbose:
        print(f"\n📊 Statistiche Calibrazione:")
        print(f"   Bias mediano: {bias_median:.3f}")

    # Usa il bias mediano per robustezza agli outlier
    bias = bias_median
    
    if abs(bias - 1.0) < bias_eps:
        print(f"ℹ️ Calibrazione: bias già accettabile ({bias:.3f}, soglia ±{bias_eps})")
        return bias

    # Applica correzione
    prev_log_scale = float(model.p2r_head.log_scale.detach().item())
    raw_adjust = float(np.log(bias))
    
    if max_adjust is not None:
        adjust = float(np.clip(raw_adjust, -max_adjust, max_adjust))
    else:
        adjust = raw_adjust
    
    model.p2r_head.log_scale.data -= torch.tensor(adjust, device=device)
    
    if clamp_range is not None:
        min_val, max_val = float(clamp_range[0]), float(clamp_range[1])
        model.p2r_head.log_scale.data.clamp_(min_val, max_val)
    
    new_log_scale = float(model.p2r_head.log_scale.detach().item())
    new_scale = float(torch.exp(model.p2r_head.log_scale.detach()).item())
    
    print(f"🔧 Calibrazione: bias={bias:.3f} → log_scale {prev_log_scale:.4f}→{new_log_scale:.4f} (scala={new_scale:.4f})")
    
    return bias


def setup_experiment(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"🧾 Directory: {exp_dir}")
    return SummaryWriter(log_dir=log_dir)


def resume_if_exists(model, optimizer, exp_dir, device):
    last_ck = os.path.join(exp_dir, "last.pth")
    if os.path.isfile(last_ck):
        ckpt = torch.load(last_ck, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"✅ Resume da {last_ck} (epoch={start_epoch})")
        return start_epoch, best_val
    print("ℹ️ Partenza da zero")
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
    print(f"💾 Checkpoint: epoch={epoch}, best={best_metric:.2f} ({'best' if is_best else 'last'})")