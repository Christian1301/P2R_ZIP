# P2R_ZIP/train_utils.py
import os, torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math

def setup_experiment(exp_dir):
    """
    Crea directory esperimento e writer TensorBoard.
    """
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def resume_if_exists(model, optimizer, exp_dir, device):
    last_ck = os.path.join(exp_dir, "last.pth")
    if os.path.isfile(last_ck):
        ckpt = torch.load(last_ck, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_val", float("inf"))
        print(f"[Resume] Ripreso da {last_ck} (epoch={start_epoch})")
        return start_epoch, best_loss
    return 1, float("inf")

def save_checkpoint(model, optimizer, epoch, val_loss, best_loss, exp_dir, is_best=False):
    os.makedirs(exp_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "best_val": best_loss,
    }
    torch.save(ckpt, os.path.join(exp_dir, "last.pth"))
    if is_best:
        torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

def collate_crowd(batch, block_size=16):
    """
    Collate function per dataset di crowd counting.
    - Uniforma le dimensioni tramite padding.
    - Tiene coerente anche la griglia ZIP con il block_size.
    """
    # 1️⃣ Trova max altezza/larghezza nel batch
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)

    # 2️⃣ Calcola la dimensione target per la griglia ZIP
    Hb = math.ceil(max_h / block_size)
    Wb = math.ceil(max_w / block_size)

    padded_imgs, padded_blocks = [], []
    for b in batch:
        img = b["image"]
        blocks = b["zip_blocks"]

        # === Padding immagine ===
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_imgs.append(img)

        # === Padding ZIP blocks ===
        pad_hb = Hb - blocks.shape[1]
        pad_wb = Wb - blocks.shape[2]
        blocks = F.pad(blocks, (0, pad_wb, 0, pad_hb), mode='constant', value=0)
        padded_blocks.append(blocks)

    # 3️⃣ Stack finale
    images = torch.stack(padded_imgs, dim=0)
    zip_blocks = torch.stack(padded_blocks, dim=0)
    points = [b["points"] for b in batch]
    paths = [b["img_path"] for b in batch]

    return {
        "image": images,
        "zip_blocks": zip_blocks,
        "points": points,
        "img_path": paths
    }