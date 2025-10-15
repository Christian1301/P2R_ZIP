# P2R_ZIP/train_utils.py
import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.nn.functional as F

def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(params, config):
    optim_config = config['OPTIM']
    lr = optim_config['LR']
    wd = optim_config['WEIGHT_DECAY']
    optimizer_type = optim_config.get('TYPE', 'adamw').lower()
    if optimizer_type == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif optimizer_type == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' non supportato.")

def get_scheduler(optimizer, config, max_epochs):
    optim_config = config['OPTIM']
    scheduler_type = optim_config.get('SCHEDULER', 'cosine').lower()
    warmup_epochs = optim_config.get('WARMUP_EPOCHS', 0)
    if scheduler_type == 'cosine':
        if warmup_epochs > 0:
            def warmup_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(warmup_epochs)
                progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
            return LambdaLR(optimizer, lr_lambda=warmup_lambda)
        else:
            return CosineAnnealingLR(optimizer, T_max=max_epochs)
    return None

def collate_fn(batch):
    """
    Collate function per gestire un batch di dizionari con campioni di dimensioni diverse.
    """
    max_h = max(item['image'].shape[1] for item in batch)
    max_w = max(item['image'].shape[2] for item in batch)

    padded_images = []
    padded_densities = []
    points_list = []
    
    for item in batch:
        img = item['image']
        den = item['density']
        pts = item['points']

        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_den = F.pad(den, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        padded_images.append(padded_img)
        padded_densities.append(padded_den)
        points_list.append(torch.from_numpy(pts)) # Converte i punti in tensori
        
    images_tensor = torch.stack(padded_images, 0)
    densities_tensor = torch.stack(padded_densities, 0)

    # Restituisce una tupla, come si aspetta il ciclo di training
    return images_tensor, densities_tensor, points_list

def setup_experiment(exp_dir):
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
        best_val = ckpt.get("best_val", float("inf"))
        print(f"✅ Ripreso l'addestramento da {last_ck} (epoch={start_epoch})")
        return start_epoch, best_val
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