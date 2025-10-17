# P2R_ZIP/train_utils.py
import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR 
import torch.nn.functional as F

def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(param_groups, optim_config):
    lr = optim_config['LR']
    wd = optim_config['WEIGHT_DECAY']
    optimizer_type = optim_config.get('TYPE', 'adamw').lower()

    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("get_optimizer si aspetta una lista non vuota di gruppi di parametri")

    if optimizer_type == 'adamw':
        return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif optimizer_type == 'adam':
        return optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' non supportato.")

def get_scheduler(optimizer, optim_config, max_epochs):
    scheduler_type = optim_config.get('SCHEDULER', 'cosine').lower()
    warmup_epochs = optim_config.get('WARMUP_EPOCHS', 0) # Utile per cosine

    # --- MODIFICA: Aggiungi gestione 'multistep' ---
    if scheduler_type == 'multistep':
        if warmup_epochs > 0:
             print("Attenzione: Warmup non è tipicamente usato con MultiStepLR nel codice ZIP.")
        # Legge i milestone e il gamma dalla configurazione
        milestones = optim_config.get('SCHEDULER_STEPS', [max_epochs]) # Default a nessun decay se non specificato
        gamma = optim_config.get('SCHEDULER_GAMMA', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # --- FINE MODIFICA ---

    elif scheduler_type == 'cosine':
        # Gestione warmup per cosine (come prima)
        if warmup_epochs > 0:
            # Funzione lambda per warmup + cosine decay
            def warmup_cosine_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # Warmup lineare da ~0 a 1
                    return float(current_epoch + 1) / float(max(1, warmup_epochs))
                else:
                    # Cosine decay dopo il warmup
                    progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            return LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        else:
            # Solo Cosine decay
            return CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs) # T_max è il numero di epoche *dopo* il warmup

    # Se nessun scheduler valido o 'none'
    return None

def collate_fn(batch):
    """
    Collate function per gestire un batch di dizionari da BaseCrowdDataset
    o dal vecchio CrowdDataset. Gestisce padding per immagini/densità.
    """
    item = batch[0]

    # --- MODIFICA QUI ---
    # Caso 1: Basato su BaseCrowdDataset (output: {'image', 'points', 'density', 'img_path'})
    # Controlla se 'points' è un Tensor, come restituito da __getitem__ dopo le transforms
    if 'density' in item and isinstance(item['points'], torch.Tensor):
    # --- FINE MODIFICA ---
        max_h = max(item['image'].shape[1] for item in batch)
        max_w = max(item['image'].shape[2] for item in batch)

        padded_images = []
        padded_densities = []
        points_list = [] # Mantiene i punti come lista di tensori

        for item in batch:
            img, den, pts_tensor = item['image'], item['density'], item['points']
            pad_h, pad_w = max_h - img.shape[1], max_w - img.shape[2]

            # Padding a destra e in basso
            padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_den = F.pad(den, (0, pad_w, 0, pad_h), mode='constant', value=0)

            padded_images.append(padded_img)
            padded_densities.append(padded_den)
            points_list.append(pts_tensor) # Aggiunge il tensore dei punti alla lista

        # Ritorna: Tensor[B, C, Hmax, Wmax], Tensor[B, 1, Hmax, Wmax], List[Tensor[Ni, 2]]
        return torch.stack(padded_images, 0), torch.stack(padded_densities, 0), points_list

    # Caso 2: Basato sul vecchio CrowdDataset (da data/adapters.py)
    # { 'image': Tensor, 'points': Tensor, 'zip_blocks': Tensor, ... }
    elif 'zip_blocks' in item and isinstance(item['points'], torch.Tensor):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        blocks = torch.stack([b["zip_blocks"] for b in batch], dim=0)
        points = [b["points"] for b in batch] # Già una lista di tensori
        # Restituisce (images, zip_blocks, points_list) - Nota: qui il secondo elemento è diverso
        # Questo caso potrebbe non essere più necessario se usi solo BaseCrowdDataset
        # Ma lo manteniamo per compatibilità (anche se l'output è leggermente diverso)
        print("Attenzione: collate_fn sta usando il formato 'zip_blocks'. Assicurati che sia intenzionale.")
        return imgs, blocks, points

    # Se nessuno dei formati noti viene riconosciuto
    raise TypeError(f"Formato batch non riconosciuto in collate_fn: {item.keys()}")


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