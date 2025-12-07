# P2R_ZIP/train_stage3_joint.py
# -*- coding: utf-8 -*-
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from losses.p2r_region_loss import P2RLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, get_optimizer, get_scheduler, canonicalize_p2r_grid
from train_stage2_p2r import calibrate_density_scale

# ============================================================
# LOSS CONDIVISA
# ============================================================
class PiHeadLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, block_size: int = 16):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        return (gt_counts_per_block > 1e-3).float()
    
    def forward(self, predictions, gt_density):
        logit_pi_maps = predictions["logit_pi_maps"]
        logit_pieno = logit_pi_maps[:, 1:2, :, :] 
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, size=logit_pieno.shape[-2:], mode='nearest'
            )
        
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
            
        loss = self.bce(logit_pieno, gt_occupancy)
        return loss, {}

# ============================================================
# UTILS & COLLATE
# ============================================================

def _round_up_8(x: int) -> int:
    return (x + 7) // 8 * 8

def collate_joint(batch):
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        dens = [b["density"] for b in batch]
        pts  = [b.get("points", None) for b in batch]
    else:
        imgs, dens, pts = zip(*[(s[0], s[1], s[2]) for s in batch])

    H_max = max(im.shape[-2] for im in imgs)
    W_max = max(im.shape[-1] for im in imgs)
    H_tgt, W_tgt = _round_up_8(H_max), _round_up_8(W_max)

    imgs_out, dens_out, pts_out = [], [], []
    for im, den, p in zip(imgs, dens, pts):
        _, H, W = im.shape
        sy, sx = H_tgt / H, W_tgt / W

        im_res = F.interpolate(im.unsqueeze(0), size=(H_tgt, W_tgt),
                               mode='bilinear', align_corners=False).squeeze(0)
        den_res = F.interpolate(den.unsqueeze(0), size=(H_tgt, W_tgt),
                                mode='bilinear', align_corners=False).squeeze(0)
        den_res *= (H * W) / (H_tgt * W_tgt)

        if p is None or (hasattr(p, "numel") and p.numel() == 0):
            p_scaled = p
        else:
            p_scaled = p.clone()
            p_scaled[:, 0] *= sx
            p_scaled[:, 1] *= sy

        imgs_out.append(im_res)
        dens_out.append(den_res)
        pts_out.append(p_scaled)

    return torch.stack(imgs_out), torch.stack(dens_out), pts_out

# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(
    model, criterion_zip, criterion_p2r, dataloader, optimizer, scheduler,
    schedule_step_mode, device, default_down, clamp_cfg, epoch,
    zip_scale, alpha
):
    model.train()
    total_loss = 0.0
    
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(
        dataloader,
        desc=f"Train Stage 3 [Ep {epoch}] [LR={current_lr:.2e}]",
    )

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) if p is not None else None for p in points]

        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss ZIP
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        # Loss P2R
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)
        
        # Loss Totale
        combined_loss = scaled_loss_zip + alpha * loss_p2r
        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler and schedule_step_mode == "iteration":
            scheduler.step()

        if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
            min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
            model.p2r_head.log_scale.data.clamp_(min_val, max_val)

        total_loss += combined_loss.item()
        
        progress_bar.set_postfix({
            "Loss": f"{combined_loss.item():.2f}",
            "BCE": f"{scaled_loss_zip.item():.2f}",
            "P2R": f"{loss_p2r.item():.2f}"
        })

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 3"):
        images = images.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_val"
        )
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        gt_count = torch.tensor([len(p) for p in points if p is not None], dtype=torch.float32, device=device)

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse, total_pred, total_gt

# ============================================================
# FUNZIONI DI SALVATAGGIO / CARICAMENTO FULL STATE
# ============================================================

def save_full_checkpoint(path, model, optimizer, scheduler, epoch, best_mae):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_mae': best_mae
    }
    torch.save(state, path)

def load_full_checkpoint(path, model, optimizer, scheduler, device):
    if not os.path.isfile(path):
        return 0, float('inf') # Start from scratch
    
    print(f"🔄 Trovato checkpoint completo: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # 1. Carica modello
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Fallback se è un vecchio formato (solo pesi)
        model.load_state_dict(checkpoint, strict=False)
        print("⚠️ Checkpoint vecchio formato (solo pesi). Optimizer reset.")
        return 0, float('inf')

    # 2. Carica Optimizer & Scheduler (Fondamentale per riprendere LR)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0)
    best_mae = checkpoint.get('best_mae', float('inf'))
    
    print(f"✅ Ripristinato stato training: Epoca {start_epoch}, Best MAE {best_mae:.2f}")
    return start_epoch, best_mae

# ============================================================
# MAIN
# ============================================================

def main():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        print("❌ Config non trovato.")
        return

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 3 (Joint ZIP[BCE] + P2R) su {device}")

    # Configs
    optim_cfg = config["OPTIM_JOINT"]
    data_cfg = config["DATA"]
    loss_cfg = config["P2R_LOSS"]
    
    # Dataset
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"], split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"], transforms=train_transforms
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"], split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"], transforms=val_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True,
        collate_fn=collate_joint
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True,
        collate_fn=collate_joint
    )

    # Modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"]["LAMBDA_SCALE"],
        "lambda_max": config["ZIP_HEAD"]["LAMBDA_MAX"],
        "use_softplus": config["ZIP_HEAD"]["USE_SOFTPLUS"],
        "lambda_noise_std": config["ZIP_HEAD"]["LAMBDA_NOISE_STD"],
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=config["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_head_kwargs=zip_head_kwargs
    ).to(device)

    # Optimizer
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': optim_cfg["LR_BACKBONE"]},
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
         'lr': optim_cfg["LR_HEADS"]}
    ]
    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # Loss
    criterion_zip = PiHeadLoss(
        pos_weight=config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0),
        block_size=data_cfg["ZIP_BLOCK_SIZE"]
    ).to(device)

    criterion_p2r = P2RLoss(
        scale_weight=float(loss_cfg.get("SCALE_WEIGHT", 1.0)),
        pos_weight=float(loss_cfg.get("POS_WEIGHT", 1.0)),
        chunk_size=int(loss_cfg.get("CHUNK_SIZE", 128)),
        min_radius=float(loss_cfg.get("MIN_RADIUS", 0.0)),
        max_radius=float(loss_cfg.get("MAX_RADIUS", 10.0))
    ).to(device)

    # Gestione Resume / Load
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, "stage3_latest.pth") # Per resume
    best_path = os.path.join(output_dir, "stage3_best.pth")     # Per best weights
    stage2_path = os.path.join(output_dir, "stage2_best.pth")   # Fallback

    start_epoch = 0
    best_mae = float("inf")

    # Logica di caricamento intelligente
    if os.path.isfile(latest_path):
        # 1. Priorità: Resume da un crash/interruzione
        start_epoch, best_mae = load_full_checkpoint(latest_path, model, optimizer, scheduler, device)
    elif not optim_cfg.get("LOAD_STAGE3_BEST", False) and os.path.isfile(stage2_path):
        # 2. Partenza da zero usando pesi Stage 2
        print(f"✅ Inizio Stage 3 con pesi Stage 2: {stage2_path}")
        state = torch.load(stage2_path, map_location=device)
        if "model" in state: state = state["model"]
        model.load_state_dict(state, strict=False)
        
        # Calibrazione P2R solo se partiamo da zero
        if hasattr(model.p2r_head, "log_scale"):
            print("🔧 Calibrazione rapida scala P2R...")
            calibrate_density_scale(
                model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"],
                max_batches=40, clamp_range=loss_cfg.get("LOG_SCALE_CLAMP"),
                max_adjust=loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")
            )
    
    # Parametri Training
    epochs_stage3 = optim_cfg["EPOCHS"]
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 50)
    no_improve = 0
    
    zip_scale = float(config["JOINT_LOSS"].get("ZIP_SCALE", 1.0))
    alpha = float(config["JOINT_LOSS"].get("ALPHA", 1.0))

    print(f"\n🚀 START Stage 3 (Ep {start_epoch+1} -> {epochs_stage3})")
    
    for epoch in range(start_epoch, epochs_stage3):
        print(f"\n--- Epoch {epoch + 1}/{epochs_stage3} ---")
        
        train_loss = train_one_epoch(
            model, criterion_zip, criterion_p2r, train_loader, optimizer,
            scheduler, str(optim_cfg.get("SCHEDULER_STEP", "iteration")).lower(),
            device, data_cfg["P2R_DOWNSAMPLE"], loss_cfg.get("LOG_SCALE_CLAMP"),
            epoch + 1, zip_scale, alpha
        )

        if scheduler and str(optim_cfg.get("SCHEDULER_STEP", "iteration")).lower() == "epoch":
            scheduler.step()

        # Salvataggio "Latest" ad ogni epoca per poter riprendere
        save_full_checkpoint(latest_path, model, optimizer, scheduler, epoch + 1, best_mae)

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"])
            bias = tot_pred / tot_gt if tot_gt > 0 else 0
            
            print(f"Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | Bias: {bias:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                # Salviamo il best (anche in formato full, per sicurezza)
                save_full_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_mae)
                print(f"✅ Saved New Best Stage 3: {best_path}")
                no_improve = 0
            else:
                no_improve += 1
            
            if patience > 0 and no_improve >= patience:
                print(f"⛔ Early stopping attiva.")
                break

    print(f"✅ Stage 3 completato! Best MAE: {best_mae:.2f}")

if __name__ == "__main__":
    main()