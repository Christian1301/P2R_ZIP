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
# from losses.composite_loss import ZIPCompositeLoss  <-- RIMOSSO
from losses.p2r_region_loss import P2RLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    save_checkpoint, canonicalize_p2r_grid,
    resume_if_exists # Assicuriamoci che questa sia importata se serve, o gestita manualmente
)
from train_stage2_p2r import calibrate_density_scale

# ============================================================
# LOSS CONDIVISA (Copiata da Stage 1 per coerenza)
# ============================================================
class PiHeadLoss(nn.Module):
    """
    Loss per la parte ZIP in Stage 3: mantiene la classificazione binaria.
    """
    def __init__(
        self,
        pos_weight: float = 5.0,
        block_size: int = 16,
    ):
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
# UTILS
# ============================================================

def adaptive_loss_weights(epoch, max_epochs, strategy="progressive"):
    """Bilanciamento adattivo tra le due loss (BCE e P2R)."""
    progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)

    if strategy == "progressive":
        # Inizia favorendo la maschera, poi sposta l'attenzione sulla regressione
        if epoch < 200:
            zip_scale = 1.0
            alpha = 0.8
        elif epoch < 600:
            phase_progress = (epoch - 200) / 400
            zip_scale = 1.0 - 0.3 * phase_progress
            alpha = 0.8 + 0.6 * phase_progress
        else:
            denom = max(max_epochs - 600, 1)
            phase_progress = min((epoch - 600) / denom, 1.0)
            zip_scale = 0.7 - 0.1 * phase_progress
            alpha = 1.4 + 0.3 * phase_progress
    elif strategy == "cosine":
        zip_scale = 0.9 - 0.3 * (1 - np.cos(np.pi * progress)) / 2
        alpha = 0.8 + 0.9 * (1 - np.cos(np.pi * progress)) / 2
    else:
        # Default statico
        zip_scale, alpha = 1.0, 1.0

    return zip_scale, alpha

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
    model,
    criterion_zip, # Ora è PiHeadLoss
    criterion_p2r,
    dataloader,
    optimizer,
    scheduler,
    schedule_step_mode,
    device,
    default_down,
    clamp_cfg=None,
    epoch: int = 1,
    max_epochs: int = 1200,
    adaptive_cfg=None,
):
    model.train()
    total_loss = 0.0
    base_zip_scale = 1.0
    base_alpha = 1.0
    adaptive_enabled = False
    adaptive_strategy = "progressive"

    if adaptive_cfg:
        base_zip_scale = adaptive_cfg.get("base_zip_scale", base_zip_scale)
        base_alpha = adaptive_cfg.get("base_alpha", base_alpha)
        adaptive_enabled = adaptive_cfg.get("enabled", False)
        adaptive_strategy = adaptive_cfg.get("strategy", adaptive_strategy)

    if adaptive_enabled:
        zip_scale, alpha = adaptive_loss_weights(epoch, max_epochs, strategy=adaptive_strategy)
    else:
        zip_scale, alpha = base_zip_scale, base_alpha

    # Modifica: Normalizziamo un po' i pesi. 
    # La BCE è spesso ~0.5-1.0, la P2R loss può essere più alta.
    # Assicuriamoci che zip_scale non sia troppo alto rispetto ad alpha.
    
    if epoch % 50 == 1:
        print(f"\n📊 Epoch {epoch}: BCE Scale={zip_scale:.3f}, P2R Alpha={alpha:.3f}")

    progress_bar = tqdm(
        dataloader,
        desc=f"Train Stage 3 [BCE={zip_scale:.2f}, P2R={alpha:.2f}]",
    )

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points_device = []
        for p in points:
            if p is None:
                points_device.append(p)
            else:
                points_device.append(p.to(device))
        points = points_device

        optimizer.zero_grad()
        outputs = model(images)
        
        # 1. Loss ZIP (BCE)
        # Nota: PiHeadLoss ignora gt_density per il conteggio, usa solo la maschera derivata
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        # 2. Loss P2R (Regression)
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_train"
        )

        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)
        
        # 3. Loss Totale
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
        current_lr = max(group['lr'] for group in optimizer.param_groups)
        progress_bar.set_postfix({
            "Tot": f"{combined_loss.item():.3f}",
            "BCE": f"{scaled_loss_zip.item():.3f}",
            "P2R": f"{loss_p2r.item():.3f}",
            "lr": f"{current_lr:.6f}"
        })

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 3"):
        images, gt_density = images.to(device), gt_density.to(device)
       
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_val"
        )
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        cell_area_tensor = pred_density.new_tensor(cell_area)

        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area_tensor
        
        gt_count_list = [len(p) for p in points if p is not None]
        if not gt_count_list:
             gt_count_list = [0] 
        gt_count = torch.tensor(gt_count_list, dtype=torch.float32, device=device)

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse, total_pred, total_gt

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

    default_down = config["DATA"].get("P2R_DOWNSAMPLE", 8)
    loss_cfg = config.get("P2R_LOSS", {})
    clamp_cfg = loss_cfg.get("LOG_SCALE_CLAMP")
    max_adjust = loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA")

    dataset_name = config["DATASET"]
    bin_config = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_config["bins"], bin_config["bin_centers"]

    upsample_to_input = config["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    
    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Caricamento Checkpoint
    optim_cfg = config["OPTIM_JOINT"]
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    stage2_checkpoint_path = os.path.join(output_dir, "stage2_best.pth")
    stage3_checkpoint_path = os.path.join(output_dir, "stage3_best.pth")
    
    load_stage3_best = bool(optim_cfg.get("LOAD_STAGE3_BEST", False))

    if load_stage3_best and os.path.isfile(stage3_checkpoint_path):
        state_dict = torch.load(stage3_checkpoint_path, map_location=device)
        if "model" in state_dict: state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Ripreso Stage 3 dal best precedente: {stage3_checkpoint_path}")
    else:
        if os.path.isfile(stage2_checkpoint_path):
            state_dict = torch.load(stage2_checkpoint_path, map_location=device)
            if "model" in state_dict: state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Inizio Stage 3 con pesi Stage 2: {stage2_checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint Stage 2 non trovato! Partenza da pesi casuali (SCONSIGLIATO).")

    # Sblocca tutto per il Joint Training
    for p in model.parameters():
        p.requires_grad = True

    # 1. Definizione Loss ZIP (BCE Mode)
    # Importante: usiamo lo stesso pos_weight dello Stage 1
    pos_weight = config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0)
    print(f"🔧 ZIP Loss Config: PiHeadLoss (BCE) con pos_weight={pos_weight}")
    
    criterion_zip = PiHeadLoss(
        pos_weight=pos_weight,
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"]
    ).to(device)

    # 2. Definizione Loss P2R
    loss_kwargs = {
        "scale_weight": float(loss_cfg.get("SCALE_WEIGHT", 1.0)),
        "pos_weight": float(loss_cfg.get("POS_WEIGHT", 1.0)),
        "chunk_size": int(loss_cfg.get("CHUNK_SIZE", 128)),
        "min_radius": float(loss_cfg.get("MIN_RADIUS", 0.0)),
        "max_radius": float(loss_cfg.get("MAX_RADIUS", 10.0))
    }
    criterion_p2r = P2RLoss(**loss_kwargs).to(device)

    # Parametri Adattivi
    base_alpha = float(config["JOINT_LOSS"].get("ALPHA", 1.0))
    base_zip_scale = float(config["JOINT_LOSS"].get("ZIP_SCALE", 1.0))
    adaptive_cfg = {
        "enabled": bool(config["JOINT_LOSS"].get("ADAPTIVE_BALANCING", False)),
        "strategy": config["JOINT_LOSS"].get("BALANCING_STRATEGY", "progressive"),
        "base_alpha": base_alpha,
        "base_zip_scale": base_zip_scale,
    }

    # Optimizer (LR differenziato)
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': optim_cfg["LR_BACKBONE"]},
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
        'lr': optim_cfg["LR_HEADS"]}
    ]

    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])
    schedule_step_mode = str(optim_cfg.get("SCHEDULER_STEP", "iteration")).lower()

    # Dataloaders
    data_cfg = config["DATA"]
    train_transforms = build_transforms(data_cfg, is_train=True)
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_transforms,
    )
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True,
        collate_fn=collate_joint,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True,
        collate_fn=collate_joint,
    )
    
    # Calibrazione iniziale (opzionale ma consigliata)
    if hasattr(model, "p2r_head") and hasattr(model.p2r_head, "log_scale"):
        print("🔧 Calibrazione rapida scala P2R...")
        calibrate_density_scale(
            model, val_loader, device, default_down,
            max_batches=10, clamp_range=clamp_cfg, max_adjust=max_adjust,
        )

    # Training Loop
    best_mae = float("inf")
    best_epoch = 0
    epochs_stage3 = optim_cfg["EPOCHS"]
    patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 0)))
    no_improve_rounds = 0

    print(f"🚀 Inizio Stage 3 — Fine-tuning (BCE+P2R) per {epochs_stage3} epoche...")

    for epoch in range(epochs_stage3):
        print(f"\n--- Epoch {epoch + 1}/{epochs_stage3} ---")
        
        train_loss = train_one_epoch(
            model, criterion_zip, criterion_p2r,
            train_loader, optimizer, scheduler, schedule_step_mode,
            device, default_down, clamp_cfg=clamp_cfg,
            epoch=epoch + 1, max_epochs=epochs_stage3, adaptive_cfg=adaptive_cfg,
        )

        if scheduler and schedule_step_mode == "epoch":
            scheduler.step()

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, default_down)
            bias = (tot_pred / tot_gt) if tot_gt > 0 else float("nan")
            
            print(f"Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | Bias: {bias:.3f}")

            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae
                best_epoch = epoch + 1
                best_path = os.path.join(output_dir, "stage3_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"✅ Saved New Best Stage 3: {best_path}")
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if patience > 0 and no_improve_rounds >= patience:
                print(f"⛔ Early stopping attiva.")
                break

    print(f"✅ Stage 3 completato! Best MAE: {best_mae:.2f}")

if __name__ == "__main__":
    main()