# P2R_ZIP/train_stage4_recovery.py
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
from train_utils import (
    init_seeds, get_optimizer, get_scheduler,
    canonicalize_p2r_grid
)

# Riusiamo la PiHeadLoss definita localmente nello stage 3 per semplicità
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

# Funzione Collate (identica a Stage 3)
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

# Training Loop
def train_one_epoch(
    model, criterion_zip, criterion_p2r, dataloader, optimizer, scheduler,
    schedule_step_mode, device, default_down, epoch, zip_scale, count_l1_w
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Train Stage 4 [ZIP={zip_scale}, L1={count_l1_w}]",
    )

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) if p is not None else None for p in points]

        optimizer.zero_grad()
        outputs = model(images)
        
        # 1. Loss ZIP (Mantenimento maschera)
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        # 2. Loss P2R (Regressione pura)
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)
        
        # 3. Loss L1 Conteggio (Forzatura)
        # Calcoliamo il conteggio predetto e quello reale per aggiungere una penalty esplicita
        pred_count = pred_density.sum(dim=(1,2,3))
        gt_counts = torch.stack([torch.tensor(len(p), device=device, dtype=torch.float) for p in points])
        loss_l1 = F.l1_loss(pred_count, gt_counts)
        
        # Combinazione: P2R base + Spinta L1 forte + Mantenimento ZIP
        combined_loss = scaled_loss_zip + loss_p2r + (loss_l1 * count_l1_w)
        
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler and schedule_step_mode == "iteration":
            scheduler.step()

        total_loss += combined_loss.item()
        progress_bar.set_postfix({
            "Tot": f"{combined_loss.item():.2f}",
            "ZIP": f"{scaled_loss_zip.item():.2f}",
            "P2R": f"{loss_p2r.item():.2f}",
            "L1": f"{loss_l1.item():.2f}"
        })

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, default_down):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 4"):
        images = images.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        
        # Calcolo conteggio corretto tenendo conto del downsampling
        down_h, down_w = down_tuple
        # Nel modello P2R, la densità è per pixel, quindi la somma diretta va bene
        # Se invece è densità spaziale, bisogna moltiplicare per l'area.
        # Assumiamo che il modello emetta densità che sommata dia il conteggio.
        
        # Verifica veloce: se il modello P2R_ZIP usa l'upsample, la mappa è HxW
        # se no è H/8 x W/8. La somma dei valori pixel DEVE dare il conteggio.
        # Normalizzazione area:
        cell_area = down_h * down_w # Fattore di scala spaziale
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        gt_count = torch.tensor([len(p) for p in points], dtype=torch.float32, device=device)

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    return mae, rmse, total_pred, total_gt

def main():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        print("❌ Config non trovato.")
        return

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 4 (Recovery & Polishing) su {device}")

    # Config Specifiche Stage 4
    optim_cfg = config["OPTIM_STAGE4"]
    loss_cfg = config["STAGE4_LOSS"]
    
    # Dataset & Dataloader
    data_cfg = config["DATA"]
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

    # Carica Pesi Stage 3 (Il "Cleaning")
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    stage3_path = os.path.join(output_dir, "stage3_best.pth")
    
    if os.path.isfile(stage3_path):
        print(f"🔄 Caricamento pesi Stage 3 da: {stage3_path}")
        state_dict = torch.load(stage3_path, map_location=device)
        if "model" in state_dict: state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("❌ ERRORE CRITICO: stage3_best.pth non trovato. Completa lo Stage 3 prima!")
        return

    # Optimizer Config
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': optim_cfg["LR_BACKBONE"]},
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
         'lr': optim_cfg["LR_HEADS"]}
    ]
    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # Loss Setup
    criterion_zip = PiHeadLoss(pos_weight=5.0, block_size=data_cfg["ZIP_BLOCK_SIZE"]).to(device)
    
    p2r_loss_cfg = config["P2R_LOSS"]
    criterion_p2r = P2RLoss(
        scale_weight=p2r_loss_cfg["SCALE_WEIGHT"],
        pos_weight=p2r_loss_cfg["POS_WEIGHT"],
        chunk_size=p2r_loss_cfg["CHUNK_SIZE"],
        min_radius=p2r_loss_cfg["MIN_RADIUS"],
        max_radius=p2r_loss_cfg["MAX_RADIUS"]
    ).to(device)

    # Parametri Training
    best_mae = float("inf")
    patience = optim_cfg["EARLY_STOPPING_PATIENCE"]
    no_improve = 0
    default_down = data_cfg["P2R_DOWNSAMPLE"]
    
    count_l1_w = float(loss_cfg["COUNT_L1_W"])
    zip_scale = float(loss_cfg["ZIP_SCALE"])

    print(f"\n🚀 START STAGE 4: Recovery (L1 weight: {count_l1_w}, ZIP scale: {zip_scale})")
    
    for epoch in range(optim_cfg["EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{optim_cfg['EPOCHS']} ---")
        
        train_loss = train_one_epoch(
            model, criterion_zip, criterion_p2r, train_loader, optimizer,
            scheduler, "epoch", device, default_down, epoch+1, 
            zip_scale, count_l1_w
        )
        
        if scheduler: scheduler.step()

        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            val_mae, val_rmse, tot_pred, tot_gt = validate(model, val_loader, device, default_down)
            bias = tot_pred / tot_gt if tot_gt > 0 else 0
            
            print(f"Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | Bias: {bias:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_path = os.path.join(output_dir, "stage4_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"🏆 NEW BEST STAGE 4: {best_path}")
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print("⛔ Early Stopping triggered.")
                break
    
    print(f"\n✅ TRAINING FINITO. Best MAE Stage 4: {best_mae:.2f}")

if __name__ == "__main__":
    main()