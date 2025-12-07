# evaluate_stage3.py
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from losses.p2r_region_loss import P2RLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn

# ============================================================
# LOSS DEFINITIONS (Mirroring Training)
# ============================================================
class PiHeadLoss(nn.Module):
    """Loss BCE per la validazione della maschera (Stage 1/3 logic)."""
    def __init__(self, pos_weight=5.0, block_size=16):
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
        return loss

# ============================================================
# VALUTAZIONE
# ============================================================
@torch.no_grad()
def evaluate_joint(model, dataloader, device, default_down, criterion_zip, criterion_p2r):
    model.eval()
    mae, mse = 0.0, 0.0
    total_loss_zip, total_loss_p2r = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0
    pi_active_stats = []

    print("\n===== VALUTAZIONE STAGE 3 (Joint) =====")
    print("Monitoraggio: MAE, RMSE, BCE Loss (Mask), P2R Loss (Density)")
    
    for idx, (images, gt_density, points) in enumerate(tqdm(dataloader, desc="[Validate Stage 3]")):
        images, gt_density = images.to(device), gt_density.to(device)
        
        # Gestione punti (lista di tensori)
        points_list = [p.to(device) for p in points]

        # Forward
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        # 1. Calcolo Loss ZIP (BCE)
        loss_zip = criterion_zip(outputs, gt_density)
        total_loss_zip += loss_zip.item()

        # 2. Calcolo Loss P2R
        # Canonicalize per allineare grid size
        _, _, h_in, w_in = images.shape
        pred_density_aligned, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_eval"
        )
        loss_p2r = criterion_p2r(pred_density_aligned, points_list, down=down_tuple)
        total_loss_p2r += loss_p2r.item()

        # 3. Metriche di Conteggio
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        # Somma densità / area pixel
        pred_count = torch.sum(pred_density_aligned, dim=(1, 2, 3)) / cell_area
        
        gt_count_tensor = torch.tensor([len(p) for p in points_list], device=device)
        
        mae += torch.abs(pred_count - gt_count_tensor).sum().item()
        mse += ((pred_count - gt_count_tensor) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count_tensor.sum().item()

        # Statistiche Maschera (Active Ratio)
        pi_logits = outputs["logit_pi_maps"]
        pi_probs = torch.sigmoid(pi_logits[:, 1:2])
        active_ratio = (pi_probs > 0.5).float().mean().item() * 100
        pi_active_stats.append(active_ratio)

        if idx == 0:
            print("\n[DEBUG PRIMO BATCH]")
            print(f"  Img Size: {h_in}x{w_in}")
            print(f"  BCE Loss: {loss_zip.item():.4f}")
            print(f"  P2R Loss: {loss_p2r.item():.4f}")
            print(f"  Pred: {pred_count[0].item():.2f} | GT: {gt_count_tensor[0].item():.2f}")
            print(f"  Mask Active: {active_ratio:.1f}%")

    # Medie
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    
    avg_mae = mae / n_samples
    avg_rmse = np.sqrt(mse / n_samples)
    avg_loss_zip = total_loss_zip / n_batches
    avg_loss_p2r = total_loss_p2r / n_batches
    avg_active = np.mean(pi_active_stats)
    bias = total_pred / total_gt if total_gt > 0 else 0

    print("\n" + "="*40)
    print(f"RISULTATI FINALI STAGE 3")
    print("="*40)
    print(f"MAE:            {avg_mae:.2f}")
    print(f"RMSE:           {avg_rmse:.2f}")
    print(f"Bias (Pred/GT): {bias:.3f}")
    print("-" * 40)
    print(f"Loss ZIP (BCE): {avg_loss_zip:.4f}")
    print(f"Loss P2R:       {avg_loss_p2r:.4f}")
    print(f"Avg Active Mask:{avg_active:.2f}%")
    print("="*40 + "\n")


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato.")
        return
        
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])
    print(f"✅ Avvio valutazione Stage 3 su {device}")

    # Configurazione Modello
    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]
    
    # In Stage 3 (Eval), upsample potrebbe essere False se così era nel training
    upsample_to_input = cfg["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    if upsample_to_input:
        print("ℹ️ Forzo UPSAMPLE_TO_INPUT=False per consistenza con P2R logic.")
        upsample_to_input = False

    zip_head_cfg = cfg.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=upsample_to_input,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Caricamento Checkpoint
    ckpt_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    ckpt_path = os.path.join(ckpt_dir, "stage3_best.pth")
    if not os.path.exists(ckpt_path):
        # Fallback a last.pth o stage2_best se stage3 non esiste
        ckpt_path = os.path.join(ckpt_dir, "stage3", "last.pth")
    
    if not os.path.exists(ckpt_path):
        print("❌ Nessun checkpoint Stage 3 trovato (nè best, nè last).")
        return

    print(f"✅ Caricamento checkpoint da {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    # Dataset & Dataloader
    data_cfg = cfg["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["OPTIM_JOINT"]["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_fn, # Uso collate_fn standard che gestisce i punti
    )

    # Setup Losses per Monitoraggio
    pos_weight = cfg.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0)
    criterion_zip = PiHeadLoss(
        pos_weight=pos_weight,
        block_size=data_cfg["ZIP_BLOCK_SIZE"]
    ).to(device)

    p2r_loss_cfg = cfg.get("P2R_LOSS", {})
    loss_kwargs = {
        "scale_weight": float(p2r_loss_cfg.get("SCALE_WEIGHT", 1.0)),
        "pos_weight": float(p2r_loss_cfg.get("POS_WEIGHT", 1.0)),
        "chunk_size": int(p2r_loss_cfg.get("CHUNK_SIZE", 128)),
        "min_radius": float(p2r_loss_cfg.get("MIN_RADIUS", 0.0)),
        "max_radius": float(p2r_loss_cfg.get("MAX_RADIUS", 10.0))
    }
    criterion_p2r = P2RLoss(**loss_kwargs).to(device)

    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    
    # Esegui Valutazione
    evaluate_joint(model, val_loader, device, default_down, criterion_zip, criterion_p2r)


if __name__ == "__main__":
    main()