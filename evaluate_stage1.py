# evaluate_stage1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import numpy as np

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn

# ============================================================
# LOSS DI VALIDAZIONE (BCE - STESSA DEL TRAINING)
# ============================================================
class PiHeadLoss(nn.Module):
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

@torch.no_grad()
def validate_checkpoint(model, criterion, dataloader, device, config, checkpoint_path):
    """
    Validazione Stage 1 (BCE Mode).
    Monitora principalmente la Loss BCE e l'Active Ratio della maschera.
    """
    model.eval()
    total_loss = 0.0
    mae, mse = 0.0, 0.0
    pi_active_stats = []

    print("\n===== VALUTAZIONE STAGE 1 (BCE) =====")
    print("Metrica Principale: BCE Loss & Active Ratio (Maschera)")
    print("------------------------------------------------------")

    progress_bar = tqdm(dataloader, desc="Validating ZIP Stage 1")

    for idx, (images, gt_density, _) in enumerate(progress_bar):
        images, gt_density = images.to(device), gt_density.to(device)

        preds = model(images)
        loss = criterion(preds, gt_density)
        total_loss += loss.item()

        # Statistiche Maschera
        pi_logits = preds["logit_pi_maps"]
        pi_probs = torch.sigmoid(pi_logits[:, 1:2]) # Probabilità "pieno"
        
        active_ratio = (pi_probs > 0.5).float().mean().item() * 100
        pi_active_stats.append(active_ratio)

        # Calcolo conteggio (solo indicativo, lambda non è allenato)
        lam_maps = preds["lambda_maps"]
        pred_count = torch.sum(pi_probs * lam_maps).item()
        gt_count = torch.sum(gt_density).item()

        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count) ** 2

        if idx % 20 == 0:
            print(f"[IMG {idx:03d}] Loss(BCE)={loss.item():.4f} "
                  f"| π>0.5={active_ratio:.1f}% "
                  f"| π_mean={pi_probs.mean().item():.3f}")

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'active': f"{active_ratio:.1f}%"
        })

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_active = np.mean(pi_active_stats)

    print("\n--- RISULTATI FINALI ---")
    print(f"Checkpoint:      {os.path.basename(checkpoint_path)}")
    print(f"Validation Loss: {avg_loss:.4f} (BCE)")
    print(f"Avg Active Mask: {avg_active:.2f}% (Target: 10-30% per folle sparse)")
    print(f"MAE (Proxy):     {avg_mae:.2f} (⚠️ Non affidabile in Stage 1)")
    print("-------------------------------------\n")


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato.")
        return
        
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    # Setup Modello
    dataset_name = config['DATASET']
    bin_config = config['BINS_CONFIG'][dataset_name]
    bins, bin_centers = bin_config['bins'], bin_config['bin_centers']

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
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        gate=config['MODEL']['GATE'],
        upsample_to_input=config['MODEL']['UPSAMPLE_TO_INPUT'],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Caricamento Checkpoint
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    # Cerca prima il best_model, poi last.pth
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(output_dir, "last.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"❌ Errore: Nessun checkpoint trovato in {output_dir}")
        return

    print(f"✅ Caricamento checkpoint da {checkpoint_path}...")
    raw_state = torch.load(checkpoint_path, map_location=device)
    state_dict = raw_state.get('model', raw_state)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Modello caricato.")

    # Setup Loss (BCE)
    pos_weight = config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0)
    criterion = PiHeadLoss(
        pos_weight=pos_weight,
        block_size=config['DATA']['ZIP_BLOCK_SIZE']
    ).to(device)

    # Dataset
    DatasetClass = get_dataset(config['DATASET'])
    data_cfg = config['DATA']
    val_tf = build_transforms(data_cfg, is_train=False)
    val_dataset = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg['VAL_SPLIT'],
        block_size=data_cfg['ZIP_BLOCK_SIZE'],
        transforms=val_tf,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['OPTIM_ZIP']['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    validate_checkpoint(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    main()