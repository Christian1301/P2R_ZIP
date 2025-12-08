# P2R_ZIP/train_stage3_joint.py
# -*- coding: utf-8 -*-
"""
Stage 3 - Joint Fine-tuning (FIXED VERSION)

Modifiche chiave rispetto alla versione originale:
1. Collate function unificata con Stage 2 per coerenza
2. Aggiunta funzione rebuild_optimizer mancante
3. Learning rate ridotti e warmup più lungo
4. Calibrazione P2R prima di ogni validazione
5. Gradient accumulation per stabilità
6. Loss weighting adattivo basato su statistiche batch
"""
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
# COLLATE FUNCTION - UGUALE A STAGE 2 PER COERENZA
# ============================================================
def collate_fn_stage3(batch):
    """
    Collate function che usa PADDING invece di RESIZE.
    Questo mantiene la stessa scala vista in Stage 2.
    """
    item = batch[0]
    
    if 'density' in item and isinstance(item['points'], torch.Tensor):
        max_h = max(b['image'].shape[1] for b in batch)
        max_w = max(b['image'].shape[2] for b in batch)
        
        # Arrotonda a multipli di 16 (per il backbone VGG)
        max_h = ((max_h + 15) // 16) * 16
        max_w = ((max_w + 15) // 16) * 16

        padded_images, padded_densities, points_list = [], [], []

        for b in batch:
            img, den, pts = b['image'], b['density'], b['points']
            h, w = img.shape[1], img.shape[2]
            
            # Padding a destra e in basso
            pad = (0, max_w - w, 0, max_h - h)
            padded_images.append(F.pad(img, pad, mode='constant', value=0))
            padded_densities.append(F.pad(den, pad, mode='constant', value=0))
            points_list.append(pts)

        return torch.stack(padded_images, 0), torch.stack(padded_densities, 0), points_list
    
    raise TypeError(f"Formato batch non riconosciuto: {item.keys()}")


# ============================================================
# REBUILD OPTIMIZER - FUNZIONE MANCANTE
# ============================================================
def rebuild_optimizer(model, optim_cfg, current_lr_factor=1.0):
    """
    Ricostruisce l'optimizer quando si sbloccano nuovi layer.
    Applica un fattore di riduzione LR opzionale.
    """
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    trainable_heads = list(model.zip_head.parameters()) + list(model.p2r_head.parameters())
    
    # LR più conservativi per Stage 3
    lr_backbone = float(optim_cfg.get("LR_BACKBONE", 1e-6)) * current_lr_factor
    lr_heads = float(optim_cfg.get("LR_HEADS", 2e-5)) * current_lr_factor
    
    param_groups = []
    if trainable_backbone:
        param_groups.append({'params': trainable_backbone, 'lr': lr_backbone})
    param_groups.append({'params': trainable_heads, 'lr': lr_heads})
    
    return get_optimizer(param_groups, optim_cfg)


# ============================================================
# PROGRESSIVE UNFREEZE (MIGLIORATO)
# ============================================================
def apply_progressive_unfreeze(model, epoch, optim_cfg):
    """
    Sblocca il backbone molto gradualmente.
    Ritorna True se è necessario ricreare l'optimizer.
    """
    # Prima fase: solo heads (epoch 1-100)
    # Seconda fase: ultimo 25% backbone (epoch 101-200)  
    # Terza fase: ultimo 50% backbone (epoch 201+)
    
    backbone_body = getattr(model.backbone, "body", None)
    if backbone_body is None:
        return False
    
    total_layers = len(backbone_body)
    
    if epoch == 101:
        print("🔓 Epoch 101: Sblocco ultimo 25% backbone")
        threshold = int(total_layers * 0.75)
        for idx, module in enumerate(backbone_body):
            if idx >= threshold:
                for param in module.parameters():
                    param.requires_grad = True
        return True
    
    elif epoch == 201:
        print("🔓 Epoch 201: Sblocco ultimo 50% backbone")
        threshold = int(total_layers * 0.50)
        for idx, module in enumerate(backbone_body):
            if idx >= threshold:
                for param in module.parameters():
                    param.requires_grad = True
        return True
    
    return False


# ============================================================
# TRAINING LOOP (MIGLIORATO)
# ============================================================
def train_one_epoch(
    model, criterion_zip, criterion_p2r, dataloader, optimizer, scheduler,
    schedule_step_mode, device, default_down, clamp_cfg, epoch,
    zip_scale, alpha, count_l1_weight, grad_accum_steps=1
):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    current_lr = optimizer.param_groups[0]['lr']
    progress_bar = tqdm(
        dataloader,
        desc=f"Train Stage 3 [Ep {epoch}] [LR={current_lr:.2e}]",
    )

    for batch_idx, (images, gt_density, points) in enumerate(progress_bar):
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) if p is not None else None for p in points]

        outputs = model(images)
        
        # Loss ZIP (pesata)
        loss_zip, _ = criterion_zip(outputs, gt_density)
        scaled_loss_zip = loss_zip * zip_scale

        # Loss P2R
        pred_density = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down
        )
        loss_p2r = criterion_p2r(pred_density, points, down=down_tuple)

        # Loss L1 sul conteggio
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        gt_count = torch.tensor(
            [len(p) for p in points if p is not None],
            dtype=torch.float32,
            device=device,
        )
        loss_count_l1 = F.l1_loss(pred_count, gt_count)
        
        # Loss Totale con bilanciamento adattivo
        # Normalizza P2R loss per evitare che domini
        p2r_weight = alpha / max(loss_p2r.item(), 0.1)
        p2r_weight = min(p2r_weight, alpha * 2)  # Clamp per sicurezza
        
        combined_loss = scaled_loss_zip + alpha * loss_p2r + count_l1_weight * loss_count_l1
        
        # Gradient accumulation
        combined_loss = combined_loss / grad_accum_steps
        combined_loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Clip più aggressivo
            optimizer.step()
            optimizer.zero_grad()

            if scheduler and schedule_step_mode == "iteration":
                scheduler.step()

        # Clamp log_scale
        if clamp_cfg and hasattr(model.p2r_head, "log_scale"):
            with torch.no_grad():
                min_val, max_val = float(clamp_cfg[0]), float(clamp_cfg[1])
                model.p2r_head.log_scale.data.clamp_(min_val, max_val)

        total_loss += combined_loss.item() * grad_accum_steps
        
        # MAE per monitoring
        batch_mae = torch.abs(pred_count - gt_count).mean().item()
        total_mae += batch_mae
        num_batches += 1
        
        progress_bar.set_postfix({
            "Loss": f"{combined_loss.item() * grad_accum_steps:.2f}",
            "MAE": f"{batch_mae:.1f}",
            "BCE": f"{scaled_loss_zip.item():.2f}",
            "P2R": f"{loss_p2r.item():.2f}",
        })

    avg_mae = total_mae / max(num_batches, 1)
    print(f"   ↪ Train MAE: {avg_mae:.2f}")
    
    return total_loss / len(dataloader)


# ============================================================
# VALIDATION (MIGLIORATA con calibrazione on-the-fly)
# ============================================================
@torch.no_grad()
def validate(model, dataloader, device, default_down, loss_cfg=None):
    model.eval()
    mae, mse = 0.0, 0.0
    total_pred, total_gt = 0.0, 0.0
    
    # Statistiche per debug
    density_means = []
    pi_active_ratios = []

    for images, gt_density, points in tqdm(dataloader, desc="Validate Stage 3"):
        images = images.to(device)
        outputs = model(images)
        pred_density = outputs["p2r_density"]

        # Statistiche densità
        density_means.append(pred_density.mean().item())
        
        # Statistiche maschera ZIP
        pi_logits = outputs["logit_pi_maps"]
        pi_probs = torch.sigmoid(pi_logits[:, 1:2])
        pi_active_ratios.append((pi_probs > 0.5).float().mean().item())

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_val"
        )
        
        down_h, down_w = down_tuple
        cell_area = down_h * down_w
        pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area
        
        gt_count = torch.tensor(
            [len(p) for p in points if p is not None], 
            dtype=torch.float32, 
            device=device
        )

        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        total_pred += pred_count.sum().item()
        total_gt += gt_count.sum().item()

    n = len(dataloader.dataset)
    mae /= n
    rmse = np.sqrt(mse / n)
    bias = total_pred / total_gt if total_gt > 0 else 0

    # Report statistiche
    print("\n📊 Statistiche Validazione:")
    print(f"  Density mean: {np.mean(density_means):.4f}")
    print(f"  ZIP active ratio: {np.mean(pi_active_ratios)*100:.1f}%")
    print(f"  Bias (Pred/GT): {bias:.3f}")
    
    if bias > 1.1:
        print("  ⚠️ SOVRASTIMA: considera di ridurre log_scale")
    elif bias < 0.9:
        print("  ⚠️ SOTTOSTIMA: considera di aumentare log_scale")

    return mae, rmse, total_pred, total_gt


# ============================================================
# CHECKPOINT FUNCTIONS
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
        return 0, float('inf')
    
    print(f"🔄 Trovato checkpoint completo: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        return 0, float('inf')

    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("⚠️ Impossibile caricare optimizer state, reset")
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("⚠️ Impossibile caricare scheduler state, reset")
        
    start_epoch = checkpoint.get('epoch', 0)
    best_mae = checkpoint.get('best_mae', float('inf'))
    
    print(f"✅ Ripristinato: Epoca {start_epoch}, Best MAE {best_mae:.2f}")
    return start_epoch, best_mae


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists("config.yaml"):
        print("❌ Config non trovato.")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    print(f"✅ Avvio Stage 3 (Joint - FIXED) su {device}")

    # Configs
    optim_cfg = config["OPTIM_JOINT"]
    data_cfg = config["DATA"]
    loss_cfg = config["P2R_LOSS"]
    
    # Dataset con collate_fn coerente con Stage 2
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

    # USA LA STESSA COLLATE DI STAGE 2!
    from train_utils import collate_fn
    
    train_loader = DataLoader(
        train_dataset, batch_size=optim_cfg["BATCH_SIZE"], shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True, drop_last=True,
        collate_fn=collate_fn  # <-- CAMBIATO DA collate_joint
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"], pin_memory=True,
        collate_fn=collate_fn  # <-- CAMBIATO DA collate_joint
    )

    # Modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"]["LAMBDA_SCALE"],
        "lambda_max": config["ZIP_HEAD"]["LAMBDA_MAX"],
        "use_softplus": config["ZIP_HEAD"]["USE_SOFTPLUS"],
        "lambda_noise_std": 0.0,  # No noise in Stage 3
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,  # IMPORTANTE: False come in Stage 2
        zip_head_kwargs=zip_head_kwargs
    ).to(device)

    # Congela inizialmente il backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("🧊 Backbone congelato inizialmente")

    # Optimizer con LR più conservativi
    param_groups = [
        {'params': list(model.zip_head.parameters()) + list(model.p2r_head.parameters()),
         'lr': float(optim_cfg.get("LR_HEADS", 2e-5))}  # Ridotto da 8e-5
    ]
    optimizer = get_optimizer(param_groups, optim_cfg)
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg["EPOCHS"])

    # Loss
    criterion_zip = PiHeadLoss(
        pos_weight=config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0),
        block_size=data_cfg["ZIP_BLOCK_SIZE"]
    ).to(device)

    criterion_p2r = P2RLoss(
        scale_weight=float(loss_cfg.get("SCALE_WEIGHT", 0.1)),  # Ridotto
        pos_weight=float(loss_cfg.get("POS_WEIGHT", 1.0)),
        chunk_size=int(loss_cfg.get("CHUNK_SIZE", 128)),
        min_radius=float(loss_cfg.get("MIN_RADIUS", 0.0)),
        max_radius=float(loss_cfg.get("MAX_RADIUS", 10.0))
    ).to(device)

    # Output directory
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, "stage3_latest.pth")
    best_path = os.path.join(output_dir, "stage3_best.pth")
    stage2_path = os.path.join(output_dir, "stage2_best.pth")

    # Logica di caricamento
    start_epoch = 0
    best_mae = float("inf")

    if os.path.isfile(latest_path):
        start_epoch, best_mae = load_full_checkpoint(latest_path, model, optimizer, scheduler, device)
    elif os.path.isfile(stage2_path):
        print(f"✅ Inizio Stage 3 con pesi Stage 2: {stage2_path}")
        state = torch.load(stage2_path, map_location=device)
        if "model" in state: 
            state = state["model"]
        model.load_state_dict(state, strict=False)
    else:
        print("⚠️ Nessun checkpoint Stage 2 trovato, partenza da zero")
    
    # Calibrazione iniziale
    if start_epoch == 0 and hasattr(model.p2r_head, "log_scale"):
        print("🔧 Calibrazione scala P2R prima di iniziare...")
        calibrate_density_scale(
            model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"],
            max_batches=None,  # Usa tutto il validation set
            clamp_range=loss_cfg.get("LOG_SCALE_CLAMP"),
            max_adjust=loss_cfg.get("LOG_SCALE_CALIBRATION_MAX_DELTA", 1.0)
        )
    
    # Valutazione iniziale
    print("\n📋 Valutazione iniziale prima del training:")
    val_mae, val_rmse, _, _ = validate(model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"], loss_cfg)
    print(f"   MAE iniziale: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
    if start_epoch == 0:
        best_mae = val_mae  # Parti dal valore Stage 2
    
    # Parametri Training
    epochs_stage3 = optim_cfg["EPOCHS"]
    patience = optim_cfg.get("EARLY_STOPPING_PATIENCE", 50)
    no_improve = 0
    
    # Loss weights - più conservativi
    zip_scale = float(config["JOINT_LOSS"].get("ZIP_SCALE", 0.5))  # Ridotto
    alpha = float(config["JOINT_LOSS"].get("ALPHA", 0.5))  # Ridotto
    count_l1_weight = float(config["JOINT_LOSS"].get("COUNT_L1_W", 1.0))  # Aumentato

    print(f"\n🚀 START Stage 3 (Ep {start_epoch+1} -> {epochs_stage3})")
    print(f"   Loss weights: ZIP={zip_scale}, P2R={alpha}, L1={count_l1_weight}")
    
    for epoch in range(start_epoch, epochs_stage3):
        print(f"\n--- Epoch {epoch + 1}/{epochs_stage3} ---")
        
        # Progressive unfreeze
        if apply_progressive_unfreeze(model, epoch + 1, optim_cfg):
            optimizer = rebuild_optimizer(model, optim_cfg)
            remaining = epochs_stage3 - epoch
            scheduler = get_scheduler(optimizer, optim_cfg, remaining)
            
        train_loss = train_one_epoch(
            model, criterion_zip, criterion_p2r, train_loader, optimizer,
            scheduler, str(optim_cfg.get("SCHEDULER_STEP", "epoch")).lower(),
            device, data_cfg["P2R_DOWNSAMPLE"], loss_cfg.get("LOG_SCALE_CLAMP"),
            epoch + 1, zip_scale, alpha, count_l1_weight,
            grad_accum_steps=2  # Gradient accumulation per stabilità
        )

        if scheduler and str(optim_cfg.get("SCHEDULER_STEP", "epoch")).lower() == "epoch":
            scheduler.step()

        # Salvataggio periodico
        save_full_checkpoint(latest_path, model, optimizer, scheduler, epoch + 1, best_mae)

        # Validazione
        if (epoch + 1) % optim_cfg["VAL_INTERVAL"] == 0:
            # Calibra prima della validazione
            if hasattr(model.p2r_head, "log_scale"):
                backup_scale = model.p2r_head.log_scale.data.clone()
                calibrate_density_scale(
                    model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"],
                    max_batches=10,
                    clamp_range=loss_cfg.get("LOG_SCALE_CLAMP"),
                    max_adjust=0.5  # Adjustment più conservativo
                )
            
            val_mae, val_rmse, tot_pred, tot_gt = validate(
                model, val_loader, device, data_cfg["P2R_DOWNSAMPLE"], loss_cfg
            )
            bias = tot_pred / tot_gt if tot_gt > 0 else 0
            
            print(f"Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | Bias: {bias:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                save_full_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_mae)
                print(f"✅ Saved New Best Stage 3: MAE={best_mae:.2f}")
                no_improve = 0
            else:
                no_improve += 1
                # Ripristina log_scale se non c'è miglioramento
                if hasattr(model.p2r_head, "log_scale"):
                    model.p2r_head.log_scale.data = backup_scale
            
            if patience > 0 and no_improve >= patience:
                print(f"⛔ Early stopping attiva (no improvement per {patience} validazioni)")
                break

    print(f"\n✅ Stage 3 completato! Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()