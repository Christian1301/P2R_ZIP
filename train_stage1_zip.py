# P2R_ZIP/train_stage1_zip.py
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
from train_utils import (
    init_seeds,
    get_optimizer,
    get_scheduler,
    resume_if_exists,
    save_checkpoint,
    collate_fn,
)

# ============================================================
# NUOVA LOSS PER STAGE 1 (BCE - Ispirata a ZIP-CLIP-EBC)
# ============================================================
class PiHeadLoss(nn.Module):
    """
    Loss per Stage 1: Classificazione Binaria (Vuoto vs Pieno).
    Si concentra sull'imparare una maschera di occupazione pulita
    prima di stimare la densità.
    Inserita qui per sostituire la vecchia logica NLL.
    """
    def __init__(
        self,
        pos_weight: float = 5.0,  # Peso maggiore per i blocchi pieni (classe rara)
        block_size: int = 16,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.block_size = block_size
        
        # BCE con pos_weight per bilanciare le classi (foreground vs background)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def compute_gt_occupancy(self, gt_density):
        """Genera la maschera binaria GT dai punti/densità."""
        # Somma la densità nel blocco per vedere se contiene persone
        gt_counts_per_block = F.avg_pool2d(
            gt_density,
            kernel_size=self.block_size,
            stride=self.block_size
        ) * (self.block_size ** 2)
        
        # Se c'è almeno una frazione di persona (soglia bassa), il blocco è "pieno"
        gt_occupancy = (gt_counts_per_block > 1e-3).float()
        return gt_occupancy
    
    def forward(self, predictions, gt_density):
        # [B, 2, H, W] -> Il canale 1 è il logit per "pieno"
        logit_pi_maps = predictions["logit_pi_maps"]
        logit_pieno = logit_pi_maps[:, 1:2, :, :] 
        
        # Calcola la Ground Truth binaria
        gt_occupancy = self.compute_gt_occupancy(gt_density)
        
        # Allinea dimensioni se necessario (gestione arrotondamenti pooling)
        if gt_occupancy.shape[-2:] != logit_pieno.shape[-2:]:
            gt_occupancy = F.interpolate(
                gt_occupancy, 
                size=logit_pieno.shape[-2:], 
                mode='nearest'
            )
        
        # Sposta il peso sul device corretto se necessario
        if self.bce.pos_weight.device != logit_pieno.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logit_pieno.device)
            
        loss = self.bce(logit_pieno, gt_occupancy)
        
        # Restituisce loss scalare e dizionario per logging
        return loss, {"pi_bce_loss": loss.detach()}


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch=1,
    clip_grad_norm=1.0,
):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Train Stage 1 (Pi-Head BCE) - Epoch {epoch}")

    pi_means = []
    
    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        
        # Calcolo Loss (Solo BCE su Pi-Head)
        loss, loss_dict = criterion(predictions, gt_density)

        loss.backward()
        
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        # Monitoraggio media probabilità "pieno"
        pi_soft = predictions["logit_pi_maps"].softmax(dim=1)[:, 1:]
        pi_means.append(pi_soft.mean().detach().item())

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'bce': f"{loss_dict.get('pi_bce_loss', 0):.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    if scheduler:
        scheduler.step()
        
    if pi_means:
        print(f"   ↪ π_mean (avg prob. occupied)={np.mean(pi_means):.3f}")
        
    return total_loss / len(dataloader)


def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    mae, mse = 0.0, 0.0
    
    # Metriche per la maschera
    pi_values = []

    with torch.no_grad():
        for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 1"):
            images, gt_density = images.to(device), gt_density.to(device)

            preds = model(images)
            loss, _ = criterion(preds, gt_density)
            total_loss += loss.item()

            # Estrai probabilità occupazione (canale 1)
            pi_maps = preds["logit_pi_maps"].softmax(dim=1)
            pi_pieno = pi_maps[:, 1:2] 
            
            # Nota: In questo stage, Lambda (densità) non è allenato dalla loss BCE.
            # Il conteggio predetto sarà probabilmente errato (Lambda è random o inizializzato),
            # ma lo calcoliamo per monitoraggio.
            lam = preds["lambda_maps"]
            pred_count = torch.sum(pi_pieno * lam, dim=(1, 2, 3))
            gt_count = torch.sum(gt_density, dim=(1, 2, 3))

            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

            pi_values.append(pi_pieno.reshape(-1).cpu())

    # Statistiche Maschera
    if pi_values:
        pi_values_np = torch.cat(pi_values).numpy()
        active_ratio = (pi_values_np > 0.5).mean() * 100.0
        
        print("\n📊 ZIP Head Statistics (Mask):")
        print(f"  Active blocks (π>0.5): {active_ratio:.2f}% (Target approx: 5-20% per folle sparse)")
        print(
            "  π distribution: mean={:.3f}, std={:.3f}".format(
                pi_values_np.mean(), pi_values_np.std()
            )
        )
        print("  ⚠️ Nota: λ non è ottimizzato in Stage 1. Ignora MAE/RMSE se alti; guarda la Loss BCE.\n")

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_rmse = (mse / len(dataloader.dataset)) ** 0.5
    
    return avg_loss, avg_mae, avg_rmse


def main():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        print("❌ ERRORE: config.yaml non trovato.")
        return

    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])

    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_cfg["bins"], bin_cfg["bin_centers"]

    # Configurazione Modello
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
        upsample_to_input=config["MODEL"]["UPSAMPLE_TO_INPUT"],
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    # Congelamento: Congeliamo la P2R head e la parte Lambda della ZIP head.
    # Alleniamo SOLO la Pi-Head (classificazione) e il Backbone.
    for p in model.p2r_head.parameters():
        p.requires_grad = False
    
    # La loss PiHeadLoss userà solo logit_pi_maps, quindi i gradienti per lambda
    # saranno implicitamente zero, ma è buona norma saperlo.

    # Configurazione Loss Stage 1 (BCE)
    # Usa POS_WEIGHT_BCE dal config o default a 5.0 (buono per bilanciare background/foreground)
    pos_weight = config.get("ZIP_LOSS", {}).get("POS_WEIGHT_BCE", 5.0)
    print(f"🔧 Loss Config: PiHeadLoss (BCE) con pos_weight={pos_weight}")
    
    criterion = PiHeadLoss(
        pos_weight=pos_weight,
        block_size=config["DATA"]["ZIP_BLOCK_SIZE"],
    ).to(device)

    # Optimizer
    optim_cfg = config["OPTIM_ZIP"]
    lr_head = optim_cfg.get("LR", optim_cfg.get("BASE_LR", 1e-4))
    lr_backbone = optim_cfg.get("LR_BACKBONE", lr_head * 0.1) # Backbone più lento

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.zip_head.parameters() if p.requires_grad]

    optimizer = get_optimizer(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        optim_cfg,
    )
    scheduler = get_scheduler(optimizer, optim_cfg, max_epochs=optim_cfg.get("EPOCHS", 100))

    # Dataset & Dataloader
    data_cfg = config["DATA"]
    train_tf = build_transforms(data_cfg, is_train=True)
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    train_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=train_tf,
    )
    val_set = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_tf,
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=optim_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=optim_cfg["NUM_WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Output Dir
    out_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    os.makedirs(out_dir, exist_ok=True)
    
    # Resume
    start_epoch, best_metric = resume_if_exists(model, optimizer, out_dir, device)
    
    # Se riprendiamo da 0 o cambiamo logica, resettiamo la metrica migliore.
    # Qui usiamo la Validation Loss (BCE) come metrica migliore perché il MAE non è affidabile.
    if start_epoch == 1:
        best_metric = float('inf')

    print(f"🚀 Inizio addestramento Stage 1 (BCE Mode) per {optim_cfg['EPOCHS']} epoche...")
    val_interval = max(1, optim_cfg.get("VAL_INTERVAL", 1))
    es_patience = max(0, int(optim_cfg.get("EARLY_STOPPING_PATIENCE", 0)))
    es_delta = float(optim_cfg.get("EARLY_STOPPING_DELTA", 0.0))
    epochs_no_improve = 0
    early_stop_triggered = False

    for epoch in range(start_epoch, optim_cfg["EPOCHS"] + 1):
        print(f"\n--- Epoch {epoch}/{optim_cfg['EPOCHS']} ---")
        
        train_loss = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            clip_grad_norm=1.0,
        )

        if epoch % val_interval == 0 or epoch == optim_cfg["EPOCHS"]:
            # Validazione
            val_loss, val_mae, val_rmse = validate(model, criterion, val_loader, device)
            print(f"Val → Loss (BCE) {val_loss:.4f} | MAE {val_mae:.2f}")

            # Salvataggio basato sulla Loss (BCE), non sul MAE
            current_metric = val_loss
            improvement = current_metric < (best_metric - es_delta)
            is_best = improvement

            if improvement:
                best_metric = current_metric
                epochs_no_improve = 0
            elif es_patience > 0:
                epochs_no_improve += 1

            save_checkpoint(
                model, optimizer, epoch, current_metric, best_metric, out_dir, is_best=is_best
            )
            
            if is_best:
                print(f"✅ Saved new best model (Loss={best_metric:.4f})")
            elif es_patience > 0:
                remaining = es_patience - epochs_no_improve
                print(
                    f"Nessun miglioramento (delta<{es_delta:.4f}). Early stop tra {max(remaining, 0)} validazioni."
                )
                if epochs_no_improve >= es_patience:
                    print("⛔ Early stopping attiva per Stage 1.")
                    early_stop_triggered = True

            if early_stop_triggered:
                break

    if early_stop_triggered:
        print("✅ Stage 1 terminato per early stopping.")
    else:
        print("✅ Addestramento Stage 1 completato.")


if __name__ == "__main__":
    main()