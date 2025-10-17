# P2R_ZIP/train_stage1_zip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
# Importa le nuove transforms
from datasets.transforms import build_transforms
from train_utils import init_seeds, get_optimizer, get_scheduler, resume_if_exists, save_checkpoint, collate_fn

def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 1 (ZIP)")

    # La collate_fn ora restituisce (images, densities, points_list)
    for images, gt_density, _ in progress_bar: # Ignora i points per la loss ZIP
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images) # Il modello in train mode restituisce il dizionario completo
        loss, loss_dict = criterion(predictions, gt_density) # La loss ZIP usa solo le predizioni ZIP e gt_density

        loss.backward()
        # Aggiungi gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}", 'count': f"{loss_dict['zip_count_loss']:.4f}",
            'lr_head': f"{optimizer.param_groups[-1]['lr']:.6f}" # Legge LR dell'ultimo gruppo (head)
        })

    # Lo scheduler step va fatto dopo l'epoca, non dopo ogni batch per CosineAnnealingLR
    if scheduler:
        scheduler.step()
    return total_loss / len(dataloader)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0

    block_size = criterion.zip_block_size

    with torch.no_grad():
        # La collate_fn restituisce (images, densities, points_list)
        for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 1"): # Ignora i points
            images, gt_density = images.to(device), gt_density.to(device)

            # Metti in train mode per ottenere il dizionario completo, poi rimetti in eval
            model.train()
            predictions = model(images) #
            model.eval()

            # Calcola la loss (utile per monitoraggio, anche se non usata per ottimizzare)
            loss, loss_dict = criterion(predictions, gt_density) #
            total_loss += loss.item()

            # Calcola il conteggio predetto usando la formula ZIP: (1 - prob_zero) * lambda
            # Estrai pi_zero (probabilità che il blocco sia vuoto) dai logit
            pi_maps = predictions["logit_pi_maps"].softmax(dim=1)
            pi_zero = pi_maps[:, 0:1] # Forma [B, 1, Hb, Wb]
            lambda_maps = predictions["lambda_maps"] # Forma [B, 1, Hb, Wb]
            # Conteggio atteso per blocco
            pred_counts_per_block = (1.0 - pi_zero) * lambda_maps #

            # Somma i conteggi attesi per blocco per ottenere il conteggio totale predetto per immagine
            pred_count = torch.sum(pred_counts_per_block, dim=(1, 2, 3)) # Forma [B]

            # Calcola il conteggio ground truth a bassa risoluzione (conteggio per blocco)
            gt_counts_per_block = F.avg_pool2d(gt_density, kernel_size=block_size) * (block_size**2)
            gt_count = torch.sum(gt_counts_per_block, dim=(1, 2, 3)) # Forma [B]

            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_mse = (mse / len(dataloader.dataset)) ** 0.5
    return avg_loss, avg_mae, avg_mse

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

    dataset_name = config['DATASET']
    bin_config = config['BINS_CONFIG'][dataset_name]
    bins, bin_centers = bin_config['bins'], bin_config['bin_centers']

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config['MODEL']['BACKBONE'],
        pi_thresh=config['MODEL']['ZIP_PI_THRESH'],
        gate=config['MODEL']['GATE'],
        upsample_to_input=config['MODEL']['UPSAMPLE_TO_INPUT']
    ).to(device) #

    # Congela la testa P2R PRIMA di creare l'ottimizzatore
    for param in model.p2r_head.parameters():
        param.requires_grad = False

    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config['ZIP_LOSS']['WEIGHT_CE'],
        zip_block_size=config['DATA']['ZIP_BLOCK_SIZE']
    ).to(device) #

    # Configurazione specifica per lo Stage 1
    optim_config = config['OPTIM_ZIP'] #
    lr_head = optim_config['LR']
    lr_backbone = optim_config.get('LR_BACKBONE', lr_head)

    # Separa i parametri per LR differenziato
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    # Nello Stage 1 addestriamo solo la zip_head (oltre al backbone)
    head_params = [p for p in model.zip_head.parameters() if p.requires_grad]

    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ]

    # Crea l'ottimizzatore con i gruppi di parametri
    optimizer = get_optimizer(param_groups, optim_config) #

    # --- Crea le Transforms e i Datasets ---
    data_config = config['DATA']
    # Costruisce le pipeline di trasformazione usando la funzione da datasets/transforms.py
    train_transforms = build_transforms(data_config, is_train=True)
    val_transforms = build_transforms(data_config, is_train=False)

    DatasetClass = get_dataset(config['DATASET']) #
    # Crea i dataset passando le transforms appropriate
    train_dataset = DatasetClass(
        root=data_config['ROOT'],
        split=data_config['TRAIN_SPLIT'],
        block_size=data_config['ZIP_BLOCK_SIZE'],
        transforms=train_transforms # Passa le transforms di training
    )
    val_dataset = DatasetClass(
        root=data_config['ROOT'],
        split=data_config['VAL_SPLIT'],
        block_size=data_config['ZIP_BLOCK_SIZE'],
        transforms=val_transforms # Passa le transforms di validazione
    )
    # --- Fine creazione Transforms e Datasets ---

    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=optim_config['NUM_WORKERS'],
        collate_fn=collate_fn, # Usa la collate_fn da train_utils
        pin_memory=True, # Opzionale: migliora trasferimento dati a GPU
        drop_last=True # Consigliato per evitare batch piccoli alla fine
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Validazione sempre con batch_size 1
        shuffle=False,
        num_workers=optim_config['NUM_WORKERS'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    epochs_stage1 = optim_config.get('EPOCHS', 1300)
    scheduler = get_scheduler(optimizer, optim_config, max_epochs=epochs_stage1) #

    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    os.makedirs(output_dir, exist_ok=True)

    start_epoch, best_mae_val = resume_if_exists(model, optimizer, output_dir, device) #

    # --- Training Loop ---
    print(f"🚀 Inizio addestramento Stage 1 per {epochs_stage1} epoche...")
    for epoch in range(start_epoch, epochs_stage1 + 1):
        print(f"--- Epoch {epoch}/{epochs_stage1} ---")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device, clip_grad_norm=1.0) # Aggiunto clip_grad_norm

        # Esegui validazione a intervalli regolari
        if epoch % optim_config['VAL_INTERVAL'] == 0 or epoch == epochs_stage1:
            val_loss, val_mae, val_rmse = validate(model, criterion, val_loader, device)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}")

            is_best = val_mae < best_mae_val
            if is_best:
                best_mae_val = val_mae

            # Salva checkpoint (last e best)
            if config['EXP']['SAVE_BEST']:
                save_checkpoint(model, optimizer, epoch, val_mae, best_mae_val, output_dir, is_best=is_best) #
                if is_best:
                    print(f"✅ Saved new best model with MAE: {best_mae_val:.2f}")
            else: # Salva comunque l'ultimo checkpoint se SAVE_BEST è False
                 save_checkpoint(model, optimizer, epoch, val_mae, best_mae_val, output_dir, is_best=False)

    print("✅ Addestramento Stage 1 completato.")

if __name__ == '__main__':
    main()