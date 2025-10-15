# P2R_ZIP/train_stage3_joint.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from losses.p2r_region_loss import P2RLoss  # Assumendo che P2RLoss sia il nome corretto della classe
from datasets import get_dataset
from train_utils import init_seeds, get_optimizer, get_scheduler

def train_one_epoch(model, criterion_zip, criterion_p2r, alpha, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 3 (Joint)")

    for images, gt_density, points in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        predictions = model(images)

        # Calcola le due loss separatamente usando i nuovi criteri
        loss_zip, loss_dict_zip = criterion_zip(predictions, gt_density)
        loss_p2r = criterion_p2r(predictions['p2r_density'], points)
        
        # Combina le loss con il peso alpha
        combined_loss = loss_zip + alpha * loss_p2r
        combined_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += combined_loss.item()
        progress_bar.set_postfix({
            'total': f"{combined_loss.item():.4f}",
            'zip': f"{loss_zip.item():.4f}",
            'p2r': f"{loss_p2r.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    mae, mse = 0.0, 0.0
    
    with torch.no_grad():
        for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 3"):
            images, gt_density = images.to(device), gt_density.to(device)

            # In modalità eval, il modello restituisce la densità finale da P2R
            predictions = model(images)
            pred_density = predictions['density']
            
            pred_count = torch.sum(pred_density, dim=(1, 2, 3))
            gt_count = torch.sum(gt_density, dim=(1, 2, 3))
            
            mae += torch.abs(pred_count - gt_count).sum().item()
            mse += ((pred_count - gt_count) ** 2).sum().item()

    avg_mae = mae / len(dataloader.dataset)
    avg_mse = (mse / len(dataloader.dataset)) ** 0.5
    return avg_mae, avg_mse

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
    ).to(device)
    
    # Carica il checkpoint dal miglior modello dello Stage 2
    # Questo checkpoint contiene già i pesi addestrati dello Stage 1
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    stage2_checkpoint_path = os.path.join(output_dir, "stage2_best.pth") # Assumendo questo nome dal tuo stage 2
    
    try:
        model.load_state_dict(torch.load(stage2_checkpoint_path))
        print(f"✅ Caricati i pesi dal checkpoint dello Stage 2: {stage2_checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Checkpoint dello Stage 2 non trovato in {stage2_checkpoint_path}. Si parte da pesi pre-addestrati (se presenti).")

    # Sblocca tutti i parametri per il fine-tuning congiunto
    for param in model.parameters():
        param.requires_grad = True

    # Definisci entrambi i criteri di loss
    criterion_zip = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config['ZIP_LOSS']['WEIGHT_CE'],
        zip_block_size=config['DATA']['ZIP_BLOCK_SIZE']
    ).to(device)
    
    criterion_p2r = P2RLoss().to(device) # Assicurati che P2RLoss non richieda parametri
    
    # Usa un learning rate più basso per il fine-tuning
    optimizer_config = config['OPTIM'].copy()
    optimizer_config['LR'] = config['OPTIM'].get('LR_STAGE3', 1e-5) # Leggi da config o usa un default
    optimizer = get_optimizer(model.parameters(), config) # get_optimizer deve essere adattato a questa struttura
    
    train_dataset, val_dataset = get_dataset(config)
    
    train_loader = DataLoader(train_dataset, batch_size=config['OPTIM']['BATCH_SIZE'], shuffle=True, num_workers=config['OPTIM']['NUM_WORKERS'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['OPTIM']['NUM_WORKERS'])

    epochs_stage3 = config['OPTIM'].get('EPOCHS_STAGE3', 400) # Leggi da config o usa un default
    scheduler = get_scheduler(optimizer, config, max_epochs=epochs_stage3)
    alpha = config['LOSS']['JOINT_ALPHA']

    best_mae = float('inf')
    
    for epoch in range(epochs_stage3):
        print(f"--- Epoch {epoch+1}/{epochs_stage3} ---")
        train_loss = train_one_epoch(model, criterion_zip, criterion_p2r, alpha, train_loader, optimizer, scheduler, device)
        
        if (epoch + 1) % config['OPTIM']['VAL_INTERVAL'] == 0:
            val_mae, val_mse = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.2f}, Val RMSE: {val_mse:.2f}")

            if val_mae < best_mae:
                best_mae = val_mae
                if config['EXP']['SAVE_BEST']:
                    torch.save(model.state_dict(), os.path.join(output_dir, "stage3_best.pth"))
                    print(f"Saved best model with MAE: {best_mae:.2f}")

if __name__ == '__main__':
    main()