# evaluate_stage1.py
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
from train_utils import init_seeds, collate_fn

@torch.no_grad()
def validate_checkpoint(model, criterion, dataloader, device, config, checkpoint_path): # Passa anche config e path
    """
    Esegue la validazione su un modello caricato.
    Utilizza il Metodo 1 per il calcolo delle metriche (coerente con la loss).
    """
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    
    block_size = criterion.zip_block_size
    
    progress_bar = tqdm(dataloader, desc="Validating Checkpoint")
    
    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)

        model.train() 
        predictions = model(images)
        model.eval() 

        loss, loss_dict = criterion(predictions, gt_density)
        total_loss += loss.item()

        # --- CALCOLO CORRETTO (METODO 1) ---
        pred_count = torch.sum(predictions["pred_density_zip"], dim=(1, 2, 3))
        gt_counts_per_block = F.avg_pool2d(gt_density, kernel_size=block_size) * (block_size**2)
        gt_count = torch.sum(gt_counts_per_block, dim=(1, 2, 3))
        
        mae += torch.abs(pred_count - gt_count).sum().item()
        mse += ((pred_count - gt_count) ** 2).sum().item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_mse = (mse / len(dataloader.dataset)) ** 0.5
    
    print("\n--- Risultati della Valutazione ---")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Dataset:      {config['DATASET']} (split: {config['DATA']['VAL_SPLIT']})")
    print(f"  Immagini:     {len(dataloader.dataset)}")
    print("-------------------------------------")
    print(f"  Validation Loss: {avg_loss:.4f}")
    print(f"  MAE:             {avg_mae:.2f}")
    print(f"  RMSE:            {avg_mse:.2f}")
    print("-------------------------------------")


def main(config, checkpoint_path):
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

    # Carica il checkpoint specificato
    if not os.path.isfile(checkpoint_path):
        print(f"Errore: Checkpoint non trovato in {checkpoint_path}")
        return

    print(f"Caricamento checkpoint da {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    print("Caricamento completato.")

    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config['ZIP_LOSS']['WEIGHT_CE'],
        zip_block_size=config['DATA']['ZIP_BLOCK_SIZE']
    ).to(device)
    
    # --- MODIFICHE PER NUOVO CONFIG E DATASET ---
    data_config = config['DATA']
    optim_config = config['OPTIM_ZIP'] # Legge la sezione corretta
    
    DatasetClass = get_dataset(config['DATASET'])
    val_dataset = DatasetClass(
        root=data_config['ROOT'], 
        split=data_config['VAL_SPLIT'], 
        block_size=data_config['ZIP_BLOCK_SIZE'] # Passa il block_size
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=optim_config['NUM_WORKERS'], # Legge da optim_config
        collate_fn=collate_fn
    )
    # --- FINE MODIFICHE ---

    # Lancia la validazione
    validate_checkpoint(model, criterion, val_loader, device, config, checkpoint_path)

if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # --- SPECIFICA QUALE CHECKPOINT USARE ---
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    
    CHECKPOINT_PATH = os.path.join(output_dir, "best_model.pth") 
    # CHECKPOINT_PATH = os.path.join(output_dir, "last.pth") 

    main(config, CHECKPOINT_PATH)