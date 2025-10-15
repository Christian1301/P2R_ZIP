# P2R_ZIP/train_stage1_zip.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
from train_utils import init_seeds, get_optimizer, get_scheduler, resume_if_exists, save_checkpoint, collate_fn

def train_one_epoch(model, criterion, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train Stage 1 (ZIP)")

    for images, gt_density, _ in progress_bar:
        images, gt_density = images.to(device), gt_density.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, gt_density)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}", 'count': f"{loss_dict['zip_count_loss']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    return total_loss / len(dataloader)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for images, gt_density, _ in tqdm(dataloader, desc="Validate Stage 1"):
            images, gt_density = images.to(device), gt_density.to(device)

            predictions = model(images)
            loss, _ = criterion(predictions, gt_density)
            total_loss += loss.item()

            pred_density_zip_upsampled = nn.functional.interpolate(
                predictions["pred_density_zip"], size=gt_density.shape[-2:], mode='bilinear', align_corners=False
            )
            pred_count = torch.sum(pred_density_zip_upsampled, dim=(1, 2, 3))
            gt_count = torch.sum(gt_density, dim=(1, 2, 3))
            
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
    ).to(device)

    for param in model.p2r_head.parameters():
        param.requires_grad = False

    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config['ZIP_LOSS']['WEIGHT_CE'],
        zip_block_size=config['DATA']['ZIP_BLOCK_SIZE']
    ).to(device)
    
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config)
    
    DatasetClass = get_dataset(config['DATASET'])
    train_dataset = DatasetClass(root=config['DATA']['ROOT'], split=config['DATA']['TRAIN_SPLIT'])
    val_dataset = DatasetClass(root=config['DATA']['ROOT'], split=config['DATA']['VAL_SPLIT'])
    
    # Usa di nuovo la collate_fn, che ora è corretta
    train_loader = DataLoader(train_dataset, batch_size=config['OPTIM']['BATCH_SIZE'], shuffle=True, num_workers=config['OPTIM']['NUM_WORKERS'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['OPTIM']['NUM_WORKERS'], collate_fn=collate_fn)

    epochs_stage1 = config['OPTIM'].get('EPOCHS_STAGE1', 1300)
    scheduler = get_scheduler(optimizer, config, max_epochs=epochs_stage1)

    best_mae = float('inf')
    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    os.makedirs(output_dir, exist_ok=True)
    
    start_epoch, best_mae = resume_if_exists(model, optimizer, output_dir, device)
    
    for epoch in range(start_epoch, epochs_stage1 + 1):
        print(f"--- Epoch {epoch}/{epochs_stage1} ---")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device)
        
        if (epoch + 1) % config['OPTIM']['VAL_INTERVAL'] == 0:
            val_loss, val_mae, val_mse = validate(model, criterion, val_loader, device)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}, Val RMSE: {val_mse:.2f}")

            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae
            
            if config['EXP']['SAVE_BEST']:
                save_checkpoint(model, optimizer, epoch, val_mae, best_mae, output_dir, is_best=is_best)
                if is_best:
                    print(f"✅ Saved new best model with MAE: {best_mae:.2f}")

if __name__ == '__main__':
    main()