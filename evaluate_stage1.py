# evaluate_stage1_diagnostics.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.p2r_zip_model import P2R_ZIP_Model
from losses.composite_loss import ZIPCompositeLoss
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, collate_fn, load_config


@torch.no_grad()
def validate_checkpoint(model, criterion, dataloader, device, config, checkpoint_path):
    """
    Validazione dettagliata Stage 1 (ZIP) con diagnostica su pi/lambda.
    """
    model.eval()
    total_loss, mae, mse = 0.0, 0.0, 0.0
    block_size = criterion.zip_block_size

    print("\n===== DEBUG ZIP HEAD =====")
    print("Controllo range di pi (probabilità blocco occupato) e lambda (intensità Poisson)")
    print("------------------------------------------------------")

    progress_bar = tqdm(dataloader, desc="Validating ZIP Stage 1")

    for idx, (images, gt_density, _) in enumerate(progress_bar):
        images, gt_density = images.to(device), gt_density.to(device)

        preds = model(images)
        loss, loss_dict = criterion(preds, gt_density)
        total_loss += loss.item()

        pi_logits = preds["logit_pi_maps"]       
        lam_maps  = preds["lambda_maps"]        
        pi_softmax = torch.softmax(pi_logits, dim=1)
        pi_not_zero = pi_softmax[:, 1:]       
        pi_mean = pi_not_zero.mean().item()
        pi_min, pi_max = pi_not_zero.min().item(), pi_not_zero.max().item()
        lam_mean = lam_maps.mean().item()
        lam_min, lam_max = lam_maps.min().item(), lam_maps.max().item()
        pct_over_01 = (pi_not_zero > 0.1).float().mean().item() * 100
        pct_over_05 = (pi_not_zero > 0.5).float().mean().item() * 100

        pred_density_zip = pi_not_zero * lam_maps
        pred_count = torch.sum(pred_density_zip).item()

        gt_counts_per_block = F.avg_pool2d(gt_density, kernel_size=block_size) * (block_size**2)
        gt_count = torch.sum(gt_counts_per_block).item()

        mae += abs(pred_count - gt_count)
        mse += (pred_count - gt_count) ** 2

        if idx % 10 == 0:
            print(f"[IMG {idx:03d}] pi:[{pi_min:.3f},{pi_max:.3f}] mean={pi_mean:.3f} "
                  f"| λ:[{lam_min:.3f},{lam_max:.3f}] mean={lam_mean:.3f} "
                  f"| >0.1={pct_over_01:.2f}% >0.5={pct_over_05:.2f}% "
                  f"| pred={pred_count:.1f}, gt={gt_count:.1f}")

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'nll': f"{loss_dict['zip_nll_loss']:.4f}",
            'ce': f"{loss_dict['zip_ce_loss']:.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_mae = mae / len(dataloader.dataset)
    avg_rmse = (mse / len(dataloader.dataset)) ** 0.5

    print("\n--- RISULTATI VALIDAZIONE ZIP ---")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Dataset:      {config['DATASET']} (split: {config['DATA']['VAL_SPLIT']})")
    print("-------------------------------------")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"MAE:             {avg_mae:.2f}")
    print(f"RMSE:            {avg_rmse:.2f}")
    print("-------------------------------------\n")


def main(config, checkpoint_path):
    device = torch.device(config['DEVICE'])
    init_seeds(config['SEED'])

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
        pi_mode=config['MODEL'].get('ZIP_PI_MODE', 'hard'),
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    if not os.path.isfile(checkpoint_path):
        print(f"❌ Errore: Checkpoint non trovato in {checkpoint_path}")
        return

    print(f"✅ Caricamento checkpoint da {checkpoint_path}...")
    raw_state = torch.load(checkpoint_path, map_location=device)
    state_dict = raw_state.get('model', raw_state)

    density_key = 'p2r_head.density_scale'
    log_scale_key = 'p2r_head.log_scale'
    if density_key in state_dict and log_scale_key not in state_dict:
        density_scale = state_dict.pop(density_key)
        state_dict[log_scale_key] = torch.log(density_scale.clamp(min=1e-8))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"ℹ️ Parametri mancanti ignorati: {missing}")
    if unexpected:
        print(f"ℹ️ Parametri extra ignorati: {unexpected}")
    print("✅ Caricamento completato.\n")

    criterion = ZIPCompositeLoss(
        bins=bins,
        weight_ce=config['ZIP_LOSS']['WEIGHT_CE'],
        zip_block_size=config['DATA']['ZIP_BLOCK_SIZE'],
        count_weight=config['ZIP_LOSS'].get('WEIGHT_COUNT', 1.0)
    ).to(device)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 (ZIP)")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path; defaults to RUN_NAME best.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    output_dir = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'])
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "best_model.pth")

    main(config, checkpoint_path)
