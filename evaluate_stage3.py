import argparse
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn, canonicalize_p2r_grid

@torch.no_grad()
def predict_tta(model, images, device):
    """
    Esegue predizione con Test-Time Augmentation (Originale + Horizontal Flip).
    Ritorna la media delle density map per migliorare la robustezza.
    """
    # 1. Predizione Immagine Standard
    output_orig = model(images)
    pred_orig = output_orig['p2r_density']
    
    # 2. Predizione Immagine Specchiata (Flip Orizzontale)
    # Flip sull'asse width (dimensione 3: B, C, H, W)
    images_flip = torch.flip(images, dims=[3]) 
    output_flip = model(images_flip)
    pred_flip_raw = output_flip['p2r_density']
    
    # "Giriamo" indietro la density map flippata per allinearla all'originale
    # (Anche se per il conteggio la somma non cambia, è corretto farlo per la mappa)
    pred_flip = torch.flip(pred_flip_raw, dims=[3])
    
    # 3. Media (Average Ensemble)
    pred_final = (pred_orig + pred_flip) / 2.0
    
    return pred_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path al file config')
    # Default cerca stage3_best, ma puoi specificare altro
    parser.add_argument('--ckpt', type=str, default=None, help='Path manuale al checkpoint')
    parser.add_argument('--tta', action='store_true', help='Abilita Test-Time Augmentation (Flip)')
    args = parser.parse_args()

    # Carica Configurazione
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config non trovato: {args.config}")
        
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['DEVICE'] if torch.cuda.is_available() else 'cpu')
    
    # Determinazione automatica checkpoint se non specificato
    if args.ckpt is None:
        # Cerca nell'ordine: Stage 3 Best -> Stage 2 Best -> Last
        base_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
        candidates = [
            os.path.join(base_dir, "stage3_best.pth"),
            os.path.join(base_dir, "stage2_best.pth"),
            os.path.join(base_dir, "best_model.pth")
        ]
        for c in candidates:
            if os.path.exists(c):
                args.ckpt = c
                break
        if args.ckpt is None:
            raise FileNotFoundError("Nessun checkpoint trovato automaticamente. Usare --ckpt.")

    print(f"🚀 Avvio Valutazione su: {args.ckpt}")
    print(f"🛠️  Modalità TTA: {'ATTIVA (Flip + Avg)' if args.tta else 'DISATTIVA (Standard)'}")

    # Dataset & Loader (Validation Split)
    val_transforms = build_transforms(cfg['DATA'], is_train=False)
    DatasetClass = get_dataset(cfg['DATASET'])
    
    val_dataset = DatasetClass(
        root=cfg['DATA']['ROOT'],
        split=cfg['DATA']['VAL_SPLIT'],
        transforms=val_transforms
    )
    
    # Batch size 1 è obbligatorio per valutazione accurata su immagini di dimensione varia
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Modello
    bin_cfg = cfg["BINS_CONFIG"][cfg["DATASET"]]
    # Controlla se usare Soft Gating (STE)
    use_ste = cfg["MODEL"].get("USE_STE_MASK", False)
    
    model = P2R_ZIP_Model(
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"], 
        bin_centers=bin_cfg["bin_centers"],
        upsample_to_input=False,
        use_ste_mask=use_ste
    ).to(device)

    # Carica Pesi
    print(f"📥 Caricamento pesi...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Loop di Valutazione
    mae_list = []
    mse_list = []

    pbar = tqdm(val_loader, desc="Valutazione")
    for images, _, points in pbar:
        images = images.to(device)
        gt_count = len(points[0])

        with torch.no_grad():
            if args.tta:
                # Usa la funzione TTA custom
                pred_density = predict_tta(model, images, device)
            else:
                # Inferenza standard
                out = model(images)
                pred_density = out['p2r_density']

            # Post-processing per il conteggio
            # Canonicalize gestisce il downsampling factor (es. 8x)
            _, _, H, W = images.shape
            pred_density, (dh, dw), _ = canonicalize_p2r_grid(pred_density, (H, W), 8.0)
            
            # Somma della density map normalizzata per l'area del blocco
            cell_area = dh * dw
            pred_count = pred_density.sum().item() / cell_area

        # Metriche
        error = abs(pred_count - gt_count)
        mae_list.append(error)
        mse_list.append(error ** 2)
        
        # Aggiorna la barra con il MAE corrente
        current_mae = np.mean(mae_list)
        pbar.set_postfix({'MAE': f"{current_mae:.2f}"})

    # Risultati Finali
    final_mae = np.mean(mae_list)
    final_rmse = np.sqrt(np.mean(mse_list))

    print("\n" + "="*40)
    print(f"📊 RISULTATI ({'TTA' if args.tta else 'Standard'})")
    print("="*40)
    print(f"MAE:  {final_mae:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()