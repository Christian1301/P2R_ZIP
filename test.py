import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn  # Assicurati che esista o rimuovi se non usi batch>1

# =============================================================================
# UTILITIES
# =============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_gt_count(points):
    """Estrae il conteggio ground truth gestendo liste e density maps."""
    if isinstance(points, (list, np.ndarray)):
        return float(len(points))
    elif isinstance(points, torch.Tensor):
        if points.dim() >= 2 and points.shape[-2] > 2: 
            # È una density map (H, W) o (1, H, W) -> Somma
            return points.sum().item()
        elif points.dim() == 2 and points.shape[1] == 2:
            # Tensore coordinate (N, 2)
            return float(points.shape[0])
        else:
            return points.item() if points.numel() == 1 else float(points.shape[0])
    return 0.0

# =============================================================================
# MODEL LOADER
# =============================================================================

def get_model(config, device):
    # Logica Dataset (identica a visualize_gating)
    dataset_section = config.get("DATASET", "shha")
    if isinstance(dataset_section, dict):
        dataset_name = dataset_section.get("NAME", "shha")
    else:
        dataset_name = dataset_section
    
    # Normalizzazione nomi
    alias_map = {'shha': 'shha', 'shhb': 'shhb', 'ucf': 'ucf', 'qnrf': 'ucf', 'nwpu': 'nwpu', 'jhu': 'jhu'}
    normalized = ''.join(ch for ch in str(dataset_name).lower() if ch.isalnum())
    dataset_key = alias_map.get(normalized, 'shha')

    # Bins config
    DEFAULT_BINS = {
        'shha': {'bins': [[0,0],[1,3],[4,6],[7,10],[11,15],[16,22],[23,32],[33,9999]], 'bin_centers': [0.0,2.0,5.0,8.5,13.0,19.0,27.5,45.0]},
        'shhb': {'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9999]], 'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.16]},
        'ucf':  {'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]], 'bin_centers': [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.4,13.4,15.4,17.4,19.4,21.8,24.8,27.8,31.2,38.8]},
    }
    
    bin_cfg = config.get("BINS_CONFIG", {}).get(dataset_key, DEFAULT_BINS.get(dataset_key, DEFAULT_BINS['shha']))
    
    model = P2R_ZIP_Model(
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        zip_head_kwargs={'use_softplus': True}
    ).to(device)
    
    return model

# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def evaluate(model, dataloader, device, args, learned_scale):
    model.eval()
    
    # Accumulatori errori
    errors = {
        'raw': [],
        'hard': [],
        'soft': []
    }
    
    # Accumulatori totali per Bias e Scale ideale
    sums = {
        'gt': 0.0,
        'pred_raw': 0.0,
        'pred_soft': 0.0,
        'pred_soft_unscaled': 0.0 
    }
    
    print(f"🚀 Inizio valutazione su {len(dataloader)} immagini...")
    print(f"⚙️  Parametri: Tau={args.tau}, Alpha={args.alpha}, Scale={learned_scale:.3f}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Gestione batch (supporta sia dict che tuple/list)
            if isinstance(batch, dict):
                images = batch['image']
                points = batch['points']
            else:
                images, points = batch[0], batch[1]
            
            images = images.to(device)
            # points può essere una lista di tensori o tensore unico
            
            # Forward
            outputs = model(images)
            raw_density = outputs['p2r_density']
            logits_pi = outputs['logit_pi_maps']
            pi_probs = torch.softmax(logits_pi, dim=1)[:, 1:2, :, :]
            
            # Loop nel batch (di solito batch_size=1 per test)
            for i in range(images.shape[0]):
                # Calcolo area cella per normalizzazione
                h_in, w_in = images.shape[2], images.shape[3]
                h_out, w_out = raw_density.shape[2], raw_density.shape[3]
                cell_area = (h_in / h_out) * (w_in / w_out)
                
                # GT Count
                curr_points = points[i] if isinstance(points, list) else points[i]
                gt = get_gt_count(curr_points)
                
                # --- STRATEGIA 1: RAW ---
                pred_raw = raw_density[i].sum().item() / cell_area
                
                # --- STRATEGIA 2: HARD GATING ---
                mask = (pi_probs[i] > args.tau).float()
                pred_hard = (raw_density[i] * mask).sum().item() / cell_area
                
                # --- STRATEGIA 3: SOFT FUSION ---
                # Formula: density * scale * ((1-alpha) + alpha * pi)
                soft_w = (1 - args.alpha) + args.alpha * pi_probs[i]
                
                # Calcolo intermedio per bias analysis
                pred_soft_unscaled = (raw_density[i] * soft_w).sum().item() / cell_area
                pred_soft = pred_soft_unscaled * learned_scale
                
                # Salvataggio errori
                errors['raw'].append(abs(pred_raw - gt))
                errors['hard'].append(abs(pred_hard - gt))
                errors['soft'].append(abs(pred_soft - gt))
                
                # Accumulo somme
                sums['gt'] += gt
                sums['pred_raw'] += pred_raw
                sums['pred_soft'] += pred_soft
                sums['pred_soft_unscaled'] += pred_soft_unscaled

    total_time = time.time() - start_time
    
    # Calcolo Metriche
    metrics = {}
    for k in errors:
        mae = np.mean(errors[k])
        mse = np.sqrt(np.mean(np.array(errors[k])**2))
        metrics[k] = (mae, mse)
        
    # Bias
    bias_raw = sums['pred_raw'] / sums['gt'] if sums['gt'] > 0 else 0
    bias_soft = sums['pred_soft'] / sums['gt'] if sums['gt'] > 0 else 0
    
    # Scale Analysis
    ideal_scale = sums['gt'] / sums['pred_soft_unscaled'] if sums['pred_soft_unscaled'] > 0 else 1.0
    
    return metrics, bias_raw, bias_soft, ideal_scale, total_time

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Globale P2R-ZIP Fusion")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tau", type=float, default=0.5, help="Soglia Hard Gating")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha Soft Fusion")
    parser.add_argument("--force-scale", type=float, default=None, help="Sovrascrivi scale factor appreso")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    
    # Checkpoint path
    if args.checkpoint is None:
        path = os.path.join(config['EXP']['OUT_DIR'], config['RUN_NAME'], "stage3_fusion_best.pth")
    else:
        path = args.checkpoint
        
    # Caricamento
    model = get_model(config, device)
    
    print(f"📥 Caricamento: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    
    # Estrazione Scale Factor
    learned_scale = 1.0
    if 'scale_comp' in state:
        try:
            ls = state['scale_comp']['log_scale']
            learned_scale = torch.exp(ls).item()
            print(f"⚖️  Scale Factor appreso: {learned_scale:.4f}")
        except:
            pass
            
    if args.force_scale is not None:
        print(f"⚠️  Override Scale: {learned_scale:.4f} -> {args.force_scale:.4f}")
        learned_scale = args.force_scale
        
    if 'model' in state: state = state['model']
    model.load_state_dict(state, strict=False)
    
    # Dataset
    # Usiamo lo split di test o val definito nel config
    data_cfg = config['DATASET'] if isinstance(config['DATASET'], dict) else config['DATA']
    if 'DATASET' in config and isinstance(config['DATASET'], dict): data_cfg.update(config['DATASET'])
    
    # Normalizzazione Dataset Name
    dataset_name_raw = data_cfg.get('NAME', 'shha')
    
    DatasetClass = get_dataset(str(dataset_name_raw).lower())
    val_trans = build_transforms(data_cfg, is_train=False)
    
    val_ds = DatasetClass(
        root=data_cfg['ROOT'],
        split=data_cfg.get('VAL_SPLIT', 'val'), # O 'test' se preferisci
        block_size=16,
        transforms=val_trans
    )
    
    # Importante: Batch size 1 per evitare errori di resize su immagini diverse
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    # Run Evaluation
    metrics, bias_raw, bias_soft, ideal_scale, dt = evaluate(model, val_loader, device, args, learned_scale)
    
    # Output Table
    t = PrettyTable(['Method', 'MAE', 'MSE', 'Bias', 'Notes'])
    t.align = 'l'
    
    t.add_row(['RAW (Stage 2)', f"{metrics['raw'][0]:.2f}", f"{metrics['raw'][1]:.2f}", f"{bias_raw:.3f}", "Baseline (No ZIP)"])
    t.add_row(['HARD (Stage 1+2)', f"{metrics['hard'][0]:.2f}", f"{metrics['hard'][1]:.2f}", "-", f"Binary Filter (Tau={args.tau})"])
    t.add_row(['SOFT (Stage 3)', f"{metrics['soft'][0]:.2f}", f"{metrics['soft'][1]:.2f}", f"{bias_soft:.3f}", f"Fusion (Alpha={args.alpha})"])
    
    print("\n" + "="*50)
    print(f"🏁 RISULTATI TEST GLOBALE ({dt:.1f}s)")
    print("="*50)
    print(t)
    print("\n📊 ANALISI SCALE FACTOR:")
    print(f"   Attuale (Used): {learned_scale:.4f}")
    print(f"   Ideale (Calc):  {ideal_scale:.4f}")
    diff = (learned_scale - ideal_scale) / ideal_scale * 100
    print(f"   Deviazione:     {diff:+.2f}%")
    
    if metrics['soft'][0] < metrics['raw'][0]:
        improv = metrics['raw'][0] - metrics['soft'][0]
        print(f"\n✅ SUCCESS: Soft Fusion migliora il MAE di {improv:.2f} punti!")
    else:
        print(f"\n⚠️ WARNING: Soft Fusion è peggiore del RAW. Controlla Alpha/Scale.")

if __name__ == "__main__":
    main()