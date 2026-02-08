import argparse
import os
import random
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms

# =============================================================================
# CONFIGURAZIONE DEFAULT
# =============================================================================

DEFAULT_STAGE_FILES = {
    "stage1": "best_model.pth",       # Solitamente ZIP pretrain
    "stage2": "stage2_best.pth",      # P2R pretrain
    "stage3": "stage3_fusion_best.pth" # Joint Training
}

# =============================================================================
# UTILITIES
# =============================================================================

def load_config(config_path: str) -> dict:
    print(f"📄 Caricamento configurazione: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converte tensore (C,H,W) normalizzato in numpy (H,W,C) BGR uint8."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    img = tensor.cpu().numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def draw_grid(img: np.ndarray, block_size: int, color=(0, 255, 255), thickness=1):
    """Disegna la griglia dei blocchi ZIP sull'immagine."""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Linee verticali
    for x in range(0, w, block_size):
        cv2.line(overlay, (x, 0), (x, h), color, thickness)
    
    # Linee orizzontali
    for y in range(0, h, block_size):
        cv2.line(overlay, (0, y), (w, y), color, thickness)
        
    return overlay

# =============================================================================
# MODEL LOADER
# =============================================================================

def get_model(config: dict, device: torch.device) -> P2R_ZIP_Model:
    # Gestione robusta DATASET (può essere stringa o dict)
    dataset_section = config.get("DATASET", "shha")
    if isinstance(dataset_section, dict):
        dataset_name = dataset_section.get("NAME", "shha")
    else:
        dataset_name = dataset_section
    
    # Normalizza nome dataset
    alias_map = {
        'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
        'shhb': 'shhb', 'shanghaitechpartb': 'shhb',
        'ucf': 'ucf', 'ucfqnrf': 'ucf', 'qnrf': 'ucf',
        'nwpu': 'nwpu', 'jhu': 'jhu'
    }
    normalized = ''.join(ch for ch in str(dataset_name).lower() if ch.isalnum())
    dataset_key = alias_map.get(normalized, 'shha')

    # Default bins per dataset
    DEFAULT_BINS = {
        'shha': {
            'bins': [[0, 0], [1, 3], [4, 6], [7, 10], [11, 15], [16, 22], [23, 32], [33, 9999]],
            'bin_centers': [0.0, 2.0, 5.0, 8.5, 13.0, 19.0, 27.5, 45.0],
        },
        'shhb': {
            'bins': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9999]],
            'bin_centers': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.16],
        },
        'ucf': {
            'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
            'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
        },
        'nwpu': {
            'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
            'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
        },
        'jhu': {
            'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
            'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
        },
    }

    bin_cfg = config.get("BINS_CONFIG", {}).get(dataset_key)
    if not bin_cfg:
        bin_cfg = DEFAULT_BINS.get(dataset_key, DEFAULT_BINS['shha'])
        print(f"ℹ️ Usando bins di default per {dataset_key}")
    
    bins, bin_centers = bin_cfg["bins"], bin_cfg["bin_centers"]

    model = P2R_ZIP_Model(
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        bins=bins,
        bin_centers=bin_centers,
        zip_head_kwargs={'use_softplus': True}
    ).to(device)
    return model

def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")
    
    print(f"📥 Caricamento pesi da: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    
    # Gestione scale_comp per Stage 3
    scale_val = 1.0
    if 'scale_comp' in state:
        try:
            log_scale = state['scale_comp']['log_scale']
            scale_val = torch.exp(log_scale).item()
            print(f"⚖️ Scale Compensation rilevato nel checkpoint: {scale_val:.3f}")
        except:
            print("⚠️ Impossibile leggere scale_comp dal checkpoint.")

    if 'model' in state:
        state = state['model']
        
    msg = model.load_state_dict(state, strict=False)
    print(f"   Stato caricamento: {msg}")
    model.eval()
    return scale_val

# =============================================================================
# VISUALIZATION LOGIC
# =============================================================================

def visualize_sample(
    model: nn.Module, 
    sample: dict, 
    device: torch.device, 
    args: argparse.Namespace,
    learned_scale: float = 1.0
):
    img_tensor = sample['image'].to(device).unsqueeze(0)
    gt_points = sample['points']
    
    # =====================================================
    # 1. FIX: Calcolo GT Count Robusto (Lista vs Mappa)
    # =====================================================
    gt_count = 0.0
    if isinstance(gt_points, (list, np.ndarray)):
        # Caso: Lista di punti [(x,y), ...]
        gt_count = float(len(gt_points))
    elif isinstance(gt_points, torch.Tensor):
        if gt_points.dim() == 2 and gt_points.shape[1] == 2:
            # Caso: Tensore coordinate [N, 2]
            gt_count = float(gt_points.shape[0])
        elif gt_points.dim() >= 2:
            # Caso: Density Map [1, H, W] o [H, W] -> SI SOMMANO I PIXEL
            gt_count = gt_points.sum().item()
        else:
            # Fallback
            gt_count = gt_points.item() if gt_points.numel() == 1 else float(gt_points.shape[0])
            
    print(f"\n🔢 VERIFICA CONTEGGIO REALE (GT): {gt_count:.2f}")
    print(f"   Tipo dati points: {type(gt_points)}")
    if isinstance(gt_points, torch.Tensor):
        print(f"   Shape points: {gt_points.shape}")
    
    # Forward Pass
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # Estrazione output
    raw_density = outputs['p2r_density']  # [1, 1, H_out, W_out]
    logits_pi = outputs['logit_pi_maps']  # [1, 2, H_out, W_out]
    pi_probs = torch.softmax(logits_pi, dim=1)[:, 1:2, :, :] # Probabilità classe 1 (folla)
    
    # Dimensioni
    _, _, h_in, w_in = img_tensor.shape
    _, _, h_out, w_out = raw_density.shape
    
    # Calcolo fattore di downsampling reale
    down_h, down_w = h_in / h_out, w_in / w_out
    cell_area = down_h * down_w
    
    # -----------------------------------------------------------
    # CALCOLO MAPPE FINALI (Hard vs Soft)
    # -----------------------------------------------------------
    
    # A. Raw Count (P2R puro)
    count_raw = raw_density.sum().item() / cell_area
    
    # B. Hard Gating (Simulazione Stage 1+2)
    mask_hard = (pi_probs > args.tau).float()
    density_hard = raw_density * mask_hard
    count_hard = density_hard.sum().item() / cell_area
    
    # C. Soft Fusion (Stage 3 logic)
    # Formula: density * scale * ((1-alpha) + alpha * pi)
    alpha = args.alpha
    scale = args.scale if args.scale is not None else learned_scale
    
    soft_weights = (1 - alpha) + alpha * pi_probs
    density_soft = raw_density * scale * soft_weights
    count_soft = density_soft.sum().item() / cell_area
    
    # -----------------------------------------------------------
    # PREPARAZIONE VISUALIZZAZIONE
    # -----------------------------------------------------------
    
    # Immagine originale
    img_np = denormalize_image(img_tensor.squeeze())
    h_img, w_img = img_np.shape[:2]
    
    # Resizing mappe per display (bilinear per density, nearest per mask)
    def resize_map(m, is_mask=False):
        m_np = m.squeeze().cpu().numpy()
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(m_np, (w_img, h_img), interpolation=interp)

    map_pi = resize_map(pi_probs)
    map_raw = resize_map(raw_density)
    map_soft = resize_map(density_soft)
    mask_disp = resize_map(mask_hard, is_mask=True)

    # Creazione overlay griglia (per mostrare le patch)
    # Block size visuale = block_size originale (16)
    img_grid = draw_grid(img_np, args.block_size, color=(255, 255, 255), thickness=1)
    
    # Creazione overlay patch classificate
    # Verde = Folla (pi > tau), Rosso = Sfondo (pi < tau)
    overlay_cls = img_np.copy()
    # Usa map_pi resizeata per creare maschere colore
    mask_fg = map_pi > args.tau
    overlay_cls[mask_fg] = overlay_cls[mask_fg] * 0.7 + np.array([0, 255, 0]) * 0.3 # Verde
    overlay_cls[~mask_fg] = overlay_cls[~mask_fg] * 0.7 + np.array([0, 0, 255]) * 0.3 # Rosso
    # Aggiungi griglia sopra
    overlay_cls = draw_grid(overlay_cls, args.block_size, color=(200, 200, 200), thickness=1)

    # -----------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    
    # Riga 1: Input e Analisi ZIP
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Input + Griglia Patch ({args.block_size}px)")
    ax1.axis('off')
    
    # SCRITTA SOTTO L'IMMAGINE
    ax1.text(0.5, -0.1, f"Persone Reali (GT): {gt_count:.1f}", 
             transform=ax1.transAxes, 
             ha='center', va='top', 
             fontsize=14, color='black', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(map_pi, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title("ZIP Confidence ($\pi$)\nGiallo=Folla, Viola=Sfondo")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(overlay_cls, cv2.COLOR_BGR2RGB))
    ax3.set_title(f"Classificazione Patch ($\\tau={args.tau}$)\nVerde=Attivo, Rosso=Filtrato")
    ax3.axis('off')
    
    # Istogramma Probabilità ZIP
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(map_pi.ravel(), bins=50, color='skyblue', edgecolor='black')
    ax4.axvline(args.tau, color='red', linestyle='--', label=f'Tau {args.tau}')
    ax4.set_title("Distribuzione Probabilità $\pi$")
    ax4.legend()
    
    # Riga 2: Density Maps e Conteggi
    
    # Raw Density (P2R puro)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(map_raw, cmap='jet')
    ax5.set_title(f"1. RAW Density (No ZIP)\nCount: {count_raw:.2f}\n(Spesso sovrastima lo sfondo)")
    ax5.axis('off')
    
    # Hard Gating (Vecchio metodo)
    ax6 = fig.add_subplot(gs[1, 1])
    masked_view = map_raw * mask_disp
    ax6.imshow(masked_view, cmap='jet')
    ax6.set_title(f"2. HARD Gating (Binary Filter)\nCount: {count_hard:.2f}\n(Rischio 'Buchi Neri')")
    ax6.axis('off')
    
    # Soft Fusion (Metodo Finale)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(map_soft, cmap='jet')
    title_soft = f"3. SOFT Fusion (Stage 3)\n$\\alpha={alpha}, Scale={scale:.2f}$\nCount: {count_soft:.2f} (Target)"
    ax7.set_title(title_soft, fontweight='bold', color='darkgreen')
    ax7.axis('off')
    
    # Log Density (per vedere dettagli bassi)
    ax8 = fig.add_subplot(gs[1, 3])
    log_map = np.log1p(map_soft)
    ax8.imshow(log_map, cmap='magma')
    ax8.set_title("Log Density (Dettagli Soft)\nMostra recupero info deboli")
    ax8.axis('off')
    
    plt.tight_layout()
    
    # Salvataggio
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    out_name = out_dir / f"gating_{args.stage}_{random.randint(0,9999)}.png"
    plt.savefig(out_name, dpi=150)
    print(f"💾 Visualizzazione salvata: {out_name}")
    plt.close()
    
    # Apri automaticamente l'immagine
    import subprocess
    import platform
    import shutil
    
    abs_path = str(out_name.resolve())
    system = platform.system()
    opened = False
    
    try:
        if system == "Linux":
            # Prova diversi visualizzatori comuni su Linux
            viewers = ["xdg-open", "eog", "feh", "display", "gpicview", "sxiv", "code"]
            for viewer in viewers:
                if shutil.which(viewer):
                    subprocess.Popen([viewer, abs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"🖼️ Immagine aperta con: {viewer}")
                    opened = True
                    break
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", abs_path])
            opened = True
        elif system == "Windows":
            os.startfile(abs_path)
            opened = True
            
        if not opened:
            print(f"📂 Nessun visualizzatore trovato. Apri manualmente:\n   {abs_path}")
    except Exception as e:
        print(f"⚠️ Impossibile aprire l'immagine automaticamente: {e}")
        print(f"📂 Percorso immagine: {abs_path}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza Gating P2R-ZIP")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--stage", type=str, default="stage3", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    
    # Parametri visualizzazione
    parser.add_argument("--block-size", type=int, default=16, help="Dimensione blocco ZIP")
    parser.add_argument("--tau", type=float, default=0.3, help="Soglia per visualizzazione Hard Gating")
    
    # Parametri Stage 3 Soft
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha per Soft Fusion (0.25 default)")
    parser.add_argument("--scale", type=float, default=None, help="Forza scale factor (se None, usa checkpoint)")
    
    # Selezione immagine
    parser.add_argument("--index", type=int, default=None, help="Indice immagine specifica")
    parser.add_argument("--random", action="store_true", help="Scegli immagine a caso")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    
    # Risoluzione Checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        base_dir = Path(config.get("EXP", {}).get("OUT_DIR", "exp")) / config.get("RUN_NAME", "run")
        ckpt_name = DEFAULT_STAGE_FILES.get(args.stage, "stage3_fusion_best.pth")
        ckpt_path = base_dir / ckpt_name
    
    # Modello
    model = get_model(config, device)
    learned_scale = load_checkpoint(model, str(ckpt_path), device)
    
    # Gestione robusta config DATA/DATASET
    dataset_section = config.get('DATASET', 'shha')
    data_section = config.get('DATA')

    if isinstance(dataset_section, dict):
        dataset_name_raw = dataset_section.get('NAME', 'shha')
        data_cfg = {}
        if isinstance(data_section, dict):
            data_cfg.update(data_section)
        data_cfg.update(dataset_section)
    else:
        dataset_name_raw = dataset_section
        data_cfg = data_section.copy() if isinstance(data_section, dict) else {}

    # Normalizza nome dataset
    alias_map = {
        'shha': 'shha', 'shanghaitecha': 'shha', 'shanghaitechparta': 'shha',
        'shhb': 'shhb', 'shanghaitechpartb': 'shhb',
        'ucf': 'ucf', 'ucfqnrf': 'ucf',
        'nwpu': 'nwpu', 'jhu': 'jhu'
    }
    normalized = ''.join(ch for ch in str(dataset_name_raw).lower() if ch.isalnum())
    dataset_name = alias_map.get(normalized, str(dataset_name_raw).lower())

    if 'NORM_MEAN' not in data_cfg:
        data_cfg['NORM_MEAN'] = [0.485, 0.456, 0.406]
    if 'NORM_STD' not in data_cfg:
        data_cfg['NORM_STD'] = [0.229, 0.224, 0.225]

    # Dataset Validazione
    val_trans = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(dataset_name)
    
    val_ds = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg.get("VAL_SPLIT", "val"),
        block_size=args.block_size,
        transforms=val_trans
    )
    
    # Selezione Sample
    if args.index is not None:
        idx = args.index
    else:
        idx = random.randint(0, len(val_ds)-1)
    
    print(f"🖼️ Visualizzazione immagine #{idx} dal dataset {dataset_name}")
    sample = val_ds[idx]
    # Gestione formato dataset (alcuni tornano tuple, altri dict)
    if isinstance(sample, tuple):
        sample = {'image': sample[0], 'points': sample[1]}
        
    # Esegui
    visualize_sample(model, sample, device, args, learned_scale)