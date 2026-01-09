import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms

CONFIG_PATH = "config.yaml"

# RIMOSSO STAGE 4
DEFAULT_STAGE_FILES = {
    "stage1": "best_model.pth",      # Best Loss (o stage1_best_acc.pth se preferito)
    "stage2": "stage2_best.pth",
    "stage3": "stage3_best.pth",
}

DEFAULT_TAU_VALUES: List[float] = [0.2, 0.4, 0.6, 0.8]

def normalize_stage(label: Optional[str]) -> str:
    """Normalizza la dicitura dello stage (1/2/3)."""
    if label is None:
        return "stage3"
    
    stage_map = {
        "1": "stage1", "stage1": "stage1",
        "2": "stage2", "stage2": "stage2",
        "3": "stage3", "stage3": "stage3",
    }
    
    key = str(label).strip().lower()
    if key not in stage_map:
        raise ValueError(f"Stage '{label}' non valido. Usa 1, 2 o 3.")
    return stage_map[key]

def resolve_checkpoint_path(config: dict, explicit_path: Optional[str], stage: str) -> Path:
    """Restituisce il percorso del checkpoint."""
    if explicit_path:
        return Path(explicit_path)
    
    out_dir = Path(config["EXP"]["OUT_DIR"]) / config["RUN_NAME"]
    
    # Fallback intelligente se il file specifico dello stage non esiste
    filename = DEFAULT_STAGE_FILES.get(stage, "stage3_best.pth")
    ckpt_path = out_dir / filename
    
    if not ckpt_path.exists():
        # Se cerchiamo stage3 ma non c'è, proviamo best_model.pth generico
        if (out_dir / "best_model.pth").exists():
            print(f"⚠️  {filename} non trovato, uso best_model.pth")
            return out_dir / "best_model.pth"
            
    return ckpt_path

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model(config: dict, device: torch.device, stage: str) -> P2R_ZIP_Model:
    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    
    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": 0.0,
    }

    # Disabilita upsample per visualizzare la griglia P2R reale (Stage 2/3)
    upsample_flag = config["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    if stage in {"stage2", "stage3"} and upsample_flag:
        upsample_flag = False

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=upsample_flag,
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)
    
    return model

def load_checkpoint(model: P2R_ZIP_Model, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    
    print(f"Caricamento pesi da: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location=device)
    
    # Gestione diverse strutture di salvataggio
    if "model" in state:
        state = state["model"]
        
    model.load_state_dict(state, strict=False)
    model.eval()

def get_random_sample(config, index=None):
    data_cfg = config["DATA"]
    val_tf = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    # Tenta di caricare con return_original=True per visualizzazione pulita
    try:
        ds = DatasetClass(
            data_cfg["ROOT"], 
            data_cfg["VAL_SPLIT"], 
            block_size=data_cfg["ZIP_BLOCK_SIZE"], 
            transforms=val_tf, 
            return_original=True
        )
        supports_orig = True
    except TypeError:
        ds = DatasetClass(
            data_cfg["ROOT"], 
            data_cfg["VAL_SPLIT"], 
            block_size=data_cfg["ZIP_BLOCK_SIZE"], 
            transforms=val_tf
        )
        supports_orig = False

    idx = index if index is not None else random.randint(0, len(ds) - 1)
    sample = ds[idx]
    
    # Unpacking flessibile
    if isinstance(sample, dict):
        img = sample["image"]
        pts = sample["points"]
        path = sample.get("img_path", f"sample_{idx}")
        orig = sample.get("original_image") if supports_orig else denormalize_tensor(img)
    else:
        # Fallback tuple
        img = sample[0]
        pts = sample[1]
        path = sample[2] if len(sample) > 2 else f"sample_{idx}"
        orig = sample[3] if supports_orig and len(sample) > 3 else denormalize_tensor(img)
        
    return img, orig, orig.shape[:2], (pts if isinstance(pts, torch.Tensor) else torch.tensor(pts)).cpu(), path

def denormalize_tensor(t):
    """Converte tensore normalizzato in immagine numpy BGR."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

@torch.no_grad()
def get_predictions(model, img_tensor):
    """Esegue inferenza ritornando sia output finale che raw (per simulazione)."""
    img_batch = img_tensor.unsqueeze(0)
    
    # return_intermediates=True è fondamentale per avere 'p2r_density_raw'
    outputs = model(img_batch, return_intermediates=True)
    
    pi_prob = outputs["logit_pi_maps"].softmax(dim=1)[:, 1:].squeeze().cpu().numpy()
    p2r_final = outputs["p2r_density"].squeeze().cpu().numpy()
    
    # Fallback se raw non è disponibile (es. modelli vecchi senza modifica)
    p2r_raw = outputs.get("p2r_density_raw")
    if p2r_raw is not None:
        p2r_raw = p2r_raw.squeeze().cpu().numpy()
    else:
        p2r_raw = p2r_final # Fallback
        
    return {
        "zip_prob": pi_prob,
        "p2r_final": p2r_final,
        "p2r_raw": p2r_raw, 
    }

def draw_patch_grid(img_rgb, patch_h, patch_w, color=(200, 200, 200), thickness=1):
    """Disegna una griglia di patch sull'immagine."""
    img_grid = img_rgb.copy()
    h, w = img_grid.shape[:2]

    # Linee verticali
    for x in range(0, w, patch_w):
        cv2.line(img_grid, (x, 0), (x, h), color, thickness)
    # Linee orizzontali
    for y in range(0, h, patch_h):
        cv2.line(img_grid, (0, y), (w, y), color, thickness)

    return img_grid


def apply_patch_mask(img_rgb, mask, patch_h, patch_w, blackout_rejected=True):
    """
    Applica la maschera patch per patch.
    mask: array 2D con la probabilita/maschera binaria per ogni patch
    Se blackout_rejected=True, le patch rifiutate diventano nere.
    """
    img_masked = img_rgb.copy()
    h, w = img_masked.shape[:2]
    mask_h, mask_w = mask.shape

    for py in range(mask_h):
        for px in range(mask_w):
            # Coordinate pixel della patch
            y1 = int(py * patch_h)
            y2 = int(min((py + 1) * patch_h, h))
            x1 = int(px * patch_w)
            x2 = int(min((px + 1) * patch_w, w))

            if mask[py, px] < 0.5:  # Patch rifiutata
                if blackout_rejected:
                    img_masked[y1:y2, x1:x2] = 0  # Nero
                else:
                    # Semi-trasparente scuro
                    img_masked[y1:y2, x1:x2] = (img_masked[y1:y2, x1:x2] * 0.3).astype(np.uint8)

    return img_masked


def visualize(orig_img, zip_prob, p2r_final, p2r_raw, pred_count, gt_count,
              tau_results, img_path, stage_label, out_path, no_log, block_size=16):

    h, w = orig_img.shape[:2]
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Calcola dimensione patch in pixel
    zip_h, zip_w = zip_prob.shape
    patch_h = h / zip_h
    patch_w = w / zip_w

    # Resize mappe alla dimensione immagine originale
    zip_view = cv2.resize(zip_prob, (w, h), interpolation=cv2.INTER_NEAREST)

    # Plotting - 3 righe
    cols = len(tau_results) + 2
    fig, axes = plt.subplots(3, cols, figsize=(4*cols, 12))
    fig.suptitle(f"Analisi Gating P2R-ZIP: {stage_label.upper()} | {Path(img_path).name}\n"
                 f"Patch Grid: {zip_h}x{zip_w} patches ({patch_h:.0f}x{patch_w:.0f} px each)", fontsize=14)

    # === RIGA 1: Immagine con Griglia e Patch Mascherati ===

    # 1. Immagine originale con griglia
    img_with_grid = draw_patch_grid(orig_rgb, int(patch_h), int(patch_w))
    axes[0,0].imshow(img_with_grid)
    axes[0,0].set_title(f"Original + Patch Grid\nGT: {gt_count:.0f}")
    axes[0,0].axis('off')

    # 2. ZIP Probability sovrapposta all'immagine
    axes[0,1].imshow(orig_rgb)
    im = axes[0,1].imshow(zip_view, vmin=0, vmax=1, cmap="RdYlGn", alpha=0.6)
    axes[0,1].set_title("ZIP Confidence Overlay")
    axes[0,1].axis('off')
    plt.colorbar(im, ax=axes[0,1], fraction=0.046)

    # 3. Immagini con patch rifiutate in nero per ogni tau
    for i, res in enumerate(tau_results):
        ax = axes[0, i+2]
        # Crea maschera binaria
        binary_mask = (res['mask'] > 0.5).astype(np.float32)
        img_masked = apply_patch_mask(orig_rgb, binary_mask, patch_h, patch_w, blackout_rejected=True)
        img_masked = draw_patch_grid(img_masked, int(patch_h), int(patch_w), color=(100, 100, 100))
        ax.imshow(img_masked)
        coverage = binary_mask.mean() * 100
        ax.set_title(f"Patches τ={res['tau']}\nCoverage: {coverage:.1f}%")
        ax.axis('off')

    # === RIGA 2: Maschere Binarie ===

    # 1. Immagine GT con punti (se disponibili)
    axes[1,0].imshow(orig_rgb)
    axes[1,0].set_title(f"Ground Truth: {gt_count:.0f}")
    axes[1,0].axis('off')

    # 2. ZIP Probability Map pura
    im2 = axes[1,1].imshow(zip_prob, vmin=0, vmax=1, cmap="viridis")
    axes[1,1].set_title(f"ZIP Prob Map\n({zip_h}x{zip_w} patches)")
    axes[1,1].axis('off')
    plt.colorbar(im2, ax=axes[1,1], fraction=0.046)

    # 3. Maschere binarie per vari Tau
    for i, res in enumerate(tau_results):
        ax = axes[1, i+2]
        ax.imshow(res['mask'], vmin=0, vmax=1, cmap="gray")
        n_active = res['mask'].sum()
        n_total = res['mask'].size
        ax.set_title(f"Binary Mask τ={res['tau']}\n{int(n_active)}/{n_total} active")
        ax.axis('off')

    # === RIGA 3: Densita e Simulazioni ===

    norm_fn = (lambda x: np.log1p(x)) if not no_log else (lambda x: x)

    # 1. Output Effettivo del Modello
    p2r_view = cv2.resize(p2r_final, (w, h), interpolation=cv2.INTER_LINEAR)
    axes[2,0].imshow(norm_fn(p2r_view), cmap="jet")
    axes[2,0].set_title(f"Model Output\nPred: {pred_count:.1f}")
    axes[2,0].axis('off')

    # 2. P2R RAW (Cosa vede il modello senza maschera)
    raw_view = cv2.resize(p2r_raw, (w, h), interpolation=cv2.INTER_LINEAR)
    axes[2,1].imshow(norm_fn(raw_view), cmap="jet")
    axes[2,1].set_title(f"P2R Raw (No Mask)\nSum: {p2r_raw.sum():.1f}")
    axes[2,1].axis('off')

    # 3. Simulazioni Strict Gating
    for i, res in enumerate(tau_results):
        ax = axes[2, i+2]
        dens_view = cv2.resize(res['dens'], (w, h), interpolation=cv2.INTER_LINEAR)
        ax.imshow(norm_fn(dens_view), cmap="jet")
        err = res['count'] - gt_count
        ax.set_title(f"Gated τ={res['tau']}\nCount: {res['count']:.1f} (err: {err:+.1f})")
        ax.axis('off')

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Grafico salvato in: {out_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza Strict Gating P2R-ZIP (Stage 1-3)")
    parser.add_argument("--checkpoint", type=str, help="Path esplicito al checkpoint")
    parser.add_argument("--stage", type=str, default="stage3", help="Stage (1, 2, 3)")
    parser.add_argument("--index", type=int, help="Indice immagine validazione")
    parser.add_argument("--output", "-o", type=str, help="File di output (es. test.png)")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.2, 0.5, 0.8], help="Soglie di test")
    parser.add_argument("--no-log", action="store_true", help="Disabilita log-scale per densità")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        stage_norm = normalize_stage(args.stage)
    except ValueError as e:
        print(f"❌ Errore: {e}")
        exit(1)
        
    model = get_model(cfg, device, stage_norm)
    ckpt = resolve_checkpoint_path(cfg, args.checkpoint, stage_norm)
    
    try:
        load_checkpoint(model, ckpt, device)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        exit(1)

    img, orig, shape, pts, path = get_random_sample(cfg, args.index)
    maps = get_predictions(model, img.to(device))
    
    # Calcoli di Scala (se P2R è downsampled)
    # cell_area = rapporto tra input e output (es. 8x8 = 64 per downsample=8)
    h_out, w_out = maps["p2r_final"].shape
    h_in, w_in = img.shape[-2:]
    cell_area = (h_in * w_in) / (h_out * w_out)  # Area di ogni cella della griglia

    # Count = sum(density) / cell_area (come in evaluate_stage3.py)
    pred_final = maps["p2r_final"].sum() / cell_area
    gt_count = len(pts)

    # Simulazione Strict Gating (Raw Density * Mask)
    tau_res = []
    for t in args.taus:
        # Maschera binaria dalla probabilità ZIP
        mask = (maps["zip_prob"] > t).astype(np.float32)

        # Allinea maschera alla griglia P2R (se diverse)
        if mask.shape != maps["p2r_raw"].shape:
            mask_resized = cv2.resize(mask, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # STRICT GATING: Moltiplicazione elemento per elemento
        filt_dens = maps["p2r_raw"] * mask_resized
        cnt = filt_dens.sum() / cell_area  # DIVIDE, non moltiplica!

        tau_res.append({"tau": t, "mask": mask, "dens": filt_dens, "count": cnt})

    print(f"\n📸 Immagine: {path}")
    print(f"🔢 GT Count: {gt_count} | Model Pred: {pred_final:.2f}")
    print("-" * 40)
    print("Simulazione Soglie (Strict Gating):")
    for r in tau_res:
        diff = r['count'] - gt_count
        print(f"   τ={r['tau']:.1f} -> Count: {r['count']:.2f} (Err: {diff:+.2f})")

    visualize(orig, maps["zip_prob"], maps["p2r_final"], maps["p2r_raw"], 
              pred_final, gt_count, tau_res, path, stage_norm, args.output, args.no_log)