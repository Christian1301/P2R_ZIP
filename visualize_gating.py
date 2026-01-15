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

DEFAULT_STAGE_FILES = {
    "stage1": "best_model.pth",
    "stage2": "stage2_best.pth",
    "stage3": "stage3_best.pth",
    "stage4": "stage4_best.pth",
}

DEFAULT_TAU_VALUES: List[float] = [0.2, 0.4, 0.6, 0.8]


def normalize_stage(label: Optional[str]) -> str:
    """Normalizza la dicitura dello stage (accetta anche 1/2/3/4)."""
    if label is None:
        return "stage4"
    stage_map = {
        "1": "stage1",
        "2": "stage2",
        "3": "stage3",
        "4": "stage4",
        "stage1": "stage1",
        "stage2": "stage2",
        "stage3": "stage3",
        "stage4": "stage4",
    }
    key = str(label).strip().lower()
    if key not in stage_map:
        raise ValueError(f"Stage '{label}' non riconosciuto. Usa 1/2/3 o stage1/stage2/stage3.")
    return stage_map[key]


def resolve_checkpoint_path(config: dict, explicit_path: Optional[str], stage: str) -> Path:
    """Restituisce il percorso del checkpoint coerente con config e stage."""
    if explicit_path:
        return Path(explicit_path)
    out_dir = Path(config["EXP"]["OUT_DIR"]) / config["RUN_NAME"]
    filename = DEFAULT_STAGE_FILES.get(stage, DEFAULT_STAGE_FILES["stage3"])
    return out_dir / filename


def load_config(config_path: str) -> dict:
    """Carica il file di configurazione YAML."""
    print(f"Caricamento configurazione da: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model(config: dict, device: torch.device, stage: str) -> P2R_ZIP_Model:
    """Inizializza il modello P2R_ZIP coerente con lo stage selezionato."""
    dataset_name = config["DATASET"]
    bin_cfg = config["BINS_CONFIG"][dataset_name]
    bins, bin_centers = bin_cfg["bins"], bin_cfg["bin_centers"]

    zip_head_cfg = config.get("ZIP_HEAD", {})
    zip_head_kwargs = {
        "lambda_scale": zip_head_cfg.get("LAMBDA_SCALE", 0.5),
        "lambda_max": zip_head_cfg.get("LAMBDA_MAX", 8.0),
        "use_softplus": zip_head_cfg.get("USE_SOFTPLUS", True),
        "lambda_noise_std": zip_head_cfg.get("LAMBDA_NOISE_STD", 0.0),
    }

    upsample_flag = config["MODEL"].get("UPSAMPLE_TO_INPUT", False)
    if stage in {"stage2", "stage3", "stage4"} and upsample_flag:
        print("ℹ️ Forzo UPSAMPLE_TO_INPUT=False per coerenza con l'addestramento Stage 2/3/4.")
        upsample_flag = False

    p2r_head_kwargs = config.get("P2R_HEAD", {})

    model = P2R_ZIP_Model(
        bins=bins,
        bin_centers=bin_centers,
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=upsample_flag,
        zip_head_kwargs=zip_head_kwargs,
        p2r_head_kwargs=p2r_head_kwargs,
    ).to(device)

    return model


def load_checkpoint(model: P2R_ZIP_Model, checkpoint_path: Path, device: torch.device):
    """Carica i pesi del checkpoint nel modello."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato in: {checkpoint_path}")

    print(f"Caricamento checkpoint da: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Attenzione: caricamento non stretto (strict=False). Dettagli: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("Modello caricato in modalità valutazione (eval).")


def get_random_sample(
    config: dict,
    sample_index: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int], torch.Tensor, str]:
    """Carica un campione (random o indicizzato) dal set di validazione."""
    data_cfg = config["DATA"]
    val_tf = build_transforms(data_cfg, is_train=False)

    DatasetClass = get_dataset(config["DATASET"])
    dataset_kwargs = {
        "root": data_cfg["ROOT"],
        "split": data_cfg["VAL_SPLIT"],
        "block_size": data_cfg["ZIP_BLOCK_SIZE"],
        "transforms": val_tf,
    }

    supports_original = True
    try:
        val_set = DatasetClass(**dataset_kwargs, return_original=True)
    except TypeError:
        supports_original = False
        val_set = DatasetClass(**dataset_kwargs)
        print("Avviso: il dataset non supporta `return_original`. Userò la de-normalizzazione per visualizzare l'immagine.")

    if sample_index is None:
        idx = random.randint(0, len(val_set) - 1)
    else:
        if not (0 <= sample_index < len(val_set)):
            raise ValueError(f"Indice {sample_index} fuori dal range [0, {len(val_set) - 1}].")
        idx = sample_index

    sample = val_set[idx]

    if isinstance(sample, dict):
        img_tensor = sample.get("image")
        points_tensor = sample.get("points")
        original_img = sample.get("original_image") if supports_original else None
        img_path = sample.get("img_path", "")
    else:
        img_tensor = sample[0]
        points_tensor = sample[1] if len(sample) > 1 else torch.zeros((0, 2))
        img_path = sample[2] if len(sample) > 2 else ""
        original_img = sample[3] if supports_original and len(sample) > 3 else None

    if img_tensor is None:
        raise RuntimeError("Il dataset non ha restituito il tensore dell'immagine.")

    if original_img is None:
        original_img = denormalize_tensor(img_tensor)
    original_shape = original_img.shape[:2]

    if points_tensor is None:
        points_tensor = torch.zeros((0, 2))
    elif not isinstance(points_tensor, torch.Tensor):
        points_tensor = torch.as_tensor(points_tensor)

    return img_tensor, original_img, original_shape, points_tensor.cpu(), img_path


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Converte un tensore normalizzato in un'immagine BGR (CV2)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def get_predictions(model: P2R_ZIP_Model, img_tensor: torch.Tensor) -> dict:
    """Esegue il forward del modello e restituisce le mappe di interesse."""
    img_batch = img_tensor.unsqueeze(0)
    
    outputs = model(img_batch)

    pi_softmax = outputs["logit_pi_maps"].softmax(dim=1)
    pi_not_zero = pi_softmax[:, 1:]  

    p2r_density = outputs["p2r_density"] 
    
    pred_zip_density = outputs.get("pred_density_zip")

    return {
        "zip_relevance_map": pi_not_zero.squeeze().cpu().numpy(),
        "p2r_density_map": p2r_density.squeeze().cpu().numpy(),
        "zip_density_map": pred_zip_density.squeeze().cpu().numpy() if pred_zip_density is not None else None,
    }


def draw_patch_grid(
    ax, 
    shape: Tuple[int, int], 
    block_size: int, 
    downsample: Tuple[float, float], 
    color: str = 'white', 
    linewidth: float = 0.5, 
    alpha: float = 0.7
):
    """
    Disegna la griglia delle patch sull'asse.
    
    Args:
        ax: Asse matplotlib
        shape: (H, W) dell'immagine visualizzata
        block_size: Dimensione del blocco ZIP in pixel (es. 16)
        downsample: (down_h, down_w) fattori di downsampling
        color: Colore delle linee
        linewidth: Spessore delle linee
        alpha: Trasparenza
    """
    h, w = shape
    down_h, down_w = downsample
    
    # Calcola la dimensione effettiva delle patch sull'immagine
    patch_h = block_size * down_h
    patch_w = block_size * down_w
    
    # Linee orizzontali
    y = 0.0
    while y <= h:
        ax.axhline(y=y, color=color, linewidth=linewidth, alpha=alpha)
        y += patch_h
    
    # Linee verticali
    x = 0.0
    while x <= w:
        ax.axvline(x=x, color=color, linewidth=linewidth, alpha=alpha)
        x += patch_w


def visualize_patch_classification(
    original_img: np.ndarray,
    zip_map: np.ndarray,
    original_shape: Tuple[int, int],
    block_size: int,
    downsample: Tuple[float, float],
    tau: float = 0.5,
) -> np.ndarray:
    """
    Crea una visualizzazione con le patch colorate in base alla classificazione ZIP.
    
    Verde = patch con persone (π > tau)
    Rosso = patch vuote (π <= tau)
    
    Args:
        original_img: Immagine originale BGR
        zip_map: Mappa di rilevanza ZIP (H_zip, W_zip)
        original_shape: (H, W) dell'immagine originale
        block_size: Dimensione del blocco ZIP
        downsample: (down_h, down_w) fattori di downsampling
        tau: Soglia per la classificazione
        
    Returns:
        Immagine con overlay colorato delle patch
    """
    h, w = original_shape
    down_h, down_w = downsample
    
    # Dimensione patch sull'immagine originale
    patch_h = block_size * down_h
    patch_w = block_size * down_w
    
    # Crea overlay
    overlay = original_img.copy()
    h_zip, w_zip = zip_map.shape
    
    for i in range(h_zip):
        for j in range(w_zip):
            y1 = int(i * patch_h)
            y2 = min(int((i + 1) * patch_h), h)
            x1 = int(j * patch_w)
            x2 = min(int((j + 1) * patch_w), w)
            
            # Salta patch invalide
            if y2 <= y1 or x2 <= x1:
                continue
            
            relevance = zip_map[i, j]
            
            if relevance > tau:
                # Verde per patch con persone
                color = (0, 255, 0)
                alpha = 0.3
            else:
                # Rosso per patch vuote
                color = (0, 0, 255)  # BGR
                alpha = 0.2
            
            # Applica overlay colorato
            roi = overlay[y1:y2, x1:x2]
            colored = np.full_like(roi, color)
            cv2.addWeighted(colored, alpha, roi, 1 - alpha, 0, roi)
            
            # Disegna bordo
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    return overlay


def visualize_patch_details(
    zip_map: np.ndarray,
    p2r_map: np.ndarray,
    block_size: int,
    downsample: Tuple[float, float],
    tau: float = 0.5,
) -> dict:
    """
    Calcola statistiche dettagliate sulle patch.
    
    Returns:
        Dizionario con statistiche delle patch
    """
    h_zip, w_zip = zip_map.shape
    down_h, down_w = downsample
    cell_area = down_h * down_w
    
    # Conta patch per categoria
    n_full = int((zip_map > tau).sum())
    n_empty = int((zip_map <= tau).sum())
    n_total = h_zip * w_zip
    
    # Calcola densità media per patch piene vs vuote
    mask_full = zip_map > tau
    mask_empty = ~mask_full
    
    # Resize p2r_map alla dimensione di zip_map se necessario
    if p2r_map.shape != zip_map.shape:
        p2r_resized = cv2.resize(p2r_map, (w_zip, h_zip), interpolation=cv2.INTER_LINEAR)
    else:
        p2r_resized = p2r_map
    
    density_full = p2r_resized[mask_full].mean() / cell_area if n_full > 0 else 0
    density_empty = p2r_resized[mask_empty].mean() / cell_area if n_empty > 0 else 0
    
    # Trova patch con massima densità
    max_density_idx = np.unravel_index(np.argmax(p2r_resized), p2r_resized.shape)
    max_density_val = p2r_resized[max_density_idx] / cell_area
    
    return {
        "n_full": n_full,
        "n_empty": n_empty,
        "n_total": n_total,
        "pct_full": 100 * n_full / n_total if n_total > 0 else 0,
        "pct_empty": 100 * n_empty / n_total if n_total > 0 else 0,
        "avg_density_full": density_full,
        "avg_density_empty": density_empty,
        "max_density_patch": max_density_idx,
        "max_density_value": max_density_val,
        "grid_shape": (h_zip, w_zip),
    }


def visualize_results(
    original_img: np.ndarray,
    zip_map: np.ndarray,
    p2r_map: np.ndarray,
    original_shape: Tuple[int, int],
    pred_count: float,
    gt_count: float,
    downsample: Tuple[float, float],
    img_path: str,
    stage_label: str,
    output_path: Optional[str] = None,
    tau_results: Optional[List[dict]] = None,
    show_log: bool = True,
    block_size: int = 16,
    show_grid: bool = True,
):
    """Visualizza le mappe principali e le versioni filtrate per diverse soglie τ."""

    tau_results = tau_results or []

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_shape
    zip_map_resized = cv2.resize(zip_map, (w, h), interpolation=cv2.INTER_LINEAR)
    p2r_map_resized = cv2.resize(p2r_map, (w, h), interpolation=cv2.INTER_LINEAR)

    p2r_linear_norm = cv2.normalize(p2r_map_resized, None, 0, 1, cv2.NORM_MINMAX)
    p2r_log_norm = cv2.normalize(np.log1p(p2r_map_resized), None, 0, 1, cv2.NORM_MINMAX)

    # Calcola statistiche patch
    patch_stats = visualize_patch_details(zip_map, p2r_map, block_size, downsample, tau=0.5)

    # Layout: 3 colonne base + colonne tau
    cols = max(3, len(tau_results) + 3)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 9))
    down_h, down_w = downsample
    
    fig.suptitle(
        f"Visualizzazione Gating P2R-ZIP — {stage_label.upper()}\n"
        f"Griglia: {patch_stats['grid_shape'][0]}×{patch_stats['grid_shape'][1]} patch "
        f"({block_size}px × down {down_h:.1f}×{down_w:.1f})",
        fontsize=14
    )

    # ===================== RIGA 1 =====================
    
    # [0,0] Immagine originale
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Immagine Originale")
    axes[0, 0].axis("off")

    # [0,1] Immagine con griglia patch
    axes[0, 1].imshow(original_rgb)
    if show_grid:
        draw_patch_grid(
            axes[0, 1], original_shape, block_size, downsample, 
            color='yellow', linewidth=0.8, alpha=0.9
        )
    h_zip, w_zip = zip_map.shape
    axes[0, 1].set_title(f"Griglia Patch\n{h_zip}×{w_zip} = {h_zip * w_zip} blocchi")
    axes[0, 1].axis("off")

    # [0,2] Maschera ZIP con griglia
    im_zip = axes[0, 2].imshow(zip_map_resized, cmap="viridis", vmin=0, vmax=1)
    if show_grid:
        draw_patch_grid(
            axes[0, 2], original_shape, block_size, downsample,
            color='white', linewidth=0.5, alpha=0.5
        )
    axes[0, 2].set_title("Maschera Rilevanza ZIP\n(π probabilità non-vuoto)")
    axes[0, 2].axis("off")
    fig.colorbar(im_zip, ax=axes[0, 2], orientation="vertical", fraction=0.046, pad=0.04)

    # [0,3+] Maschere per vari tau
    for idx, res in enumerate(tau_results, start=3):
        if idx >= cols:
            break
        axes[0, idx].imshow(res["mask_original"], cmap="gray", vmin=0, vmax=1)
        if show_grid:
            draw_patch_grid(
                axes[0, idx], original_shape, block_size, downsample,
                color='cyan', linewidth=0.5, alpha=0.5
            )
        n_active = int((zip_map >= res["tau"]).sum())
        axes[0, idx].set_title(f"Maschera τ={res['tau']:.1f}\n{n_active} patch attive")
        axes[0, idx].axis("off")

    # ===================== RIGA 2 =====================
    
    # [1,0] Densità P2R lineare
    im_p2r = axes[1, 0].imshow(p2r_linear_norm, cmap="jet")
    axes[1, 0].set_title(f"Densità P2R (lineare)\nPred={pred_count:.1f} | GT={gt_count:.0f}")
    axes[1, 0].axis("off")
    fig.colorbar(im_p2r, ax=axes[1, 0], orientation="vertical", fraction=0.046, pad=0.04)

    # [1,1] Classificazione patch colorata
    patch_viz = visualize_patch_classification(
        original_img, zip_map, original_shape, block_size, downsample, tau=0.5
    )
    axes[1, 1].imshow(cv2.cvtColor(patch_viz, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"Classificazione Patch (τ=0.5)\n"
        f"🟢 Piene: {patch_stats['n_full']} ({patch_stats['pct_full']:.1f}%) | "
        f"🔴 Vuote: {patch_stats['n_empty']} ({patch_stats['pct_empty']:.1f}%)"
    )
    axes[1, 1].axis("off")

    # [1,2] Densità log o overlay
    if show_log:
        im_log = axes[1, 2].imshow(p2r_log_norm, cmap="jet")
        axes[1, 2].set_title(
            f"Densità P2R (log1p)\n"
            f"Avg piena: {patch_stats['avg_density_full']:.2f} | "
            f"Avg vuota: {patch_stats['avg_density_empty']:.2f}"
        )
        axes[1, 2].axis("off")
        fig.colorbar(im_log, ax=axes[1, 2], orientation="vertical", fraction=0.046, pad=0.04)
    else:
        axes[1, 2].imshow(original_rgb)
        axes[1, 2].imshow(zip_map_resized, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[1, 2].set_title("Originale + ZIP Overlay")
        axes[1, 2].axis("off")

    # [1,3+] Densità filtrate per tau
    for idx, res in enumerate(tau_results, start=3):
        if idx >= cols:
            break
        filt_resized = cv2.resize(res["filtered_density"], (w, h), interpolation=cv2.INTER_LINEAR)
        filt_norm = cv2.normalize(filt_resized, None, 0, 1, cv2.NORM_MINMAX)
        im_tau = axes[1, idx].imshow(filt_norm, cmap="jet")
        axes[1, idx].set_title(f"Densità filtrata τ={res['tau']:.1f}\nCount={res['count']:.1f}")
        axes[1, idx].axis("off")
        fig.colorbar(im_tau, ax=axes[1, idx], orientation="vertical", fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Gestione output
    backend = matplotlib.get_backend().lower()
    headless = backend in {"agg", "pdf", "ps", "svg", "cairo"} or not os.environ.get("DISPLAY")

    if output_path is None and not headless:
        plt.show()
    else:
        if output_path is None:
            output_path = "visualizations/visualize_gating.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figura salvata in: {output_path}")
        print(f"➡️  Campione: {img_path} | Pred={pred_count:.2f} | GT={gt_count:.2f}")
    plt.close(fig)


def print_patch_statistics(patch_stats: dict, tau_results: List[dict]):
    """Stampa statistiche dettagliate delle patch."""
    print("\n" + "=" * 60)
    print("📊 STATISTICHE PATCH")
    print("=" * 60)
    print(f"   Griglia:        {patch_stats['grid_shape'][0]} × {patch_stats['grid_shape'][1]} = {patch_stats['n_total']} patch")
    print(f"   Patch piene:    {patch_stats['n_full']} ({patch_stats['pct_full']:.1f}%)")
    print(f"   Patch vuote:    {patch_stats['n_empty']} ({patch_stats['pct_empty']:.1f}%)")
    print(f"   Densità media (piene):  {patch_stats['avg_density_full']:.3f} persone/patch")
    print(f"   Densità media (vuote):  {patch_stats['avg_density_empty']:.3f} persone/patch")
    print(f"   Patch max densità:      {patch_stats['max_density_patch']} → {patch_stats['max_density_value']:.2f} persone")
    
    if tau_results:
        print("\n   Conteggi per soglia τ:")
        for res in tau_results:
            n_active = int((res["tau"] <= 1.0) and True)  # placeholder
            print(f"      τ = {res['tau']:.1f}: {res['count']:.1f} persone")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

    parser = argparse.ArgumentParser(description="Visualizza le mappe di gating P2R-ZIP con griglia patch.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Percorso del checkpoint da caricare. Se omesso usa RUN_NAME/OUT_DIR da config.")
    parser.add_argument("--stage", type=str, default="stage2",
                        help="Stage di riferimento (1/2/3/4 oppure stage1/stage2/stage3/stage4).")
    parser.add_argument("--index", type=int, default=None,
                        help="Indice del campione di validazione da visualizzare (default: random).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed per la selezione casuale del campione.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Percorso del file immagine da salvare (default: auto in headless).")
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAU_VALUES,
                        help="Elenco di soglie τ da applicare alla maschera ZIP (default: 0.2 0.4 0.6 0.8).")
    parser.add_argument("--no-log", action="store_true",
                        help="Disattiva la visualizzazione logaritmica della densità P2R.")
    parser.add_argument("--no-grid", action="store_true",
                        help="Disattiva la visualizzazione della griglia delle patch.")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Override della dimensione del blocco ZIP (default: da config).")
    args = parser.parse_args()
    
    # Validazione stage
    try:
        stage_label = normalize_stage(args.stage)
    except ValueError as exc:
        print(f"ERRORE: {exc}")
        exit(1)

    # Setup
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(CONFIG_PATH)
    
    # Block size da args o config
    block_size = args.block_size or config["DATA"].get("ZIP_BLOCK_SIZE", 16)
    
    # Carica modello
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint, stage_label)
    model = get_model(config, device, stage_label)
    try:
        load_checkpoint(model, checkpoint_path, device)
    except FileNotFoundError as exc:
        print(f"ERRORE: {exc}")
        exit(1)

    # Carica campione
    img_tensor, original_img, original_shape, points_tensor, img_path = get_random_sample(config, args.index)
    
    # Predizioni
    maps = get_predictions(model, img_tensor.to(device))

    # Calcola downsampling e conteggi
    p2r_map = maps["p2r_density_map"]
    H_in, W_in = img_tensor.shape[-2:]
    H_out, W_out = p2r_map.shape[-2:]
    
    if H_out == H_in and W_out == W_in:
        down_h = down_w = 1.0
        pred_count = float(p2r_map.sum())
    else:
        down_h = H_in / max(H_out, 1)
        down_w = W_in / max(W_out, 1)
        pred_count = float(p2r_map.sum() / (down_h * down_w))
    
    gt_count = float(points_tensor.shape[0])

    # Conteggio ZIP (opzionale)
    zip_pred_count = None
    zip_density_map = maps.get("zip_density_map")
    if zip_density_map is not None:
        zip_pred_count = float(zip_density_map.sum())

    # Calcola risultati per diverse soglie tau
    tau_results: List[dict] = []
    zip_map = maps["zip_relevance_map"]
    tau_values = [float(t) for t in (args.taus or [])]
    
    for tau in tau_values:
        mask_zip = (zip_map >= tau).astype(np.float32)
        mask_original = cv2.resize(mask_zip, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_density = cv2.resize(mask_zip, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
        filtered_density = p2r_map * mask_density
        filtered_count = float(filtered_density.sum())
        if not (H_out == H_in and W_out == W_in):
            filtered_count /= (down_h * down_w)
        tau_results.append({
            "tau": tau,
            "mask_original": mask_original,
            "filtered_density": filtered_density,
            "count": filtered_count,
        })

    # Stampa info
    print(
        f"\nConteggi P2R: pred={pred_count:.2f}, gt={gt_count:.2f}, down=({down_h:.2f},{down_w:.2f}), "
        f"H_outxW_out={H_out}x{W_out}, H_inxW_in={H_in}x{W_in}"
    )
    if zip_pred_count is not None:
        print(f"Conteggi ZIP (per blocco {block_size}px): pred={zip_pred_count:.2f}")
    if gt_count > 0:
        diff = pred_count - gt_count
        print(f"Delta pred-gt = {diff:+.2f} (rel {diff / max(gt_count, 1e-6):+.2%})")

    # Statistiche patch
    patch_stats = visualize_patch_details(zip_map, p2r_map, block_size, (down_h, down_w), tau=0.5)
    print_patch_statistics(patch_stats, tau_results)

    # Visualizzazione
    visualize_results(
        original_img,
        zip_map,
        p2r_map,
        original_shape,
        pred_count,
        gt_count,
        (down_h, down_w),
        img_path,
        stage_label,
        output_path=args.output,
        tau_results=tau_results,
        show_log=not args.no_log,
        block_size=block_size,
        show_grid=not args.no_grid,
    )