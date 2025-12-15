"""
Stage 3 evaluation script with π-masking support.

Usa la soglia ZIP_PI_THRESH dal config per applicare il masking.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import (
    init_seeds,
    canonicalize_p2r_grid,
    collate_fn,
    calibrate_density_scale_v2 as calibrate_density_scale,
)


def _load_checkpoint(model, output_dir, device):
    candidates = [
        os.path.join(output_dir, "stage4_best.pth"),
        os.path.join(output_dir, "stage3_best.pth"),
        os.path.join(output_dir, "stage2_best.pth"),
    ]

    for ckpt_path in candidates:
        if not os.path.isfile(ckpt_path):
            continue
        print(f"🔄 Caricamento pesi da: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        missing = model.load_state_dict(state_dict, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            print(
                f"⚠️ load_state_dict mismatch → missing={missing.missing_keys}, "
                f"unexpected={missing.unexpected_keys}"
            )
        return True

    print("❌ Nessun checkpoint trovato per Stage 4/3/2")
    return False


def compute_count_raw(pred_density, down_h, down_w):
    """Conteggio RAW (senza maschera)."""
    cell_area = down_h * down_w
    return torch.sum(pred_density, dim=(1, 2, 3)) / cell_area


def compute_count_masked(pred_density, pi_probs, down_h, down_w, threshold=0.5):
    """Conteggio MASKED (con maschera π)."""
    cell_area = down_h * down_w
    if pi_probs.shape[-2:] != pred_density.shape[-2:]:
        pi_probs = F.interpolate(
            pi_probs, 
            size=pred_density.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    mask = (pi_probs > threshold).float()
    masked_density = pred_density * mask
    return torch.sum(masked_density, dim=(1, 2, 3)) / cell_area, mask.mean().item() * 100


@torch.no_grad()
def evaluate_stage3(model, dataloader, device, default_down, pi_threshold=0.5):
    """
    Valutazione Stage 3 con π-masking.
    
    Args:
        model: modello P2R-ZIP
        dataloader: validation DataLoader
        device: device
        default_down: downsample factor
        pi_threshold: soglia per π-masking (da config)
    """
    model.eval()

    # Metriche RAW
    abs_errors_raw, sq_errors_raw = [], []
    # Metriche MASKED
    abs_errors_masked, sq_errors_masked = [], []
    
    sparse_errors, medium_errors, dense_errors = [], [], []
    density_means, density_maxima = [], []
    pi_activity = []
    total_pred_raw, total_pred_masked, total_gt = 0.0, 0.0, 0.0
    coverages = []

    print(f"\n===== VALUTAZIONE STAGE 3 (π-threshold={pi_threshold}) =====")

    for images, _, points in tqdm(dataloader, desc="[Eval Stage 3]"):
        images = images.to(device)
        points_gpu = [p.to(device) if p is not None else None for p in points]

        outputs = model(images)
        pred_density = outputs["p2r_density"]

        _, _, h_in, w_in = images.shape
        pred_density, down_tuple, _ = canonicalize_p2r_grid(
            pred_density, (h_in, w_in), default_down, warn_tag="stage3_eval"
        )

        down_h, down_w = down_tuple
        
        # Count RAW
        pred_count_raw = compute_count_raw(pred_density, down_h, down_w)
        
        # Get π probs
        pi_probs = outputs.get("pi_probs")
        if pi_probs is None:
            pi_logits = outputs.get("logit_pi_maps")
            if pi_logits is not None:
                pi_probs = torch.softmax(pi_logits, dim=1)[:, 1:2]
            else:
                # Fallback: no masking
                pi_probs = torch.ones_like(pred_density)
        
        # Count MASKED
        pred_count_masked, coverage = compute_count_masked(
            pred_density, pi_probs, down_h, down_w, threshold=pi_threshold
        )
        coverages.append(coverage)

        gt_counts = torch.tensor(
            [len(p) if p is not None else 0 for p in points_gpu],
            device=device,
            dtype=torch.float32,
        )

        # Errori RAW
        batch_errors_raw = torch.abs(pred_count_raw - gt_counts)
        abs_errors_raw.extend(batch_errors_raw.cpu().tolist())
        sq_errors_raw.extend(((pred_count_raw - gt_counts) ** 2).cpu().tolist())
        
        # Errori MASKED
        batch_errors_masked = torch.abs(pred_count_masked - gt_counts)
        abs_errors_masked.extend(batch_errors_masked.cpu().tolist())
        sq_errors_masked.extend(((pred_count_masked - gt_counts) ** 2).cpu().tolist())

        total_pred_raw += pred_count_raw.sum().item()
        total_pred_masked += pred_count_masked.sum().item()
        total_gt += gt_counts.sum().item()

        dens_cpu = pred_density.detach().cpu()
        density_means.append(dens_cpu.mean().item())
        density_maxima.append(dens_cpu.max().item())

        # π activity
        if pi_probs is not None:
            active_ratio = (pi_probs > 0.5).float().mean().item() * 100.0
            pi_activity.append(active_ratio)

        # Per-density analysis
        for err_raw, err_masked, gt_val in zip(
            batch_errors_raw.cpu().tolist(), 
            batch_errors_masked.cpu().tolist(),
            gt_counts.cpu().tolist()
        ):
            if gt_val <= 100:
                sparse_errors.append((err_raw, err_masked, gt_val))
            elif gt_val <= 500:
                medium_errors.append((err_raw, err_masked, gt_val))
            else:
                dense_errors.append((err_raw, err_masked, gt_val))

    if not abs_errors_raw:
        print("⚠️ Val loader vuoto, nessuna metrica calcolata")
        return {}

    # Calcola metriche
    mae_raw = float(np.mean(abs_errors_raw))
    rmse_raw = float(np.sqrt(np.mean(sq_errors_raw)))
    bias_raw = total_pred_raw / total_gt if total_gt > 0 else 0.0
    
    mae_masked = float(np.mean(abs_errors_masked))
    rmse_masked = float(np.sqrt(np.mean(sq_errors_masked)))
    bias_masked = total_pred_masked / total_gt if total_gt > 0 else 0.0
    
    avg_coverage = np.mean(coverages)
    improvement = mae_raw - mae_masked

    # Report
    print("\n" + "=" * 65)
    print("📊 RISULTATI STAGE 3")
    print("=" * 65)
    
    print(f"\n{'Metrica':<15} {'RAW':<12} {'MASKED (τ={:.2f})':<18} {'Δ':<10}".format(pi_threshold))
    print("-" * 55)
    print(f"{'MAE':<15} {mae_raw:<12.2f} {mae_masked:<18.2f} {improvement:+.2f}")
    print(f"{'RMSE':<15} {rmse_raw:<12.2f} {rmse_masked:<18.2f} {rmse_raw - rmse_masked:+.2f}")
    print(f"{'Bias':<15} {bias_raw:<12.3f} {bias_masked:<18.3f} {bias_raw - bias_masked:+.3f}")
    
    print(f"\n📈 Coverage medio: {avg_coverage:.1f}%")
    
    if pi_activity:
        print(f"   π-head active ratio: {np.mean(pi_activity):.1f}%")

    # Per-density breakdown
    def _fmt_bucket(name, errors):
        if not errors:
            return f"   {name}: n/a"
        raw_errs = [e[0] for e in errors]
        masked_errs = [e[1] for e in errors]
        imp = np.mean(raw_errs) - np.mean(masked_errs)
        status = "✅" if imp > 0 else "⚠️"
        return f"   {name}: RAW={np.mean(raw_errs):.1f} → MASKED={np.mean(masked_errs):.1f} ({imp:+.1f}) {status} [{len(errors)} imgs]"

    print("\n📊 Per densità:")
    print(_fmt_bucket("Sparse  (0-100)", sparse_errors))
    print(_fmt_bucket("Medium (100-500)", medium_errors))
    print(_fmt_bucket("Dense  (500+)", dense_errors))

    # Risultato finale
    print("\n" + "=" * 65)
    print("🏁 RISULTATO FINALE")
    print("=" * 65)
    print(f"""
   ┌─────────────────────────────────────────────┐
   │  MAE RAW:       {mae_raw:6.2f}                      │
   │  MAE MASKED:    {mae_masked:6.2f}  (τ = {pi_threshold})            │
   │  ──────────────────────────────────────     │
   │  MIGLIORAMENTO: {improvement:+6.2f}                      │
   │  BIAS:          {bias_masked:.3f}                       │
   │  COVERAGE:      {avg_coverage:5.1f}%                      │
   └─────────────────────────────────────────────┘
""")
    
    if mae_masked < 65:
        print("   🎉 TARGET RAGGIUNTO! MAE < 65")
    elif mae_masked < 68:
        print("   ✅ Ottimo risultato! MAE < 68")

    print("=" * 65)

    return {
        "mae_raw": mae_raw,
        "mae_masked": mae_masked,
        "mae": mae_masked,  # Metrica principale = MASKED
        "rmse_raw": rmse_raw,
        "rmse_masked": rmse_masked,
        "bias_raw": bias_raw,
        "bias_masked": bias_masked,
        "coverage": avg_coverage,
        "improvement": improvement,
        "pi_threshold": pi_threshold,
        "pi_active": float(np.mean(pi_activity)) if pi_activity else None,
    }


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["DEVICE"])
    init_seeds(cfg["SEED"])

    # *** LEGGE SOGLIA DAL CONFIG ***
    pi_threshold = cfg["MODEL"].get("ZIP_PI_THRESH", 0.5)
    print(f"✅ Usando π-threshold = {pi_threshold} (da config)")

    data_cfg = cfg["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(cfg["DATASET"])
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms,
    )

    optim_cfg = cfg.get("OPTIM_STAGE4", cfg.get("OPTIM_STAGE3", {}))
    num_workers = int(optim_cfg.get("NUM_WORKERS", 4))

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dataset_name = cfg["DATASET"]
    bin_cfg = cfg["BINS_CONFIG"][dataset_name]
    zip_head_kwargs = {
        "lambda_scale": cfg["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": cfg["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": cfg["ZIP_HEAD"].get("USE_SOFTPLUS", True),
        "lambda_noise_std": 0.0,
    }

    model = P2R_ZIP_Model(
        bins=bin_cfg["bins"],
        bin_centers=bin_cfg["bin_centers"],
        backbone_name=cfg["MODEL"]["BACKBONE"],
        pi_thresh=pi_threshold,
        gate=cfg["MODEL"]["GATE"],
        upsample_to_input=cfg["MODEL"].get("UPSAMPLE_TO_INPUT", False),
        zip_head_kwargs=zip_head_kwargs,
    ).to(device)

    out_dir = os.path.join(cfg["EXP"]["OUT_DIR"], cfg["RUN_NAME"])
    if not _load_checkpoint(model, out_dir, device):
        return

    p2r_cfg = cfg.get("P2R_LOSS", {})
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)

    print("\n🔧 Calibrazione log_scale pre-eval...")
    calibrate_density_scale(
        model,
        val_loader,
        device,
        default_down,
        max_batches=15,
        clamp_range=p2r_cfg.get("LOG_SCALE_CLAMP"),
        max_adjust=0.5,
    )

    # *** USA LA SOGLIA DAL CONFIG ***
    evaluate_stage3(model, val_loader, device, default_down, pi_threshold=pi_threshold)


if __name__ == "__main__":
    main()