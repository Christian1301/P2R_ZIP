"""Stage 3 evaluation script matching the recovery training pipeline."""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn
from train_stage2_p2r import calibrate_density_scale


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


@torch.no_grad()
def evaluate_stage4(model, dataloader, device, default_down):
	model.eval()

	abs_errors, sq_errors = [], []
	sparse_errors, medium_errors, dense_errors = [], [], []
	density_means, density_maxima = [], []
	pi_activity = []
	total_pred, total_gt = 0.0, 0.0

	print("\n===== VALUTAZIONE STAGE 4 (Recovery) =====")

	for images, _, points in tqdm(dataloader, desc="[Eval Stage 4]"):
		images = images.to(device)
		points_gpu = [p.to(device) if p is not None else None for p in points]

		outputs = model(images)
		pred_density = outputs["p2r_density"]

		_, _, h_in, w_in = images.shape
		pred_density, down_tuple, _ = canonicalize_p2r_grid(
			pred_density, (h_in, w_in), default_down, warn_tag="stage4_eval"
		)

		down_h, down_w = down_tuple
		cell_area = down_h * down_w
		pred_count = torch.sum(pred_density, dim=(1, 2, 3)) / cell_area

		gt_counts = torch.tensor(
			[len(p) if p is not None else 0 for p in points_gpu],
			device=device,
			dtype=torch.float32,
		)

		batch_errors = torch.abs(pred_count - gt_counts)
		abs_errors.extend(batch_errors.cpu().tolist())
		sq_errors.extend(((pred_count - gt_counts) ** 2).cpu().tolist())

		total_pred += pred_count.sum().item()
		total_gt += gt_counts.sum().item()

		dens_cpu = pred_density.detach().cpu()
		density_means.append(dens_cpu.mean().item())
		density_maxima.append(dens_cpu.max().item())

		pi_logits = outputs.get("logit_pi_maps")
		if pi_logits is not None:
			pi_probs = torch.sigmoid(pi_logits[:, 1:2])
			active_ratio = (pi_probs > 0.5).float().mean().item() * 100.0
			pi_activity.append(active_ratio)

		for err, gt_val in zip(batch_errors.cpu().tolist(), gt_counts.cpu().tolist()):
			if gt_val <= 100:
				sparse_errors.append(err)
			elif gt_val <= 500:
				medium_errors.append(err)
			else:
				dense_errors.append(err)

	if not abs_errors:
		print("⚠️ Val loader vuoto, nessuna metrica calcolata")
		return {}

	mae = float(np.mean(abs_errors))
	rmse = float(np.sqrt(np.mean(sq_errors)))
	bias = total_pred / total_gt if total_gt > 0 else 0.0

	def _fmt_bucket(name, values):
		if not values:
			return f"   {name}: n/a"
		return f"   {name}: MAE={np.mean(values):.2f} ({len(values)} imgs)"

	print("\n📊 Risultati Stage 4")
	print(f"   MAE:  {mae:.2f}")
	print(f"   RMSE: {rmse:.2f}")
	print(f"   Bias: {bias:.3f}")
	print(_fmt_bucket("Sparse  (0-100)", sparse_errors))
	print(_fmt_bucket("Medium (100-500)", medium_errors))
	print(_fmt_bucket("Dense  (500+)", dense_errors))

	if pi_activity:
		print(f"   π-head active ratio medio: {np.mean(pi_activity):.1f}%")

	print("\n🔎 Density diagnostics")
	print(f"   mean(μ): {np.mean(density_means):.4f}")
	print(f"   max(μ):  {np.mean(density_maxima):.4f}")

	return {
		"mae": mae,
		"rmse": rmse,
		"bias": bias,
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

	data_cfg = cfg["DATA"]
	val_transforms = build_transforms(data_cfg, is_train=False)
	DatasetClass = get_dataset(cfg["DATASET"])
	val_dataset = DatasetClass(
		root=data_cfg["ROOT"],
		split=data_cfg["VAL_SPLIT"],
		block_size=data_cfg["ZIP_BLOCK_SIZE"],
		transforms=val_transforms,
	)

	optim_stage4 = cfg.get("OPTIM_STAGE4", {})
	num_workers = int(optim_stage4.get("NUM_WORKERS", 4))

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
		pi_thresh=cfg["MODEL"]["ZIP_PI_THRESH"],
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

	evaluate_stage4(model, val_loader, device, default_down)


if __name__ == "__main__":
	main()
