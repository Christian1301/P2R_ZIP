"""
Inference engine for P2R-ZIP Crowd Counting Demo.
Extracted and adapted from visualize_gating.py
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from io import BytesIO
import base64

# ── Imports from project ─────────────────────────────────────────────
import sys
# Add parent directory to path so we can import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.p2r_zip_model import P2R_ZIP_Model
from train_utils import canonicalize_p2r_grid


# ── Default bins configs ──────────────────────────────────────────────

DEFAULT_BINS_CONFIG = {
    'shha': {
        'bins': [[0,0],[1,3],[4,6],[7,10],[11,15],[16,22],[23,32],[33,9999]],
        'bin_centers': [0.0,2.0,5.0,8.5,13.0,19.0,27.5,45.0],
    },
    'shhb': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.16],
    },
    'ucf': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
    'jhu': {
        'bins': [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,23],[24,26],[27,29],[30,33],[34,9999]],
        'bin_centers': [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.43,13.43,15.44,17.44,19.43,21.83,24.85,27.87,31.24,38.86],
    },
}

DATASET_ALIAS = {
    'shha': 'shha', 'shhb': 'shhb', 'ucf': 'ucf', 'jhu': 'jhu',
    'qnrf': 'ucf',
}


# ── Model registry ───────────────────────────────────────────────────

MODELS_REGISTRY = {
    "ShanghaiTech Part A": {
        "config": "config.yaml",
        "checkpoint": "weights/shha.pth",
        "dataset_key": "shha",
        "description": "Dense urban scenes, 300-3000 people",
        "alpha": 0.25,
    },
    "ShanghaiTech Part B": {
        "config": "config_shhb.yaml",
        "checkpoint": "weights/shhb.pth",
        "dataset_key": "shhb",
        "description": "Sparse street scenes, 10-600 people",
        "alpha": 0.15,
    },
    "UCF-QNRF": {
        "config": "config_qnrf.yaml",
        "checkpoint": "weights/ucf.pth",
        "dataset_key": "ucf",
        "description": "Extremely dense crowds, up to 12,000+ people",
        "alpha": 0.15,
    },
    "JHU-Crowd++": {
        "config": "config_jhu.yaml",
        "checkpoint": "weights/jhu.pth",
        "dataset_key": "jhu",
        "description": "Diverse scenes and weather, 0-25,000 people",
        "alpha": 0.15,
    },
}


class CrowdCountingEngine:
    """Engine that manages model loading and inference."""

    def __init__(self, project_root: str, device: str = "auto"):
        self.project_root = project_root
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._cache = {}  # model_name -> (model, scale_val, config)
        print(f"🖥️  CrowdCountingEngine initialized on {self.device}")

    def get_available_models(self) -> dict:
        """Returns dict of model_name -> info for models with existing checkpoints."""
        available = {}
        for name, info in MODELS_REGISTRY.items():
            ckpt_path = os.path.join(self.project_root, info["checkpoint"])
            if os.path.isfile(ckpt_path):
                available[name] = {
                    "description": info["description"],
                    "checkpoint_exists": True,
                }
            else:
                available[name] = {
                    "description": info["description"],
                    "checkpoint_exists": False,
                    "missing_path": ckpt_path,
                }
        return available

    def _load_model(self, model_name: str):
        """Load model + checkpoint into cache."""
        # Se viene richiesto ShanghaiTech Part B, usa sempre Part A
        if model_name == "ShanghaiTech Part B":
            print("⚠️  'ShanghaiTech Part B' non disponibile, uso 'ShanghaiTech Part A'.")
            model_name = "ShanghaiTech Part A"

        if model_name in self._cache:
            return self._cache[model_name]

        info = MODELS_REGISTRY[model_name]
        config_path = os.path.join(self.project_root, info["config"])
        ckpt_path = os.path.join(self.project_root, info["checkpoint"])

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        dataset_key = info["dataset_key"]
        bins_config = config.get('BINS_CONFIG', {})
        if dataset_key in bins_config:
            bin_cfg = bins_config[dataset_key]
        else:
            bin_cfg = DEFAULT_BINS_CONFIG[dataset_key]

        zip_head_cfg = config.get('ZIP_HEAD', {})
        model_cfg = config.get('MODEL', {})

        model = P2R_ZIP_Model(
            backbone_name=model_cfg.get('BACKBONE', 'vgg16_bn'),
            pi_thresh=model_cfg.get('ZIP_PI_THRESH', 0.5),
            bins=bin_cfg['bins'],
            bin_centers=bin_cfg['bin_centers'],
            zip_head_kwargs={
                'lambda_scale': zip_head_cfg.get('LAMBDA_SCALE', 1.2),
                'lambda_max': zip_head_cfg.get('LAMBDA_MAX', 8.0),
                'use_softplus': zip_head_cfg.get('USE_SOFTPLUS', True),
            },
        ).to(self.device)

        # Load checkpoint
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        scale_val = 1.0
        if 'scale_comp' in state:
            try:
                log_scale = state['scale_comp']['log_scale']
                scale_val = torch.exp(log_scale).item()
            except Exception:
                pass

        model_state = state.get('model', state)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        self._cache[model_name] = (model, scale_val, config)
        print(f"✅ Loaded model: {model_name} (scale={scale_val:.3f})")
        return model, scale_val, config

    @torch.no_grad()
    def predict(self, pil_image: Image.Image, model_name: str) -> dict:
        """
        Run inference on a PIL image.

        Returns dict with:
          - count_raw, count_soft
          - images: dict of base64-encoded PNG visualizations
          - stats: dict of numeric stats
        """
        model, scale_val, config = self._load_model(model_name)
        info = MODELS_REGISTRY[model_name]
        alpha = info["alpha"]
        tau = config.get('MODEL', {}).get('ZIP_PI_THRESH', 0.3)
        default_down = 8

        # ── Preprocess ────────────────────────────────────────────
        img_rgb = pil_image.convert("RGB")
        img_tensor = TF.to_tensor(img_rgb)
        img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Pad to multiple of 32
        _, _, H, W = img_tensor.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))

        # ── Forward ───────────────────────────────────────────────
        outputs = model(img_tensor)
        raw_density = outputs['p2r_density']
        logits_pi = outputs['logit_pi_maps']
        pi_probs = torch.softmax(logits_pi, dim=1)[:, 1:2, :, :]
        lambda_maps = outputs.get('lambda_maps')

        # Canonicalize
        _, _, H_in, W_in = img_tensor.shape
        raw_density, down_tuple, _ = canonicalize_p2r_grid(raw_density, (H_in, W_in), default_down)
        cell_area = down_tuple[0] * down_tuple[1]

        # ── Counts ────────────────────────────────────────────────
        count_raw = (raw_density.sum() * scale_val / cell_area).item()

        # Soft fusion
        if pi_probs.shape[-2:] != raw_density.shape[-2:]:
            pi_aligned = F.interpolate(pi_probs, size=raw_density.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pi_aligned = pi_probs

        soft_weights = (1 - alpha) + alpha * pi_aligned
        density_soft = raw_density * scale_val * soft_weights
        count_soft = (density_soft.sum() / cell_area).item()

        # Hard mask
        mask_hard = (pi_aligned > tau).float()
        density_hard = raw_density * scale_val * mask_hard
        count_hard = (density_hard.sum() / cell_area).item()

        # ── Generate visualizations ──────────────────────────────
        # Convert maps to numpy at original image resolution
        def to_heatmap(tensor_map, size_hw):
            m = tensor_map.squeeze().cpu().numpy()
            m = cv2.resize(m, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
            return m

        h_orig, w_orig = H, W  # original (before padding)
        pi_map = to_heatmap(pi_probs, (h_orig, w_orig))
        raw_map = to_heatmap(raw_density * scale_val, (h_orig, w_orig))
        soft_map = to_heatmap(density_soft, (h_orig, w_orig))

        # Original image as numpy
        img_np = np.array(img_rgb)

        images = {
            "pi_map": self._render_heatmap(pi_map, "π Occupancy Map", cmap='RdYlGn', vmin=0, vmax=1, overlay_img=img_np),
            "density_raw": self._render_heatmap(raw_map, "Raw Density", cmap='inferno', overlay_img=img_np),
            "density_soft": self._render_heatmap(soft_map, "Soft Fusion Density", cmap='inferno', overlay_img=img_np),
        }

        stats = {
            "count_raw": round(count_raw, 1),
            "count_soft": round(count_soft, 1),
            "pi_mean": round(pi_probs.mean().item(), 3),
            "pi_coverage": round((pi_probs > tau).float().mean().item() * 100, 1),
            "scale_compensation": round(scale_val, 3),
            "alpha": alpha,
            "model": model_name,
        }

        return {
            "count": round(count_soft, 1),
            "images": images,
            "stats": stats,
        }

    def _render_heatmap(self, data: np.ndarray, title: str, cmap='inferno',
                        vmin=None, vmax=None, overlay_img=None) -> str:
        """Render a heatmap as base64-encoded PNG."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if overlay_img is not None:
            # Resize overlay to match data
            h, w = data.shape[:2]
            overlay = cv2.resize(overlay_img, (w, h))
            ax.imshow(overlay, alpha=0.4)

        im = ax.imshow(data, cmap=cmap, alpha=0.7 if overlay_img is not None else 1.0,
                        vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')