"""
P2R-ZIP Crowd Counting Demo - Flask Backend
Serves a Claude-like chat interface for crowd counting inference.
"""

import os
import sys
import json
import time
import uuid
import glob
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from io import BytesIO
import base64
import numpy as np

# Setup path
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"[DEBUG] DEMO_DIR    = {DEMO_DIR}")
print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DEBUG] index.html exists = {os.path.isfile(os.path.join(DEMO_DIR, 'index.html'))}")

from inference import CrowdCountingEngine, MODELS_REGISTRY

# ── Flask App ─────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=DEMO_DIR)

# Initialize engine
engine = CrowdCountingEngine(project_root=PROJECT_ROOT)


@app.route('/')
def index():
    return send_from_directory(DEMO_DIR, 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(DEMO_DIR, 'static'), filename)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return available models."""
    available = engine.get_available_models()
    models = []
    for name, info in available.items():
        models.append({
            "name": name,
            "description": MODELS_REGISTRY[name]["description"],
            "available": info["checkpoint_exists"],
        })
    return jsonify({"models": models})


# ── Ground Truth Lookup ───────────────────────────────────────────────

# Common GT directory patterns per dataset
GT_SEARCH_PATHS = {
    "ShanghaiTech Part A": [
        "data/ShanghaiTech/part_A/test_data/ground-truth",
        "data/SHA/test/gt",
        "data/shha/test/gt",
    ],
    "ShanghaiTech Part B": [
        "data/ShanghaiTech/part_B/test_data/ground-truth",
        "data/SHB/test/gt",
        "data/shhb/test/gt",
    ],
    "UCF-QNRF": [
        "data/UCF-QNRF_ECCV18/test",
        "data/QNRF/test/gt",
        "data/qnrf/test/gt",
    ],
    "JHU-Crowd++": [
        "data/jhu_crowd_v2.0/test/gt",
        "data/JHU/test/gt",
        "data/jhu/test/gt",
    ],
}


def find_ground_truth(filename, model_name):
    """
    Try to find the ground truth count for a given image filename.
    Looks for .mat or .npy files in standard dataset GT directories.
    Returns the count as a float, or None if not found.
    """
    if not filename:
        return None

    stem = os.path.splitext(filename)[0]  # e.g. "IMG_1" from "IMG_1.jpg"

    search_dirs = GT_SEARCH_PATHS.get(model_name, [])

    for rel_dir in search_dirs:
        gt_dir = os.path.join(PROJECT_ROOT, rel_dir)
        if not os.path.isdir(gt_dir):
            continue

        # Try .mat (ShanghaiTech format: GT_IMG_X.mat)
        for pattern in [f"GT_{stem}.mat", f"{stem}.mat", f"{stem}_ann.mat"]:
            mat_path = os.path.join(gt_dir, pattern)
            if os.path.isfile(mat_path):
                try:
                    from scipy.io import loadmat
                    mat = loadmat(mat_path)
                    # ShanghaiTech stores points in 'image_info' or 'annPoints'
                    if 'image_info' in mat:
                        count = mat['image_info'][0][0][0][0][1][0][0]
                    elif 'annPoints' in mat:
                        count = len(mat['annPoints'])
                    elif 'gt' in mat:
                        count = len(mat['gt'])
                    else:
                        # Try first array-like value
                        for v in mat.values():
                            if hasattr(v, '__len__') and not isinstance(v, str):
                                count = len(v)
                                break
                        else:
                            continue
                    return float(count)
                except Exception:
                    continue

        # Try .npy
        for pattern in [f"{stem}.npy", f"GT_{stem}.npy"]:
            npy_path = os.path.join(gt_dir, pattern)
            if os.path.isfile(npy_path):
                try:
                    data = np.load(npy_path, allow_pickle=True)
                    if data.ndim == 2:
                        count = len(data)  # array of points
                    else:
                        count = float(data.sum())  # density map
                    return float(count)
                except Exception:
                    continue

    return None


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run inference on uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        model_name = request.form.get('model', 'ShanghaiTech Part A')

        if model_name not in MODELS_REGISTRY:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        # Load image
        img_bytes = file.read()
        pil_image = Image.open(BytesIO(img_bytes)).convert('RGB')

        # Get original image as base64 for display
        buf = BytesIO()
        pil_image.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        original_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # Run inference
        t0 = time.time()
        result = engine.predict(pil_image, model_name)
        inference_time = round(time.time() - t0, 2)

        # Try to find ground truth: from form input first, then from file lookup
        gt_from_form = request.form.get('ground_truth')
        if gt_from_form is not None:
            try:
                ground_truth = float(gt_from_form)
            except ValueError:
                ground_truth = None
        else:
            ground_truth = find_ground_truth(file.filename, model_name)

        return jsonify({
            "success": True,
            "count": result["count"],
            "stats": result["stats"],
            "images": result["images"],
            "original_image": original_b64,
            "inference_time": inference_time,
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat(),
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"Model files missing: {str(e)}"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🧑‍🤝‍🧑  P2R-ZIP Crowd Counting Demo")
    print("=" * 60)

    available = engine.get_available_models()
    for name, info in available.items():
        status = "✅" if info["checkpoint_exists"] else "❌"
        print(f"  {status} {name}: {MODELS_REGISTRY[name]['description']}")

    print(f"\n  Device: {engine.device}")
    print(f"  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)