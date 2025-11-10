# demo_p2r_zip.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.p2r_zip_model import P2RZIPModel 
from utils import load_checkpoint

# ===============================
# PARAMETRI DI BASE
# ===============================
IMG_PATH = "demo/example.jpg"     
CHECKPOINT = "checkpoints/best.pth"
TAU_VALUES = [0.2, 0.4, 0.6, 0.8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# CARICAMENTO MODELLO
# ===============================
model = P2RZIPModel()
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.to(DEVICE).eval()

# ===============================
# CARICAMENTO IMMAGINE
# ===============================
img = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.
img_tensor = img_tensor.to(DEVICE)

# ===============================
# FORWARD PASS
# ===============================
with torch.no_grad():
    pi_map, density_map = model(img_tensor)

# pi_map, density_map ∈ [B,1,H,W]
pi_map = pi_map.squeeze().cpu().numpy()
density_map = density_map.squeeze().cpu().numpy()

# ===============================
# POST-PROCESSING & VISUALIZZAZIONE
# ===============================
results = []
for tau in TAU_VALUES:
    mask = (pi_map < tau).astype(np.float32)
    filtered_density = density_map * mask
    count_estimate = float(filtered_density.sum())
    results.append((tau, mask, filtered_density, count_estimate))

cols = len(results) + 1
fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))

# Colonna iniziale: immagine e π-map
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Input Image")
axes[0, 0].axis("off")

im_pi = axes[1, 0].imshow(pi_map, cmap="inferno")
axes[1, 0].set_title("ZIP π-map (probabilità di background)")
axes[1, 0].axis("off")
fig.colorbar(im_pi, ax=axes[1, 0], fraction=0.046, pad=0.04)

for idx, (tau, mask, filtered, count_estimate) in enumerate(results, start=1):
    ax_mask = axes[0, idx]
    ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax_mask.set_title(f"Maschera (τ = {tau:.1f})")
    ax_mask.axis("off")

    ax_density = axes[1, idx]
    im_density = ax_density.imshow(filtered, cmap="jet")
    ax_density.set_title(f"Densità filtrata\nCount = {count_estimate:.1f}")
    ax_density.axis("off")
    fig.colorbar(im_density, ax=ax_density, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print("Conteggi stimati per ciascuna soglia τ:")
for tau, _, _, count_estimate in results:
    print(f"  τ = {tau:.1f}: {count_estimate:.1f} persone")
