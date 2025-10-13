# demo_p2r_zip.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.p2r_zip_model import P2RZIPModel  # il file principale della pipeline
from utils import load_checkpoint  # se usi la tua funzione di caricamento

# ===============================
# PARAMETRI DI BASE
# ===============================
IMG_PATH = "demo/example.jpg"     # immagine di test
CHECKPOINT = "checkpoints/best.pth"  # pesi addestrati
TAU = 0.6                         # soglia per mascheramento
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
# POST-PROCESSING
# ===============================
mask = (pi_map < TAU).astype(np.float32)
filtered_density = density_map * mask
count_estimate = filtered_density.sum()

# ===============================
# VISUALIZZAZIONE
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Input Image")

axes[0, 1].imshow(pi_map, cmap='inferno')
axes[0, 1].set_title(f"ZIP π-map (probabilità di background)")

axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title(f"Maschera binaria (τ = {TAU})")

axes[1, 1].imshow(filtered_density, cmap='jet')
axes[1, 1].set_title(f"Densità filtrata / Conteggio stimato = {count_estimate:.1f}")

plt.tight_layout()
plt.show()

print(f"Conteggio stimato: {count_estimate:.1f} persone")
