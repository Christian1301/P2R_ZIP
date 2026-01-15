# Esegui questo nel tuo ambiente
import torch
from datasets import get_dataset
from datasets.transforms import build_transforms
import yaml

with open('config_newloss.yaml') as f:
    cfg = yaml.safe_load(f)

DatasetClass = get_dataset(cfg["DATASET"])
val_tf = build_transforms(cfg['DATA'], is_train=False)

ds = DatasetClass(
    root=cfg["DATA"]["ROOT"],
    split=cfg["DATA"]["VAL_SPLIT"],
    block_size=16,
    transforms=val_tf
)

# Prendi un campione
img, density, points = ds[0]

print(f"Immagine shape: {img.shape}")  # [C, H, W]
print(f"Punti shape: {points.shape}")
print(f"Primi 5 punti:\n{points[:5]}")
print(f"Min/Max colonna 0: {points[:,0].min():.1f} - {points[:,0].max():.1f}")
print(f"Min/Max colonna 1: {points[:,1].min():.1f} - {points[:,1].max():.1f}")