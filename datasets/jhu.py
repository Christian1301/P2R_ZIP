# P2R_ZIP/datasets/jhu.py
import os
import glob
import numpy as np
from .base_dataset import BaseDataset

class JHU_Crowd(BaseDataset):
    def load_data(self):
        """Carica i path delle immagini JHU."""
        img_dir = os.path.join(self.root, self.split, "images")
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        if len(imgs) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata in {img_dir}")
        
        print(f"JHU Dataset {self.split}: Trovate {len(imgs)} immagini.")
        self.img_paths = imgs

    def load_points(self, img_path):
        gt_path = img_path.replace("/images/", "/gt/").replace(".jpg", ".txt")
        pts = []
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        pts.append([x, y])
        return np.array(pts, dtype=np.float32)