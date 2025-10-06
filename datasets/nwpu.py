# P2R_ZIP/datasets/nwpu.py
import os, glob, numpy as np
from .base_dataset import BaseCrowdDataset

class NWPU(BaseCrowdDataset):
    def get_image_list(self, split):
        img_dir = os.path.join(self.root, split, "images")
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        return imgs

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
