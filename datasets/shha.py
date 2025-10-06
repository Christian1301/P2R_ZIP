# P2R_ZIP/datasets/shha.py
import os, glob, scipy.io as sio, numpy as np
from .base_dataset import BaseCrowdDataset

class SHHA(BaseCrowdDataset):
    def get_image_list(self, split):
        img_dir = os.path.join(self.root, f"part_A_{split}")
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        return imgs

    def load_points(self, img_path):
        mat_path = img_path.replace(".jpg", ".mat").replace("images", "ground_truth")
        mat = sio.loadmat(mat_path)
        pts = mat["image_info"][0,0][0,0][0]
        return np.array(pts, dtype=np.float32)
