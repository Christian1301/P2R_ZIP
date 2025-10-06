# P2R_ZIP/datasets/ucf_qnrf.py
import os
import glob
import scipy.io as sio
import numpy as np
from .base_dataset import BaseCrowdDataset

class UCF_QNRF(BaseCrowdDataset):
    def get_image_list(self, split):
        img_dir = os.path.join(self.root, split, "images")
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        return imgs

    def load_points(self, img_path):
        gt_path = img_path.replace("/images/", "/gt/").replace(".jpg", "_ann.mat")
        pts = []
        if os.path.exists(gt_path):
            mat = sio.loadmat(gt_path)
            pts = mat["annPoints"]
        return np.array(pts, dtype=np.float32)
