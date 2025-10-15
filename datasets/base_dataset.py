# P2R_ZIP/datasets/base_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class BaseCrowdDataset(Dataset):
    def __init__(self, root, split, transforms=None, block_size=16):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.block_size = block_size
        self.image_list = self.get_image_list(split)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img_path = self.image_list[i]
        pts = self.load_points(img_path)
        img = self.load_image(img_path)
        den = self.points_to_density(pts, img.shape[1:])

        if self.transforms:
            # Assicurati che le trasformazioni restituiscano img, pts, den
            img, pts, den = self.transforms(img, pts, den)

        return {
            "image": img,
            "points": pts,
            "density": den,
            "img_path": img_path,
        }

    def get_image_list(self, split):
        raise NotImplementedError

    def load_points(self, img_path):
        raise NotImplementedError

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def points_to_density(self, points, shape):
        h, w = shape
        den = torch.zeros((1, h, w), dtype=torch.float32)
        for x, y in points:
            x, y = int(x), int(y)
            if 0 <= y < h and 0 <= x < w:
                den[0, y, x] = 1.0
        return den