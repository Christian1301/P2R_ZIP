# P2R_ZIP/datasets/base_dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class BaseCrowdDataset(Dataset):
    """
    Classe base per dataset di crowd counting.
    Ogni sottoclasse deve implementare:
        - load_points(img_path): restituisce np.ndarray Nx2
        - get_image_list(split): restituisce lista path immagini
    """
    def __init__(self, root, split="train", block_size=32, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.root = root
        self.split = split
        self.block = block_size
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std  = torch.tensor(std).view(3,1,1)
        self.img_list = self.get_image_list(split)
        assert len(self.img_list) > 0, f"Nessuna immagine trovata in {root}/{split}"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        pts = self.load_points(img_path)
        pts = np.array(pts, dtype=np.float32)
        Hb, Wb = int(np.ceil(H / self.block)), int(np.ceil(W / self.block))
        blocks = np.zeros((Hb, Wb), dtype=np.float32)
        for (x, y) in pts:
            xb = min(int(x // self.block), Wb - 1)
            yb = min(int(y // self.block), Hb - 1)
            blocks[yb, xb] += 1

        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        timg = torch.from_numpy(arr).permute(2,0,1)
        timg = (timg - self.mean) / self.std

        return {
            "image": timg,
            "points": torch.from_numpy(pts),
            "zip_blocks": torch.from_numpy(blocks).unsqueeze(0),
            "img_path": img_path
        }
