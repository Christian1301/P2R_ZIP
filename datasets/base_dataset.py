# P2R_ZIP/datasets/base_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
import cv2
from PIL import Image 

class BaseCrowdDataset(Dataset):
    def __init__(self, root, split, transforms=None, block_size=16):
        self.root = root
        self.split = split
        self.transforms = transforms 
        self.block_size = block_size
        self.image_list = self.get_image_list(split)
        if not self.image_list:
             raise FileNotFoundError(f"Nessuna immagine trovata per split '{split}' in root '{root}'")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img_path = self.image_list[i]
        pts = self.load_points(img_path) 
        img = self.load_image(img_path) 
        
        den_np = self.points_to_density_numpy(pts, img.size[::-1]) # Passa (h, w)

        img_tensor, pts_transformed, den_tensor = None, pts, None # Default
        if self.transforms:
            img_transformed, pts_transformed, den_np_transformed = self.transforms(img, pts, den_np)
            
            if not isinstance(img_transformed, torch.Tensor):
                 raise TypeError("Le trasformazioni devono restituire un Tensor per l'immagine")
            img_tensor = img_transformed
            
            if den_np_transformed is not None:
                if isinstance(den_np_transformed, torch.Tensor):
                    den_tensor = den_np_transformed
                else: 
                    den_tensor = torch.from_numpy(den_np_transformed).unsqueeze(0)
            else: 
                 h, w = img_tensor.shape[1:]
                 den_tensor = self.points_to_density_tensor(pts_transformed, (h, w))
                 
            pts_tensor = torch.from_numpy(pts_transformed) if pts_transformed is not None and len(pts_transformed) > 0 else torch.zeros((0,2), dtype=torch.float32)

        else:
             img_tensor = TF.to_tensor(img)
             den_tensor = torch.from_numpy(den_np).unsqueeze(0)
             pts_tensor = torch.from_numpy(pts) if pts is not None and len(pts) > 0 else torch.zeros((0,2), dtype=torch.float32)

        return {
            "image": img_tensor,
            "points": pts_tensor, 
            "density": den_tensor,
            "img_path": img_path,
        }

    def get_image_list(self, split):
        raise NotImplementedError("Implementa in subclass")

    def load_points(self, img_path):
        raise NotImplementedError("Implementa in subclass") 

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img

    def points_to_density_numpy(self, points, shape):
        h, w = shape
        den = np.zeros((h, w), dtype=np.float32)
        if points is not None:
             for x, y in points:
                x, y = int(x), int(y)
                if 0 <= y < h and 0 <= x < w:
                    den[y, x] = 1.0 
        return den
        
    def points_to_density_tensor(self, points, shape):
        h, w = shape
        den = torch.zeros((1, h, w), dtype=torch.float32)
        if points is not None:
             for x, y in points: 
                x, y = int(x), int(y)
                if 0 <= y < h and 0 <= x < w:
                    den[0, y, x] = 1.0
        return den