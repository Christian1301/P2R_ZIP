# P2R_ZIP/datasets/base_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
import cv2
from PIL import Image # Usiamo PIL per compatibilità con torchvision.transforms

class BaseCrowdDataset(Dataset):
    # Modifica: Accetta transforms nell'init
    def __init__(self, root, split, transforms=None, block_size=16):
        self.root = root
        self.split = split
        self.transforms = transforms # Salva le trasformazioni
        self.block_size = block_size
        self.image_list = self.get_image_list(split)
        if not self.image_list:
             raise FileNotFoundError(f"Nessuna immagine trovata per split '{split}' in root '{root}'")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img_path = self.image_list[i]
        pts = self.load_points(img_path) # Carica come numpy array
        img = self.load_image(img_path) # Carica come PIL Image
        
        # Genera mappa densità come numpy array *prima* delle trasformazioni
        den_np = self.points_to_density_numpy(pts, img.size[::-1]) # Passa (h, w)

        # Applica le trasformazioni
        img_tensor, pts_transformed, den_tensor = None, pts, None # Default
        if self.transforms:
            # Passa PIL image, numpy points, numpy density
            img_transformed, pts_transformed, den_np_transformed = self.transforms(img, pts, den_np)
            
            # Le transforms ora dovrebbero restituire Tensori normalizzati ecc.
            # Se ToTensor è l'ultima transform geometrica, img_transformed è già un Tensor
            # Se Normalize è l'ultima, è già normalizzato
            
            # Assicuriamoci che l'output sia nel formato corretto
            if not isinstance(img_transformed, torch.Tensor):
                 raise TypeError("Le trasformazioni devono restituire un Tensor per l'immagine")
            img_tensor = img_transformed
            
            # Se le transforms hanno restituito la densità (es. dopo crop/resize)
            if den_np_transformed is not None:
                if isinstance(den_np_transformed, torch.Tensor):
                    den_tensor = den_np_transformed
                else: # Se ancora numpy, converti
                    den_tensor = torch.from_numpy(den_np_transformed).unsqueeze(0)
            else: # Altrimenti, rigenera la densità dai punti trasformati
                 h, w = img_tensor.shape[1:]
                 den_tensor = self.points_to_density_tensor(pts_transformed, (h, w))
                 
            # Converte i punti trasformati (ancora numpy) in Tensore
            pts_tensor = torch.from_numpy(pts_transformed) if pts_transformed is not None and len(pts_transformed) > 0 else torch.zeros((0,2), dtype=torch.float32)

        else:
             # Se non ci sono transforms, converti manualmente
             img_tensor = TF.to_tensor(img)
             # Normalizza manualmente (se non fatto dalle transforms)
             # mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
             # std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
             # img_tensor = (img_tensor - mean) / std
             den_tensor = torch.from_numpy(den_np).unsqueeze(0)
             pts_tensor = torch.from_numpy(pts) if pts is not None and len(pts) > 0 else torch.zeros((0,2), dtype=torch.float32)


        return {
            "image": img_tensor,
            "points": pts_tensor, # Ora è un tensore
            "density": den_tensor,
            "img_path": img_path,
        }

    def get_image_list(self, split):
        raise NotImplementedError("Implementa in subclass")

    def load_points(self, img_path):
        raise NotImplementedError("Implementa in subclass") # Restituisce numpy array

    def load_image(self, img_path):
        # Modifica: Carica come PIL Image
        img = Image.open(img_path).convert("RGB")
        return img

    def points_to_density_numpy(self, points, shape):
        # Genera densità come numpy array
        h, w = shape
        den = np.zeros((h, w), dtype=np.float32)
        if points is not None:
             for x, y in points:
                x, y = int(x), int(y)
                if 0 <= y < h and 0 <= x < w:
                    den[y, x] = 1.0 # O usa un kernel Gaussiano se preferisci
        return den
        
    def points_to_density_tensor(self, points, shape):
        # Genera densità come tensore (utile se le transforms non la gestiscono)
        h, w = shape
        den = torch.zeros((1, h, w), dtype=torch.float32)
        if points is not None:
             for x, y in points: # points qui è ancora numpy
                x, y = int(x), int(y)
                if 0 <= y < h and 0 <= x < w:
                    den[0, y, x] = 1.0
        return den