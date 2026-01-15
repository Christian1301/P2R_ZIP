import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Classe base robusta.
    Genera la GT Density (Dot Map) automaticamente DOPO le trasformazioni,
    garantendo che le dimensioni immagine/densità siano sempre allineate.
    """
    def __init__(self, root, split, block_size=16, transforms=None):
        self.root = root
        self.split = split
        self.block_size = block_size
        self.transforms = transforms
        self.img_paths = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        # 1. Carica Immagine
        img = Image.open(img_path).convert('RGB')
        
        # 2. Carica Punti
        pts = self.load_points(img_path) 
        
        # 3. Gestione Density Pre-Transform
        # Se esiste un .npy lo carichiamo, ma potremmo scartarlo dopo se le dimensioni non tornano
        den_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth')
        den_np = None
        if os.path.exists(den_path):
            try:
                den_np = np.load(den_path).astype(np.float32, copy=False)
            except:
                pass 

        # 4. Applicazione Trasformazioni
        if self.transforms:
            img_tensor, pts_transformed, den_np_transformed = self.transforms(img, pts, den_np)
        else:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            pts_transformed = torch.from_numpy(pts).float()
            den_np_transformed = den_np

        # 5. Fix Formato Punti (Tensor)
        if isinstance(pts_transformed, torch.Tensor):
            pts_tensor = pts_transformed
        elif isinstance(pts_transformed, np.ndarray):
            pts_tensor = torch.from_numpy(pts_transformed).float()
        else:
            pts_tensor = torch.tensor(pts_transformed, dtype=torch.float32) if len(pts_transformed) > 0 else torch.zeros((0, 2), dtype=torch.float32)

        # 6. Generazione GT Density (IL FIX)
        # Assicuriamoci che la densità corrisponda ESATTAMENTE alle dimensioni dell'immagine trasformata.
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        
        # Controlla se la densità uscita dalle trasformazioni è valida e ha le dimensioni giuste
        use_transformed_den = False
        if den_np_transformed is not None:
            # Recupera H, W della densità
            if isinstance(den_np_transformed, np.ndarray):
                dh, dw = den_np_transformed.shape[-2:] # gestisce (H,W) o (C,H,W)
            else:
                dh, dw = den_np_transformed.shape[-2:]
            
            if dh == h and dw == w:
                use_transformed_den = True
        
        if use_transformed_den:
             # Usiamo quella trasformata
             if isinstance(den_np_transformed, np.ndarray):
                 gt_density = torch.from_numpy(den_np_transformed)
             else:
                 gt_density = den_np_transformed
             if gt_density.dim() == 2: gt_density = gt_density.unsqueeze(0)
        
        else:
             # RIGENERAZIONE "ON-THE-FLY" (Dot Map)
             # Questo salva il training: crea una mappa zeri delle dimensioni giuste e mette un 1 dove ci sono i punti.
             gt_density = torch.zeros((1, h, w), dtype=torch.float32)
             
             if len(pts_tensor) > 0:
                 p = pts_tensor.long()
                 # Sicurezza: rimuovi punti che per arrotondamento escono di 1 pixel
                 p[:, 0] = p[:, 0].clamp(0, w - 1)
                 p[:, 1] = p[:, 1].clamp(0, h - 1)
                 
                 # Metodo veloce per disegnare i punti: Scatter Add
                 # Calcola indice lineare (y * w + x)
                 indices = p[:, 1] * w + p[:, 0]
                 values = torch.ones(indices.size(0), dtype=torch.float32)
                 
                 # Somma 1.0 nella mappa (gestisce anche punti sovrapposti)
                 gt_density.view(-1).scatter_add_(0, indices, values)

        return img_tensor, gt_density, pts_tensor

    def load_points(self, img_path):
        return np.array([], dtype=np.float32)

BaseCrowdDataset = BaseDataset