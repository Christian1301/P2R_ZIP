# P2R_ZIP/datasets/base_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image


class BaseCrowdDataset(Dataset):
    """
    Dataset base per crowd counting.

    FIX IMPORTANTI:
    ----------------
    ✔ Ordine parametri __init__(root, split, block_size, transforms)
    ✔ Supporto corretto a transforms=Compose
    ✔ Verifica robusta sulle trasformazioni
    ✔ Ritorno sempre consistente: image Tensor, density Tensor, points Tensor
    """

    def __init__(self, root, split, block_size=16, transforms=None):
        self.root = root
        self.split = split
        self.block_size = int(block_size)
        self.transforms = transforms

        # Carica lista immagini dallo split (stringa o lista di split)
        self.image_list = self.get_image_list(split)
        if not self.image_list:
            raise FileNotFoundError(
                f"Nessuna immagine trovata per split '{split}' in root: {root}"
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]

        # Carica immagine e punti
        pts = self.load_points(img_path)  # numpy N×2
        img = self.load_image(img_path)  # PIL Image

        # GT densità iniziale prima delle trasformazioni
        den_np = self.points_to_density_numpy(pts, img.size[::-1])  # shape (H, W)

        # Se NON ci sono trasformazioni → ritorna il formato base
        if self.transforms is None:
            img_tensor = TF.to_tensor(img)
            den_tensor = torch.from_numpy(den_np).unsqueeze(0)
            pts_tensor = (
                torch.from_numpy(pts.astype(np.float32))
                if pts is not None and len(pts) > 0
                else torch.zeros((0, 2), dtype=torch.float32)
            )
            return {
                "image": img_tensor,
                "points": pts_tensor,
                "density": den_tensor,
                "img_path": img_path,
            }

        # --- Applica trasformazioni ---
        img_t, pts_t, den_np_t = self.transforms(img, pts, den_np)

        if not isinstance(img_t, torch.Tensor):
            raise TypeError(
                f"ERRORE: transforms ha restituito img non-Tensor: {type(img_t)}\n"
                f"Valore transforms={self.transforms}"
            )

        # Converte punti
        if pts_t is not None and len(pts_t) > 0:
            pts_tensor = torch.from_numpy(pts_t.astype(np.float32))
        else:
            pts_tensor = torch.zeros((0, 2), dtype=torch.float32)

        # Converte densità
        if den_np_t is None:
            # Ricostruisci densità dai punti trasformati
            h, w = img_t.shape[1:]
            den_tensor = self.points_to_density_tensor(pts_tensor.numpy(), (h, w))
        elif isinstance(den_np_t, torch.Tensor):
            den_tensor = den_np_t
        else:
            den_tensor = torch.from_numpy(den_np_t.astype(np.float32)).unsqueeze(0)

        return {
            "image": img_t,
            "points": pts_tensor,
            "density": den_tensor,
            "img_path": img_path,
        }

    # ================================================================
    # Funzioni per la densità
    # ================================================================
    def points_to_density_numpy(self, points, shape):
        """Versione numpy: crea una density map con 1 nei punti GT."""
        h, w = shape
        den = np.zeros((h, w), dtype=np.float32)

        if points is not None:
            for x, y in points:
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    den[y, x] = 1.0
        return den

    def points_to_density_tensor(self, points, shape):
        """Versione torch della density map."""
        h, w = shape
        den = torch.zeros((1, h, w), dtype=torch.float32)

        if points is not None:
            for x, y in points:
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    den[0, y, x] = 1.0
        return den

    # ================================================================
    # Da sovrascrivere nelle subclass
    # ================================================================
    def get_image_list(self, split):
        raise NotImplementedError

    def load_points(self, img_path):
        raise NotImplementedError

    def load_image(self, img_path):
        return Image.open(img_path).convert("RGB")
