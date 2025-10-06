# P2R_ZIP/data/adapters.py
import os, glob, math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def _read_points_file(path):
    """
    Supporta:
      - .txt: ogni riga "x y"
      - .npy: array Nx2
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        return arr.astype(np.float32)
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                pts.append([x, y])
    return np.array(pts, dtype=np.float32)

def _to_tensor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr).permute(2,0,1)
    return t

class CrowdDataset(Dataset):
    """
    Ritorna:
      - img: Tensor [3,H,W]
      - points: Tensor [Ni,2] in pixel (x,y) per immagine (lista nel collate)
      - zip_blocks: Tensor [1,Hb,Wb] dei conteggi per blocco (derivati dai punti)
    """
    def __init__(self, img_root, points_root=None, img_list_file=None,
                 norm_mean=(0.485,0.456,0.406), norm_std=(0.229,0.224,0.225),
                 img_exts=(".jpg",".jpeg",".png"), block_size=32, split="train"):
        self.img_root = img_root
        self.points_root = points_root
        self.block = block_size
        self.norm_mean = torch.tensor(norm_mean).view(3,1,1)
        self.norm_std = torch.tensor(norm_std).view(3,1,1)

        if img_list_file and os.path.isfile(img_list_file):
            with open(img_list_file, "r") as f:
                rels = [l.strip() for l in f if l.strip()]
            self.imgs = [os.path.join(img_root, r) for r in rels]
        else:
            files = []
            for e in img_exts:
                files.extend(glob.glob(os.path.join(img_root, f"**/*{e}"), recursive=True))
            self.imgs = sorted(files)

        assert len(self.imgs) > 0, "Nessuna immagine trovata."

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # points path: sostituisci estensione con .txt/.npy
        points = np.zeros((0,2), dtype=np.float32)
        if self.points_root:
            rel = os.path.relpath(img_path, self.img_root)
            base = os.path.splitext(rel)[0]
            # prova npy poi txt
            p1 = os.path.join(self.points_root, base + ".npy")
            p2 = os.path.join(self.points_root, base + ".txt")
            if os.path.isfile(p1):
                points = _read_points_file(p1)
            elif os.path.isfile(p2):
                points = _read_points_file(p2)

        # build zip blocks from points
        Hb = math.ceil(H / self.block)
        Wb = math.ceil(W / self.block)
        blocks = np.zeros((Hb, Wb), dtype=np.float32)
        for (x, y) in points:
            xb = min(int(x // self.block), Wb - 1)
            yb = min(int(y // self.block), Hb - 1)
            blocks[yb, xb] += 1.0

        timg = _to_tensor(img)
        timg = (timg - self.norm_mean) / self.norm_std
        tpoints = torch.from_numpy(points) if points.size > 0 else torch.zeros((0,2), dtype=torch.float32)
        tblocks = torch.from_numpy(blocks).unsqueeze(0)

        sample = {
            "image": timg,
            "points": tpoints,
            "zip_blocks": tblocks,
            "size_hw": torch.tensor([H, W], dtype=torch.int32),
            "img_path": img_path
        }
        return sample

def collate_crowd(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    blocks = torch.stack([b["zip_blocks"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    sizes = torch.stack([b["size_hw"] for b in batch], dim=0)
    paths = [b["img_path"] for b in batch]
    return {"image": imgs, "zip_blocks": blocks, "points": points, "size_hw": sizes, "img_path": paths}
