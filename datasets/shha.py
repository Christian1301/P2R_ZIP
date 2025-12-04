# P2R_ZIP/datasets/shha.py
import os
import glob
import numpy as np
import scipy.io as sio
from .base_dataset import BaseCrowdDataset


class SHHA(BaseCrowdDataset):

    def get_image_list(self, split):
        """
        Supporta SPLIT multipli:
        - "train_data"
        - "val_data"
        - "test_data"
        - ["train_data", "val_data"]
        """

        # --- SUPPORTO A LISTE ---
        if isinstance(split, (list, tuple)):
            imgs = []
            for s in split:
                imgs.extend(self.get_image_list(s))
            return sorted(set(imgs))

        # Normalizzazione nomi split
        if split in ["train", "train_data"]:
            split_dir = "train_data"
        elif split in ["val", "val_data"]:
            split_dir = "val_data"
        elif split in ["test", "test_data"]:
            split_dir = "test_data"
        else:
            split_dir = split

        candidates = [
            os.path.join(self.root, split_dir, "images"),
            os.path.join(self.root, split_dir, "img"),
        ]

        for d in candidates:
            imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
            if imgs:
                return imgs

        raise FileNotFoundError(
            f"Nessuna immagine trovata per split '{split}' nei path {candidates}"
        )

    # -------------------------------------------------------------------

    def load_points(self, img_path):
        """
        Supporta:
        - ShanghaiTech originale (.mat)
        - ZIP / P2R (.npy)
        """

        base_dir = os.path.dirname(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        mat_paths = [
            os.path.join(base_dir, "ground_truth", f"GT_{base_name}.mat"),
            os.path.join(base_dir, "ground-truth", f"GT_{base_name}.mat"),
        ]

        # ZIP / P2R
        npy_paths = [
            os.path.join(base_dir, "labels", f"{base_name}.npy"),
            os.path.join(base_dir, "new-anno", f"GT_{base_name}.npy"),
        ]

        # 1. Prova con i file .mat originali
        for mpath in mat_paths:
            if os.path.isfile(mpath):
                mat = sio.loadmat(mpath)
                pts = mat["image_info"][0, 0][0, 0][0]
                return np.array(pts, dtype=np.float32)

        # 2. Prova con file ZIP/P2R
        for npy in npy_paths:
            if os.path.isfile(npy):
                pts = np.load(npy)
                return np.array(pts[:, :2], dtype=np.float32)

        raise FileNotFoundError(
            f"Nessun ground-truth per {img_path}\nCercati:\n"
            + "\n".join(mat_paths + npy_paths)
        )
