# P2R_ZIP/datasets/shha.py
import os, glob, numpy as np, scipy.io as sio
from .base_dataset import BaseCrowdDataset

class SHHA(BaseCrowdDataset):
    def get_image_list(self, split):
        """
        Gestisce i path del dataset per:
        - Dataset originale (.mat)
        - Versione ZIP o P2R (.npy)
        """
        if split in ["train", "train_data"]:
            split_dir = "train_data"
        elif split in ["val", "test", "test_data"]:
            split_dir = "test_data"
        else:
            split_dir = split

        candidates = [
            os.path.join(self.root, split_dir, "images"),
            os.path.join(self.root, split_dir, "img"),
            os.path.join(self.root, split_dir, "train", "images"),
        ]
        for d in candidates:
            imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
            if len(imgs) > 0:
                return imgs
        raise FileNotFoundError(f"Nessuna immagine trovata in {candidates}")

    def load_points(self, img_path):
        """
        Carica i punti associati a un'immagine.
        Supporta .mat (ShanghaiTech originale), .npy (ZIP, P2R).
        """
        base_dir = os.path.dirname(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]

        mat_path = os.path.join(base_dir, "ground_truth", f"GT_{base_name}.mat")
        mat_path2 = os.path.join(base_dir, "ground-truth", f"GT_{base_name}.mat")
        npy_path_zip = os.path.join(base_dir, "labels", f"{base_name}.npy")
        npy_path_p2r = os.path.join(base_dir, "new-anno", f"GT_{base_name}.npy")

        if os.path.isfile(mat_path) or os.path.isfile(mat_path2):
            mpath = mat_path if os.path.isfile(mat_path) else mat_path2
            mat = sio.loadmat(mpath)
            pts = mat["image_info"][0, 0][0, 0][0]
            return np.array(pts, dtype=np.float32)

        elif os.path.isfile(npy_path_zip):
            pts = np.load(npy_path_zip)
            return np.array(pts[:, :2], dtype=np.float32)
        elif os.path.isfile(npy_path_p2r):
            pts = np.load(npy_path_p2r)
            return np.array(pts[:, :2], dtype=np.float32)

        else:
            raise FileNotFoundError(
                f"Nessun file ground truth trovato per {img_path}\n"
                f"Controllati:\n  {mat_path}\n  {mat_path2}\n  {npy_path_zip}\n  {npy_path_p2r}"
            )
