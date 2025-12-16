# P2R_ZIP/datasets/ucf_qnrf.py
# VERSIONE AGGIORNATA per struttura flat (immagini e txt nella stessa cartella)
#
# Struttura attesa:
# ROOT/
# ├── train/scene01/img_0001.jpg, img_0001.txt, ...
# └── test/scene01/img_0001.jpg, img_0001.txt, ...
#
# Supporta anche struttura originale con Train/Test e .mat files

import os
import glob
import numpy as np
from .base_dataset import BaseCrowdDataset

class UCF_QNRF(BaseCrowdDataset):
    """
    UCF-QNRF Dataset Loader.
    
    Supporta due strutture:
    1. Originale: ROOT/Train/images/*.jpg + ROOT/Train/gt/*_ann.mat
    2. Flat: ROOT/train/scene01/*.jpg + *.txt (stesso nome)
    """
    
    def get_image_list(self, split):
        """Trova tutte le immagini per lo split specificato."""
        
        # Mapping split names
        split_mapping = {
            "train": ["Train", "train"],
            "test": ["Test", "test"],
            "val": ["Test", "test"],  # UCF-QNRF non ha val, usa test
        }
        
        split_candidates = split_mapping.get(split.lower(), [split])
        
        imgs = []
        
        for split_name in split_candidates:
            # Prova struttura originale: ROOT/Split/images/*.jpg
            img_dir = os.path.join(self.root, split_name, "images")
            if os.path.isdir(img_dir):
                found = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
                if found:
                    print(f"[UCF-QNRF] Trovate {len(found)} immagini in {img_dir}")
                    return found
            
            # Prova struttura flat: ROOT/split/scene*/*.jpg
            scene_pattern = os.path.join(self.root, split_name, "scene*")
            scene_dirs = glob.glob(scene_pattern)
            
            for scene_dir in scene_dirs:
                found = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
                imgs.extend(found)
            
            if imgs:
                print(f"[UCF-QNRF] Trovate {len(imgs)} immagini in {self.root}/{split_name}/scene*/")
                return sorted(imgs)
            
            # Prova struttura flat diretta: ROOT/split/*.jpg
            flat_dir = os.path.join(self.root, split_name)
            if os.path.isdir(flat_dir):
                found = sorted(glob.glob(os.path.join(flat_dir, "*.jpg")))
                if found:
                    print(f"[UCF-QNRF] Trovate {len(found)} immagini in {flat_dir}")
                    return found
        
        raise FileNotFoundError(
            f"Nessuna immagine trovata per split '{split}' in {self.root}\n"
            f"Cercato in: {split_candidates}"
        )
    
    def load_points(self, img_path):
        """
        Carica i punti GT per un'immagine.
        
        Supporta:
        1. File .mat (struttura originale): ROOT/Split/gt/img_name_ann.mat
        2. File .txt (struttura flat): stesso path dell'immagine, estensione .txt
        """
        pts = []
        
        # === PROVA 1: File .txt nella stessa cartella ===
        txt_path = img_path.replace(".jpg", ".txt")
        if os.path.isfile(txt_path):
            pts = self._load_txt_points(txt_path)
            return np.array(pts, dtype=np.float32)
        
        # === PROVA 2: File .mat in cartella gt/ (struttura originale) ===
        # img_path: ROOT/Train/images/img_0001.jpg
        # gt_path:  ROOT/Train/gt/img_0001_ann.mat
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(os.path.dirname(img_path))  # ROOT/Train
        
        mat_path = os.path.join(parent_dir, "gt", f"{base_name}_ann.mat")
        if os.path.isfile(mat_path):
            pts = self._load_mat_points(mat_path)
            return np.array(pts, dtype=np.float32)
        
        # === PROVA 3: File .mat senza suffisso _ann ===
        mat_path2 = os.path.join(parent_dir, "gt", f"{base_name}.mat")
        if os.path.isfile(mat_path2):
            pts = self._load_mat_points(mat_path2)
            return np.array(pts, dtype=np.float32)
        
        # Nessun GT trovato - ritorna array vuoto (immagine senza persone?)
        print(f"[UCF-QNRF] Warning: GT non trovato per {img_path}")
        return np.array(pts, dtype=np.float32)
    
    def _load_txt_points(self, txt_path):
        """Carica punti da file .txt (formato: x y per riga)."""
        pts = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        pts.append([x, y])
                    except ValueError:
                        continue
        return pts
    
    def _load_mat_points(self, mat_path):
        """Carica punti da file .mat (formato UCF-QNRF)."""
        import scipy.io as sio
        
        try:
            mat = sio.loadmat(mat_path)
            
            # Prova diverse chiavi comuni
            for key in ["annPoints", "image_info", "points", "gt"]:
                if key in mat:
                    data = mat[key]
                    
                    # Gestisci struttura nested di image_info
                    if key == "image_info":
                        try:
                            data = data[0][0][0][0][0]
                        except (IndexError, TypeError):
                            continue
                    
                    if hasattr(data, 'shape') and len(data.shape) >= 2:
                        return data[:, :2].tolist()
            
            print(f"[UCF-QNRF] Warning: chiavi disponibili in {mat_path}: {list(mat.keys())}")
            return []
            
        except Exception as e:
            print(f"[UCF-QNRF] Errore caricamento {mat_path}: {e}")
            return []