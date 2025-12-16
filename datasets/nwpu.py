# P2R_ZIP/datasets/nwpu.py
# VERSIONE AGGIORNATA per struttura flat (immagini e txt nella stessa cartella)
#
# Struttura attesa:
# ROOT/
# ├── train/scene01/0001.jpg, 0001.txt, ...
# ├── val/scene01/...
# └── test/scene01/...
#
# Il file .txt contiene coordinate x y (una per riga)

import os
import glob
import numpy as np
from .base_dataset import BaseCrowdDataset

class NWPU(BaseCrowdDataset):
    """
    NWPU-Crowd Dataset Loader.
    
    Supporta due strutture:
    1. Standard: ROOT/split/images/*.jpg + ROOT/split/gt/*.txt
    2. Flat: ROOT/split/scene*/*.jpg + *.txt (stesso nome)
    """
    
    def get_image_list(self, split):
        """Trova tutte le immagini per lo split specificato."""
        
        imgs = []
        
        # === PROVA 1: Struttura standard ROOT/split/images/*.jpg ===
        img_dir = os.path.join(self.root, split, "images")
        if os.path.isdir(img_dir):
            found = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            if found:
                print(f"[NWPU] Trovate {len(found)} immagini in {img_dir}")
                return found
        
        # === PROVA 2: Struttura flat ROOT/split/scene*/*.jpg ===
        scene_pattern = os.path.join(self.root, split, "scene*")
        scene_dirs = sorted(glob.glob(scene_pattern))
        
        for scene_dir in scene_dirs:
            found = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
            imgs.extend(found)
        
        if imgs:
            print(f"[NWPU] Trovate {len(imgs)} immagini in {len(scene_dirs)} scene directories")
            return sorted(imgs)
        
        # === PROVA 3: Struttura flat diretta ROOT/split/*.jpg ===
        flat_dir = os.path.join(self.root, split)
        if os.path.isdir(flat_dir):
            found = sorted(glob.glob(os.path.join(flat_dir, "*.jpg")))
            if found:
                print(f"[NWPU] Trovate {len(found)} immagini in {flat_dir}")
                return found
        
        raise FileNotFoundError(
            f"Nessuna immagine trovata per split '{split}' in {self.root}\n"
            f"Cercato: {img_dir}, {scene_pattern}, {flat_dir}"
        )
    
    def load_points(self, img_path):
        """
        Carica i punti GT per un'immagine.
        
        Supporta:
        1. File .txt nella stessa cartella (struttura flat)
        2. File .txt in cartella gt/ (struttura standard)
        """
        pts = []
        
        # === PROVA 1: File .txt nella stessa cartella ===
        txt_path = img_path.replace(".jpg", ".txt")
        if os.path.isfile(txt_path):
            pts = self._load_txt_points(txt_path)
            return np.array(pts, dtype=np.float32)
        
        # === PROVA 2: File .txt in cartella gt/ ===
        # img_path: ROOT/train/images/0001.jpg
        # gt_path:  ROOT/train/gt/0001.txt
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Trova la cartella gt relativa
        img_dir = os.path.dirname(img_path)
        parent_dir = os.path.dirname(img_dir)
        
        gt_candidates = [
            os.path.join(parent_dir, "gt", f"{base_name}.txt"),
            os.path.join(img_dir.replace("/images", "/gt"), f"{base_name}.txt"),
        ]
        
        for gt_path in gt_candidates:
            if os.path.isfile(gt_path):
                pts = self._load_txt_points(gt_path)
                return np.array(pts, dtype=np.float32)
        
        # Nessun GT trovato - potrebbe essere un'immagine senza persone
        # NWPU ha molte immagini con 0 persone
        return np.array(pts, dtype=np.float32)
    
    def _load_txt_points(self, txt_path):
        """
        Carica punti da file .txt.
        
        Formato NWPU: x y (spazi o virgole come separatori)
        Alcune annotazioni potrebbero avere campi extra.
        """
        pts = []
        
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Sostituisci virgole con spazi e splitta
                    parts = line.replace(",", " ").split()
                    
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            pts.append([x, y])
                        except ValueError:
                            # Salta righe malformate
                            continue
        except Exception as e:
            print(f"[NWPU] Errore lettura {txt_path}: {e}")
        
        return pts