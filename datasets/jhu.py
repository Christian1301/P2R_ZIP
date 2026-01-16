import os
import numpy as np
from .base_dataset import BaseDataset

class JHU_Crowd(BaseDataset):
    def load_data(self):
        """
        Carica i path delle immagini leggendo i file .list (es. train.list).
        Struttura attesa: root/train.list contenente righe "path/img.jpg path/gt.txt"
        """
        # Mappa il nome dello split al file list corrispondente
        list_filename = f"{self.split}.list"
        list_path = os.path.join(self.root, list_filename)

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"File lista non trovato: {list_path}")

        self.img_paths = []
        
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    # parts[0] è il percorso relativo dell'immagine (es. train/scene01/0001.jpg)
                    rel_path = parts[0]
                    abs_path = os.path.join(self.root, rel_path)
                    self.img_paths.append(abs_path)

        print(f"JHU Dataset ({self.split}): Trovate {len(self.img_paths)} immagini da {list_filename}")

        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata nel file {list_filename}")

    def load_points(self, img_path):
        """
        Carica i punti dal file .txt.
        In questa versione del dataset, il file .txt è nella stessa cartella del .jpg.
        """
        # Sostituisce l'estensione .jpg con .txt
        gt_path = img_path.replace(".jpg", ".txt")
        
        pts = []
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Salta righe vuote
                    if not line:
                        continue
                    
                    parts = line.split()
                    # Prende solo x e y (i primi due valori)
                    # Alcuni dataset JHU hanno: x y w h sigma blur... a noi servono solo x y
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            pts.append([x, y])
                        except ValueError:
                            # Gestisce eventuali header o righe malformate
                            continue
                            
        return np.array(pts, dtype=np.float32)