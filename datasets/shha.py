import os
import glob
import numpy as np
from .base_dataset import BaseDataset

class SHHA(BaseDataset):
    def load_data(self):
        """
        Gestisce i path per la struttura specifica:
        ROOT/train/scene01/*.jpg
        ROOT/test/scene01/*.jpg (o simile)
        Carica i percorsi delle immagini in self.img_paths.
        """
        split = self.split
        # Mappa i nomi standard (train_data) ai nomi reali (train)
        if split in ["train", "train_data"]:
            actual_dir = "train"
        elif split in ["val", "test", "test_data"]:
            actual_dir = "test"
        else:
            actual_dir = split

        # TENTATIVO 1: Cerca dentro la sottocartella 'scene01'
        path_scene01 = os.path.join(self.root, actual_dir, "scene01", "*.jpg")
        imgs = sorted(glob.glob(path_scene01))

        # TENTATIVO 2: Se non trova nulla, cerca direttamente nella cartella train/test
        if len(imgs) == 0:
            path_direct = os.path.join(self.root, actual_dir, "*.jpg")
            imgs = sorted(glob.glob(path_direct))

        # TENTATIVO 3: Struttura originale con images/
        if len(imgs) == 0:
            path_images = os.path.join(self.root, actual_dir, "images", "*.jpg")
            imgs = sorted(glob.glob(path_images))

        # Check finale
        if len(imgs) == 0:
            raise FileNotFoundError(
                f"Nessuna immagine trovata per lo split '{split}'.\n"
                f"Ho cercato in: {path_scene01}\n"
                f"E in: {path_direct}"
            )
        
        print(f"Dataset {split}: Trovate {len(imgs)} immagini.")
        self.img_paths = imgs

    def load_points(self, img_path):
        """
        Carica i punti dai file .txt situati ACCANTO all'immagine.
        Formato atteso: "X Y" (es. 533 583)
        """
        # Sostituisce l'estensione .jpg con .txt
        txt_path = img_path.replace('.jpg', '.txt')
        
        # Fallback per estensioni maiuscole
        if not os.path.exists(txt_path):
             txt_path = img_path.replace('.jpg', '.TXT')

        points = []
        
        if os.path.isfile(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split()
                    
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            points.append([x, y])
                        except ValueError:
                            continue
            
            return np.array(points, dtype=np.float32)

        else:
            raise FileNotFoundError(f"File di annotazione non trovato: {txt_path}")