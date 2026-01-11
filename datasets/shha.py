# P2R_ZIP/datasets/shha.py
import os
import glob
import numpy as np
from .base_dataset import BaseCrowdDataset

class SHHA(BaseCrowdDataset):
    def get_image_list(self, split):
        """
        Gestisce i path per la tua struttura specifica:
        ROOT/train/scene01/*.jpg
        ROOT/test/scene01/*.jpg (o simile)
        """
        # Mappa i nomi standard (train_data) ai tuoi nomi reali (train)
        if split in ["train", "train_data"]:
            actual_dir = "train"
        elif split in ["val", "test", "test_data"]:
            actual_dir = "test"
        else:
            actual_dir = split

        # TENTATIVO 1: Cerca dentro la sottocartella 'scene01' (come nel tuo caso)
        path_scene01 = os.path.join(self.root, actual_dir, "scene01", "*.jpg")
        imgs = sorted(glob.glob(path_scene01))

        # TENTATIVO 2: Se non trova nulla, cerca direttamente nella cartella train/test
        if len(imgs) == 0:
            path_direct = os.path.join(self.root, actual_dir, "*.jpg")
            imgs = sorted(glob.glob(path_direct))

        # Check finale
        if len(imgs) == 0:
            raise FileNotFoundError(
                f"Nessuna immagine trovata per lo split '{split}'.\n"
                f"Ho cercato in: {path_scene01}\n"
                f"E in: {path_direct}"
            )
        
        print(f"Dataset {split}: Trovate {len(imgs)} immagini.")
        return imgs

    def load_points(self, img_path):
        """
        Carica i punti dai file .txt situati ACCANTO all'immagine.
        Formato atteso: "X Y" (es. 533 583)
        """
        # Sostituisce l'estensione .jpg con .txt
        # Es: .../scene01/IMG_100.jpg -> .../scene01/IMG_100.txt
        txt_path = img_path.replace('.jpg', '.txt')
        
        # Fallback per estensioni maiuscole (a volte capita su Linux)
        if not os.path.exists(txt_path):
             txt_path = img_path.replace('.jpg', '.TXT')

        points = []
        
        if os.path.isfile(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    # Divide la riga in parti basandosi sugli spazi
                    parts = line.split()
                    
                    # Ci aspettiamo almeno 2 numeri (x, y)
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            points.append([x, y])
                        except ValueError:
                            # Se la riga non contiene numeri validi, la salta
                            continue
            
            # Restituisce un array numpy di float32
            return np.array(points, dtype=np.float32)

        else:
            # Se il file txt non esiste, solleva un errore (così te ne accorgi subito)
            raise FileNotFoundError(f"File di annotazione non trovato: {txt_path}")