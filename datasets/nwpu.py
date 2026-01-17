import os
import glob
import numpy as np
from .base_dataset import BaseDataset

class NWPU(BaseDataset):
    """
    NWPU-Crowd Dataset Loader.
    Struttura rilevata:
    - root/train/scene01/0001.jpg
    - root/train/scene01/0001.txt
    """
    
    def load_data(self):
        """
        Cerca le immagini navigando nelle cartelle (es. scene01, scene02...)
        poiché non sono presenti file .list nella root.
        """
        # Gestione split: NWPU ha solitamente 'train', 'val', 'test'
        # Se chiedi 'val' e non esiste, usiamo 'test' (o 'train' se preferisci fare split automatico)
        split_name = self.split
        if split_name == "val":
            # Spesso il val set è una parte del train o il test set stesso in alcuni benchmark
            # Qui cerchiamo la cartella "val", se non c'è proviamo "test"
            if not os.path.isdir(os.path.join(self.root, "val")):
                split_name = "test"

        # Costruisce il path base: es. .../NWPU/train
        base_dir = os.path.join(self.root, split_name)
        
        if not os.path.exists(base_dir):
             raise FileNotFoundError(f"Cartella dello split non trovata: {base_dir}")

        self.img_paths = []
        
        # === STRATEGIA DI RICERCA ===
        # Cerca pattern: ROOT/split/scene*/*.jpg
        # Questo cattura scene01, scene02, etc.
        search_pattern = os.path.join(base_dir, "scene*", "*.jpg")
        found_imgs = sorted(glob.glob(search_pattern))
        
        if len(found_imgs) > 0:
            print(f"[NWPU] Trovate {len(found_imgs)} immagini in {base_dir}/scene*")
            self.img_paths = found_imgs
        else:
            direct_pattern = os.path.join(base_dir, "*.jpg")
            found_imgs = sorted(glob.glob(direct_pattern))
            if len(found_imgs) > 0:
                print(f"[NWPU] Trovate {len(found_imgs)} immagini direttamente in {base_dir}")
                self.img_paths = found_imgs
            else:
                 raise RuntimeError(f"Nessuna immagine .jpg trovata in {base_dir} (cercato in scene* e root)")

    def load_points(self, img_path):
        """
        Carica i punti GT dal file .txt che si trova ACCANTO all'immagine.
        """
        pts = []
        
        # Sostituzione diretta: immagine.jpg -> immagine.txt
        txt_path = img_path.replace(".jpg", ".txt")
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parsing robusto: gestisce spazi o virgole
                        parts = line.replace(",", " ").split()
                        
                        # NWPU solitamente ha formato: x y (e a volte altri dati che ignoriamo)
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                pts.append([x, y])
                            except ValueError:
                                continue
            except Exception as e:
                print(f"[NWPU] Errore lettura file {txt_path}: {e}")
                
        return np.array(pts, dtype=np.float32)