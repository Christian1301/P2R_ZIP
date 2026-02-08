import os
import numpy as np
from .base_dataset import BaseDataset

class UCF_QNRF(BaseDataset):
    """
    Loader per UCF-QNRF (Versione pre-processata con .list e .txt).
    Struttura:
    - root/train.list
    - root/train/sceneXX/img_0001.jpg
    - root/train/sceneXX/img_0001.txt (Annotazioni)
    """

    def load_data(self):
        """
        Carica i percorsi delle immagini leggendo i file .list.
        """
        # Mappa il nome dello split al file list (es. train -> train.list)
        # Gestisce anche eventuali maiuscole/minuscole
        split_name = self.split.lower()
        if split_name == "val": 
            split_name = "test"  # UCF spesso usa test come validazione se non specificato diversamente
            
        list_filename = f"{split_name}.list"
        list_path = os.path.join(self.root, list_filename)

        if not os.path.exists(list_path):
            raise FileNotFoundError(
                f"File lista non trovato: {list_path}\n"
                f"Assicurati di essere nella cartella corretta (dovrebbe contenere {list_filename})"
            )

        self.img_paths = []
        
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    # parts[0] è il percorso relativo (es. train/scene01/img_0001.jpg)
                    rel_path = parts[0]
                    abs_path = os.path.join(self.root, rel_path)
                    
                    # Verifica opzionale che il file esista davvero
                    if not os.path.exists(abs_path):
                         # A volte i .list hanno path che iniziano con / o senza...
                         # Proviamo a sistemare se necessario, ma di base ci fidiamo del list
                         pass 
                         
                    self.img_paths.append(abs_path)

        print(f"[UCF-QNRF] Split '{self.split}': Trovate {len(self.img_paths)} immagini da {list_filename}")

        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata nel file {list_filename}")

    def load_points(self, img_path):
        """
        Carica i punti dal file .txt situato accanto all'immagine.
        """
        # Sostituisce l'estensione .jpg con .txt
        txt_path = img_path.replace(".jpg", ".txt")
        
        pts = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Gestisce separatori diversi (spazio o virgola)
                    parts = line.replace(",", " ").split()
                    
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            pts.append([x, y])
                        except ValueError:
                            continue
                            
        return np.array(pts, dtype=np.float32)