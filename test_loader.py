import yaml
from easydict import EasyDict as edict
from datasets.shha import SHHA

# Carica una configurazione finta
cfg = edict()
# IMPORTANTE: Metti qui il percorso che hai trovato con pwd
cfg.ROOT = "/home/C.ROMANO50/datasets/content/ShangaiTech-A" 

try:
    # Prova a inizializzare il dataset
    ds = SHHA(cfg.ROOT, split="train")
    print("Inizializzazione OK!")
    
    # Prova a leggere la prima immagine
    item = ds[0]
    print(f"Immagine caricata: {item['img_path']}")
    print(f"Dimensione immagine: {item['image'].shape}")
    print(f"Numero persone (punti): {len(item['points'])}")
    print("TUTTO FUNZIONA CORRETTAMENTE!")

except Exception as e:
    print(f"ERRORE: {e}")