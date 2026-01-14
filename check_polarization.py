import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import collate_fn

def check_polarization(config_path, checkpoint_path, device='cuda'):
    # 1. Carica Configurazione
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🔄 Caricamento configurazione da {config_path}")
    print(f"🔄 Caricamento checkpoint da {checkpoint_path}")

    # 2. Inizializza Modello
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    dataset_name = config["DATASET"]
    if "BINS_CONFIG" in config:
        bin_cfg = config["BINS_CONFIG"][dataset_name]
        bins = bin_cfg["bins"]
        bin_centers = bin_cfg["bin_centers"]
    else:
        bins = 10
        bin_centers = None 

    model = P2R_ZIP_Model(
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        bins=bins, 
        bin_centers=bin_centers,
        upsample_to_input=False
    ).to(device)
    
    # 3. Carica i Pesi
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. Costruisci il Dataloader
    print(f"📚 Caricamento dataset: {dataset_name}...")
    DatasetClass = get_dataset(dataset_name)
    val_tf = build_transforms(config["DATA"], is_train=False)
    
    val_dataset = DatasetClass(
        root=config["DATA"]["ROOT"], 
        split=config["DATA"]["VAL_SPLIT"], 
        transforms=val_tf
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    all_pis = []

    print("🚀 Inizio analisi sul Validation Set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch
            
            images = images.to(device)
            
            # --- FORWARD PASS ---
            features = model.backbone(images)
            zip_out = model.zip_head(features, model.bin_centers)
            
            # --- FIX: Gestione corretta dell'output (Dizionario) ---
            if isinstance(zip_out, dict):
                logit_pi = zip_out["logit_pi_maps"]
            elif isinstance(zip_out, tuple):
                logit_pi = zip_out[0]
            else:
                logit_pi = zip_out

            # Sigmoide per probabilità
            pi = torch.sigmoid(logit_pi)
            
            # Estrazione valori foreground
            if pi.shape[1] == 2:
                pi_vals = pi[:, 1, :, :].cpu().numpy().flatten()
            else:
                pi_vals = pi.cpu().numpy().flatten()
                
            # Campionamento (10%)
            if len(pi_vals) > 0:
                sampled_vals = np.random.choice(pi_vals, size=max(1, int(len(pi_vals)*0.1)), replace=False)
                all_pis.extend(sampled_vals)

    # 5. Analisi Statistica
    if len(all_pis) == 0:
        print("❌ Nessun dato raccolto.")
        return

    all_pis = np.array(all_pis)
    
    uncertain_mask = (all_pis > 0.2) & (all_pis < 0.8)
    uncertain_percentage = np.mean(uncertain_mask) * 100
    
    print(f"\n📊 --- RISULTATI ANALISI ---")
    print(f"Totale pixel analizzati: {len(all_pis)}")
    print(f"Valore Medio pi: {np.mean(all_pis):.4f}")
    print(f"⚠️  Percentuale in Zona Grigia (0.2 - 0.8): {uncertain_percentage:.2f}%")
    
    # 6. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(all_pis, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribuzione Probabilità ZIP (pi)\nIncertezza (0.2-0.8): {uncertain_percentage:.1f}%')
    plt.xlabel('Probabilità (0=Vuoto, 1=Folla)')
    plt.ylabel('Frequenza (log scale)')
    plt.yscale('log')
    plt.axvline(0.3, color='r', linestyle='--', label='Tua Soglia (0.3)')
    plt.axvspan(0.2, 0.8, color='yellow', alpha=0.2, label='Zona Incertezza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'polarization_check.png'
    plt.savefig(output_file)
    print(f"\n✅ Grafico salvato come: {output_file}")
    
    if uncertain_percentage > 10:
        print("\n❌ VERDETTO: Modello NON polarizzato. Soglia 0.3 rischiosa.")
        print("   -> Necessario riaddestramento Stage 1 con Polarization Loss.")
    else:
        print("\n✅ VERDETTO: Modello ben polarizzato. Soglia 0.3 sicura.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Errore: Checkpoint non trovato: {args.checkpoint}")
    else:
        check_polarization(args.config, args.checkpoint)