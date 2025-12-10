#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNOSTICA: P2R Head vs π-Head Bottleneck Analysis

Obiettivo: Capire se l'errore viene dal P2R head (stima densità) 
o dal π-head (maschera regioni attive).

Test eseguiti:
1. P2R puro (senza maschera π) - usa tutta la density map
2. π-head analysis - quanto masking sta facendo?
3. Correlazione errore vs π-coverage
4. Analisi per fasce di densità
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from models.p2r_zip_model import P2R_ZIP_Model
from datasets import get_dataset
from datasets.transforms import build_transforms
from train_utils import init_seeds, canonicalize_p2r_grid, collate_fn


def compute_count_from_density(density, down_h, down_w):
    """Calcola il conteggio dalla density map."""
    cell_area = down_h * down_w
    return torch.sum(density, dim=(1, 2, 3)) / cell_area


@torch.no_grad()
def diagnose_bottleneck(model, dataloader, device, default_down, verbose=True):
    """
    Analisi dettagliata per capire dove sta il problema.
    """
    model.eval()
    
    results = {
        # Conteggi
        'gt_counts': [],
        'p2r_raw_counts': [],      # P2R senza maschera
        'p2r_masked_counts': [],   # P2R con maschera π
        'zip_counts': [],          # Conteggio ZIP
        
        # Metriche π-head
        'pi_coverage': [],         # % pixels attivi
        'pi_mean': [],             # Media probabilità π
        
        # Per analisi spaziale
        'density_in_active': [],   # Densità media in regioni attive
        'density_in_inactive': [], # Densità media in regioni inattive
        
        # Errori
        'error_p2r_raw': [],
        'error_p2r_masked': [],
        'error_zip': [],
    }
    
    for images, gt_density, points in tqdm(dataloader, desc="Diagnostica"):
        images = images.to(device)
        gt_density = gt_density.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # === 1. P2R RAW (senza maschera) ===
        p2r_density_raw = outputs["p2r_density"]
        _, _, h_in, w_in = images.shape
        p2r_density_raw, down_tuple, _ = canonicalize_p2r_grid(
            p2r_density_raw, (h_in, w_in), default_down
        )
        down_h, down_w = down_tuple
        
        p2r_raw_count = compute_count_from_density(p2r_density_raw, down_h, down_w)
        
        # === 2. π-head analysis ===
        logit_pi = outputs["logit_pi_maps"]  # [B, 2, H, W]
        pi_probs = torch.softmax(logit_pi, dim=1)
        pi_active = pi_probs[:, 1:2, :, :]  # Probabilità "pieno"
        
        # Upsample π alla dimensione di p2r_density se necessario
        if pi_active.shape[-2:] != p2r_density_raw.shape[-2:]:
            pi_active_upsampled = F.interpolate(
                pi_active, size=p2r_density_raw.shape[-2:], mode='nearest'
            )
        else:
            pi_active_upsampled = pi_active
        
        # Maschera binaria (threshold 0.5)
        pi_mask = (pi_active_upsampled > 0.5).float()
        
        # === 3. P2R MASKED ===
        p2r_density_masked = p2r_density_raw * pi_mask
        p2r_masked_count = compute_count_from_density(p2r_density_masked, down_h, down_w)
        
        # === 4. ZIP count (se disponibile) ===
        if "zip_expected_count" in outputs:
            zip_count = outputs["zip_expected_count"].squeeze()
        else:
            # Calcola da density map ZIP
            zip_density = outputs.get("zip_density_map", gt_density)
            zip_count = zip_density.sum(dim=(1,2,3))
        
        # === 5. Ground truth ===
        for idx, pts in enumerate(points):
            gt = len(pts) if pts is not None else 0
            
            results['gt_counts'].append(gt)
            results['p2r_raw_counts'].append(p2r_raw_count[idx].item())
            results['p2r_masked_counts'].append(p2r_masked_count[idx].item())
            
            if isinstance(zip_count, torch.Tensor) and zip_count.numel() > idx:
                results['zip_counts'].append(zip_count[idx].item() if zip_count.dim() > 0 else zip_count.item())
            else:
                results['zip_counts'].append(0)
            
            # π metrics
            pi_cov = pi_mask[idx].mean().item() * 100
            pi_m = pi_active_upsampled[idx].mean().item()
            results['pi_coverage'].append(pi_cov)
            results['pi_mean'].append(pi_m)
            
            # Densità in regioni attive vs inattive
            density_map = p2r_density_raw[idx, 0]
            mask = pi_mask[idx, 0]
            
            if mask.sum() > 0:
                density_active = density_map[mask > 0.5].mean().item()
            else:
                density_active = 0
                
            if (1 - mask).sum() > 0:
                density_inactive = density_map[mask < 0.5].mean().item()
            else:
                density_inactive = 0
                
            results['density_in_active'].append(density_active)
            results['density_in_inactive'].append(density_inactive)
            
            # Errori
            results['error_p2r_raw'].append(abs(p2r_raw_count[idx].item() - gt))
            results['error_p2r_masked'].append(abs(p2r_masked_count[idx].item() - gt))
            results['error_zip'].append(abs(results['zip_counts'][-1] - gt))
    
    return results


def analyze_results(results):
    """Analizza i risultati diagnostici."""
    
    gt = np.array(results['gt_counts'])
    p2r_raw = np.array(results['p2r_raw_counts'])
    p2r_masked = np.array(results['p2r_masked_counts'])
    pi_cov = np.array(results['pi_coverage'])
    
    err_raw = np.array(results['error_p2r_raw'])
    err_masked = np.array(results['error_p2r_masked'])
    
    print("\n" + "="*70)
    print("📊 DIAGNOSI BOTTLENECK: P2R vs π-Head")
    print("="*70)
    
    # === 1. Confronto MAE ===
    mae_raw = np.mean(err_raw)
    mae_masked = np.mean(err_masked)
    
    print(f"\n🎯 MAE Comparison:")
    print(f"   P2R RAW (no mask):    {mae_raw:.2f}")
    print(f"   P2R MASKED (con π):   {mae_masked:.2f}")
    print(f"   Differenza:           {mae_masked - mae_raw:+.2f}")
    
    if mae_masked > mae_raw * 1.1:
        print(f"   ⚠️  La maschera π PEGGIORA il risultato del {((mae_masked/mae_raw)-1)*100:.1f}%")
        print(f"   → Il π-head sta mascherando regioni con persone!")
    elif mae_masked < mae_raw * 0.9:
        print(f"   ✅ La maschera π MIGLIORA il risultato del {((mae_raw/mae_masked)-1)*100:.1f}%")
    else:
        print(f"   ≈  La maschera π ha effetto minimo")
    
    # === 2. Bias Analysis ===
    print(f"\n📈 Bias Analysis:")
    bias_raw = np.sum(p2r_raw) / np.sum(gt)
    bias_masked = np.sum(p2r_masked) / np.sum(gt)
    print(f"   P2R RAW bias:    {bias_raw:.3f} (1.0 = perfetto)")
    print(f"   P2R MASKED bias: {bias_masked:.3f}")
    
    # === 3. π-head Coverage ===
    print(f"\n🎭 π-Head Coverage:")
    print(f"   Media:  {np.mean(pi_cov):.1f}%")
    print(f"   Min:    {np.min(pi_cov):.1f}%")
    print(f"   Max:    {np.max(pi_cov):.1f}%")
    
    # === 4. Analisi per fasce ===
    print(f"\n📊 Analisi per Fasce di Densità:")
    
    sparse_idx = gt <= 100
    medium_idx = (gt > 100) & (gt <= 500)
    dense_idx = gt > 500
    
    for name, idx in [("Sparse (0-100)", sparse_idx), 
                       ("Medium (100-500)", medium_idx),
                       ("Dense (500+)", dense_idx)]:
        if idx.sum() == 0:
            continue
            
        mae_r = np.mean(err_raw[idx])
        mae_m = np.mean(err_masked[idx])
        cov = np.mean(pi_cov[idx])
        n = idx.sum()
        
        delta = mae_m - mae_r
        symbol = "⚠️" if delta > 5 else "✅" if delta < -5 else "≈"
        
        print(f"\n   {name} ({n} imgs):")
        print(f"      MAE raw:    {mae_r:.1f}")
        print(f"      MAE masked: {mae_m:.1f} ({delta:+.1f}) {symbol}")
        print(f"      π coverage: {cov:.1f}%")
        
        # Correlazione errore con coverage
        if n > 3:
            corr = np.corrcoef(pi_cov[idx], err_masked[idx])[0, 1]
            print(f"      Corr(π_cov, error): {corr:.3f}")
    
    # === 5. Densità in regioni attive vs inattive ===
    print(f"\n🔍 Densità per Regione:")
    d_active = np.array(results['density_in_active'])
    d_inactive = np.array(results['density_in_inactive'])
    
    print(f"   In regioni ATTIVE (π>0.5):   {np.mean(d_active):.3f}")
    print(f"   In regioni INATTIVE (π<0.5): {np.mean(d_inactive):.3f}")
    
    if np.mean(d_inactive) > 0.1 * np.mean(d_active):
        print(f"   ⚠️  C'è densità significativa in regioni mascherate!")
        print(f"   → Il π-head maschera aree con persone")
    
    # === 6. VERDETTO FINALE ===
    print("\n" + "="*70)
    print("🏁 VERDETTO FINALE")
    print("="*70)
    
    # Determina il bottleneck
    if mae_masked > mae_raw * 1.15:
        print("""
   🔴 BOTTLENECK: π-Head
   
   Il π-head sta mascherando regioni che contengono persone.
   La density map P2R di per sé è migliore.
   
   AZIONI SUGGERITE:
   1. Ridurre il threshold π (da 0.5 a 0.3)
   2. Aumentare pos_weight nella BCE del π-head
   3. Nel joint training, dare meno peso alla ZIP loss
   4. Considerare di non usare la maschera per il conteggio finale
""")
        return "pi_head"
        
    elif mae_raw > 70 and bias_raw < 0.85:
        print("""
   🔴 BOTTLENECK: P2R Head (sottostima)
   
   Il P2R head sottostima sistematicamente, specialmente nelle scene dense.
   
   AZIONI SUGGERITE:
   1. Aumentare log_scale iniziale
   2. Aggiungere loss asimmetrica che penalizza di più la sottostima
   3. Data augmentation con focus su scene dense
""")
        return "p2r_underestimate"
        
    elif mae_raw > 70 and bias_raw > 1.15:
        print("""
   🔴 BOTTLENECK: P2R Head (sovrastima)
   
   Il P2R head sovrastima sistematicamente.
   
   AZIONI SUGGERITE:
   1. Ridurre log_scale
   2. Aumentare weight decay
   3. Verificare il target della loss spaziale
""")
        return "p2r_overestimate"
        
    else:
        print(f"""
   🟡 BOTTLENECK: Varianza alta
   
   MAE RAW: {mae_raw:.1f}, Bias: {bias_raw:.3f}
   Non c'è un singolo bottleneck chiaro.
   
   AZIONI SUGGERITE:
   1. Joint training con loss bilanciata
   2. Focus sul ridurre varianza nelle scene dense
   3. Considerare ensemble o TTA
""")
        return "variance"


def suggest_joint_training_config(bottleneck, results):
    """Suggerisce configurazione per joint training basata sulla diagnosi."""
    
    print("\n" + "="*70)
    print("⚙️  CONFIGURAZIONE SUGGERITA PER JOINT TRAINING")
    print("="*70)
    
    gt = np.array(results['gt_counts'])
    pi_cov = np.array(results['pi_coverage'])
    
    # Analisi per determinare i pesi
    dense_idx = gt > 500
    dense_coverage = np.mean(pi_cov[dense_idx]) if dense_idx.sum() > 0 else 30
    
    if bottleneck == "pi_head":
        config = {
            "ZIP_SCALE": 0.05,      # Molto basso - π-head problematico
            "P2R_ALPHA": 1.0,       # Alto - P2R è buono
            "COUNT_L1_W": 3.0,      # Alto - supervisione diretta
            "PI_THRESHOLD": 0.3,   # Più basso per non mascherare troppo
            "UNFREEZE_BACKBONE": False,
        }
        print("""
   Il π-head maschera troppe persone. Configurazione conservativa:
""")
    elif bottleneck == "p2r_underestimate":
        config = {
            "ZIP_SCALE": 0.2,
            "P2R_ALPHA": 0.5,
            "COUNT_L1_W": 5.0,      # Molto alto per forzare conteggio corretto
            "PI_THRESHOLD": 0.5,
            "UNFREEZE_BACKBONE": True,  # Serve adattare features
            "ASYMMETRIC_LOSS": True,    # Penalizza sottostima di più
        }
        print("""
   P2R sottostima. Serve supervisione forte sul conteggio:
""")
    else:
        config = {
            "ZIP_SCALE": 0.15,
            "P2R_ALPHA": 0.7,
            "COUNT_L1_W": 2.5,
            "PI_THRESHOLD": 0.5,
            "UNFREEZE_BACKBONE": True,
        }
        print("""
   Configurazione bilanciata per joint training:
""")
    
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    return config


def main():
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml non trovato")
        return
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["DEVICE"])
    init_seeds(config["SEED"])
    
    print("🔍 Avvio Diagnostica Bottleneck...")
    print(f"   Device: {device}")
    
    # Dataset
    data_cfg = config["DATA"]
    val_transforms = build_transforms(data_cfg, is_train=False)
    DatasetClass = get_dataset(config["DATASET"])
    
    val_dataset = DatasetClass(
        root=data_cfg["ROOT"],
        split=data_cfg["VAL_SPLIT"],
        block_size=data_cfg["ZIP_BLOCK_SIZE"],
        transforms=val_transforms
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    # Modello
    bin_config = config["BINS_CONFIG"][config["DATASET"]]
    zip_head_kwargs = {
        "lambda_scale": config["ZIP_HEAD"].get("LAMBDA_SCALE", 0.5),
        "lambda_max": config["ZIP_HEAD"].get("LAMBDA_MAX", 8.0),
        "use_softplus": config["ZIP_HEAD"].get("USE_SOFTPLUS", True),
    }
    
    model = P2R_ZIP_Model(
        bins=bin_config["bins"],
        bin_centers=bin_config["bin_centers"],
        backbone_name=config["MODEL"]["BACKBONE"],
        pi_thresh=config["MODEL"]["ZIP_PI_THRESH"],
        gate=config["MODEL"]["GATE"],
        upsample_to_input=False,
        zip_head_kwargs=zip_head_kwargs
    ).to(device)
    
    # Carica checkpoint Stage 2
    output_dir = os.path.join(config["EXP"]["OUT_DIR"], config["RUN_NAME"])
    ckpt_path = os.path.join(output_dir, "stage2_best.pth")
    
    if not os.path.isfile(ckpt_path):
        print(f"❌ Checkpoint non trovato: {ckpt_path}")
        return
    
    print(f"✅ Caricamento: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        state = state["model"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    
    # Esegui diagnostica
    default_down = data_cfg.get("P2R_DOWNSAMPLE", 8)
    results = diagnose_bottleneck(model, val_loader, device, default_down)
    
    # Analizza risultati
    bottleneck = analyze_results(results)
    
    # Suggerisci configurazione
    suggested_config = suggest_joint_training_config(bottleneck, results)
    
    # Salva risultati
    import json
    results_path = os.path.join(output_dir, "bottleneck_diagnosis.json")
    
    # Converti a liste per JSON
    results_json = {k: [float(x) for x in v] for k, v in results.items()}
    results_json["bottleneck"] = bottleneck
    results_json["suggested_config"] = suggested_config
    
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Risultati salvati in: {results_path}")


if __name__ == "__main__":
    main()