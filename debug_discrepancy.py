#!/usr/bin/env python3
"""Debug: confronta parametri training vs evaluation"""

import torch
import yaml

# Carica checkpoint
ckpt = torch.load('exp/shha_target60_v9/stage2_best.pth', map_location='cpu')

print("="*60)
print("🔍 DEBUG CHECKPOINT")
print("="*60)

# Chiavi disponibili
print(f"\nChiavi nel checkpoint: {list(ckpt.keys())}")

# MAE salvato
if 'mae' in ckpt:
    print(f"\nMAE salvato nel checkpoint: {ckpt['mae']:.2f}")
if 'best_mae' in ckpt:
    print(f"Best MAE: {ckpt['best_mae']:.2f}")

# log_scale
if 'model' in ckpt:
    state = ckpt['model']
elif 'model_state_dict' in ckpt:
    state = ckpt['model_state_dict']
else:
    state = ckpt

for key in state:
    if 'log_scale' in key:
        print(f"\n{key}: {state[key].item():.4f}")

# Config usato
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

print(f"\nP2R_DOWNSAMPLE in config: {cfg['DATA'].get('P2R_DOWNSAMPLE', 8)}")
print(f"ZIP_BLOCK_SIZE: {cfg['DATA']['ZIP_BLOCK_SIZE']}")
