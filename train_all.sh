#!/bin/bash
set -e  # interrompe lo script se un comando fallisce
mkdir -p logs

echo "🚀 Avvio Stage 1 (ZIP)..."
python3 train_stage1_zip.py  > logs/stage1.log 2>&1
echo "✅ Stage 1 completato!"

echo "🚀 Avvio Stage 2 (P2R)..."
python3 train_stage2_p2r.py  > logs/stage2.log 2>&1
echo "✅ Stage 2 completato!"

echo "🚀 Avvio Stage 3 (JOINT)..."
python3 train_stage3_joint.py  > logs/stage3.log 2>&1
echo "✅ Tutti gli stadi completati!"