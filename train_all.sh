#!/bin/bash
set -e  
rm -rf logs
mkdir -p logs

echo "🚀 Avvio Stage 1 (ZIP)..."
python3 train_stage1_zip.py  > logs/stage1.log 2>&1
echo "✅ Stage 1 completato!"

echo "🚀 Avvio Stage 2 (P2R)..."
#python3 train_stage2_p2r.py --config config.yaml --resume  > logs/stage2.log 2>&1
python3 train_stage2_v13.py --config config.yaml --dm-weight 0.5 --ot-weight 0.3 --epochs 5000 > logs/stage2.log 2>&1
echo "✅ Stage 2 completato!"

echo "🚀 Avvio Stage 3 (JOINT)..."
python3 train_stage3_joint.py  > logs/stage3.log 2>&1
echo "✅ Stage 3 completato!"

echo "🚀 Avvio Valutazioni..."

python3 evaluate_stage1.py > logs/ev_stage1.log 2>&1
echo "✅ Valutazione 1 completata!"

python3 evaluate_stage2.py > logs/ev_stage2.log 2>&1
echo "✅ Valutazione 2 completata!"

python3 tune_alpha.py > logs/tune_alpha.log 2>&1
echo "✅ Tuning alpha completato!"

python3 evaluate_stage3.py > logs/ev_stage3.log 2>&1
echo "✅ Valutazione 3 completata!"

python3 visualize_gating.py > logs/visualize_gating.log 2>&1
echo "✅ Visualizzazione completata!"