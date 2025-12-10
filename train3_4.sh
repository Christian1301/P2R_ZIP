echo "🚀 Avvio Stage 3 (JOINT)..."
python3 train_stage3_joint.py  > logs/stage3.log 2>&1
echo "✅ Stage 3 completato!"

echo "🚀 Avvio Stage 4 (RECOVERY)..."
python3 train_stage4_recovery.py  > logs/stage4.log 2>&1
echo "✅ Stage 4 completato!"
echo "🚀 Avvio Valutazioni..."

python3 evaluate_stage3.py > logs/ev_stage3.log 2>&1
echo "✅ Valutazione 3 completata!"

python3 evaluate_stage4.py > logs/ev_stage4.log 2>&1
echo "✅ Valutazione 4 completata!"