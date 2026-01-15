#!/bin/bash
#SBATCH --job-name=p2r_crowd
#SBATCH --account=did_crowd_counting_339   # ECCO IL NOME ESATTO (col 339 finale)
#SBATCH --partition=aiq                    # Usiamo la partizione AIQ (suggerita dal QOS)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=07:00:00                    # Limite massimo
#SBATCH --output=master_log_%j.txt
#SBATCH --error=master_err_%j.txt
#SBATCH --mail-user=c.romano50@studenti.unisa.it
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# --- CONFIGURAZIONE AMBIENTE ---
echo "⚙️  Caricamento moduli e ambiente..."
module load anaconda/3
source activate p2r_env

# Controllo GPU
nvidia-smi

#!/bin/bash
set -e  
#rm -rf logs
mkdir -p logs

echo "🚀 Avvio Stage 1 (ZIP)..."
#python3 train_stage1_zip.py  > logs/stage1.log 2>&1
echo "✅ Stage 1 completato!"

echo "🚀 Avvio Stage 2 (P2R)..."
#python3 train_stage2_p2r.py --config config.yaml > logs/stage2.log 2>&1
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