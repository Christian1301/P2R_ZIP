#!/bin/bash
#SBATCH --job-name=train_nwpu
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
mkdir -p logs_nwpu

CONFIG="config_nwpu.yaml"
echo "📋 Usando configurazione: $CONFIG"

echo "🚀 Avvio Stage 1 (ZIP)..."
python3 train_stage1_zip.py --config $CONFIG > logs_nwpu/stage1.log 2>&1
echo "✅ Stage 1 completato!"

echo "🚀 Avvio Stage 2 (P2R)..."
python3 train_stage2_p2r.py --config $CONFIG > logs_nwpu/stage2.log 2>&1
echo "✅ Stage 2 completato!"

echo "🚀 Avvio Stage 3 (JOINT)..."
python3 train_stage3_joint.py --config $CONFIG > logs_nwpu/stage3.log 2>&1
echo "✅ Stage 3 completato!"

echo "🚀 Avvio Valutazioni..."

python3 evaluate_stage1.py --config $CONFIG > logs_nwpu/ev_stage1.log 2>&1
echo "✅ Valutazione 1 completata!"

python3 evaluate_stage2.py --config $CONFIG > logs_nwpu/ev_stage2.log 2>&1
echo "✅ Valutazione 2 completata!"

python evaluate_stage3.py --tta --tta-flip-only --config $CONFIG > logs_nwpu/ev_stage3.log 2>&1
echo "✅ Valutazione 3 completata!"

python3 visualize_gating.py --config $CONFIG > logs_nwpu/visualize_gating.log 2>&1
echo "✅ Visualizzazione completata!"