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

# --- INIZIO SCRIPT ---
set -e

# ATTENZIONE: Se riprendi il lavoro (RESUME), lascia commentato 'rm -rf logs'
# rm -rf logs     
mkdir -p lognewloss

echo "🚀 Avvio Stage 1 (ZIP)..."
# Se hai già finito lo stage 1, commenta la riga sotto:
python train_stage1_zip_focal.py --config config_newloss.yaml > lognewloss/stage1.log 2>&1
echo "✅ Stage 1 completato!"

python check_polarization.py --config config_newloss.yaml --checkpoint exp/shha_p2rzip_newloss/best_model.pth > lognewloss/check_polarization.log 2>&1

echo "🚀 Avvio Stage 2 (P2R)..."
# Resume attivo per sicurezza
python train_stage2_p2r_newloss.py --config config_newloss.yaml --points-format xy > lognewloss/stage2.log 2>&1
echo "✅ Stage 2 completato!"

echo "🚀 Avvio Stage 3 (JOINT)..."
python train_stage3_joint_newloss.py --config config_newloss.yaml --points-format xy > lognewloss/stage3.log 2>&1
echo "✅ Stage 3 completato!"

echo "🚀 Avvio Valutazioni..."
python evaluate_stage1_multithresh.py --ckpt exp/shha_p2rzip_newloss/best_model.pth --config config_newloss.yaml > lognewloss/ev_stage1.log 2>&1
python evaluate_stage2.py --config config_newloss.yaml > lognewloss/ev_stage2.log 2>&1
python evaluate_stage3.py --config config_newloss.yaml > lognewloss/ev_stage3.log 2>&1
python visualize_gating.py --config config_newloss.yaml > lognewloss/visualize_gating.log 2>&1

echo "🏆 TUTTO FINITO CON SUCCESSO!"