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
mkdir -p logsb

echo "🚀 Avvio Stage 1 (ZIP)..."
# Se hai già finito lo stage 1, commenta la riga sotto:
python train_stage1_zip.py --config config_shhb.yaml > logsb/stage1.log 2>&1
echo "✅ Stage 1 completato!"

echo "🚀 Avvio Stage 2 (P2R)..."
# Resume attivo per sicurezza
python train_stage2_p2r.py --config config_shhb.yaml > logsb/stage2.log 2>&1
echo "✅ Stage 2 completato!"

echo "🚀 Avvio Stage 3 (JOINT)..."
python train_stage3_joint.py --config config_shhb.yaml > logsb/stage3.log 2>&1
echo "✅ Stage 3 completato!"

echo "🏆 TUTTO FINITO CON SUCCESSO!"